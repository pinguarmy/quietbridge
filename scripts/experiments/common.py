"""Shared helpers for the real-clip experiments (not a product module).

Segments come from a manifest (build_manifest.py); frames and anchors from the track_clips.py cache.
ROIs are made by the product's own code path: native frames -> `LipReader.resample` (25 fps grid)
-> `lipflow.face.mouth_rois`. Degradations are applied to the native frame list *before* that path,
the way a slow or stalling camera would deliver them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_SCRATCH = os.environ.get(
    "QB_EXP_SCRATCH",
    "/tmp/claude-0/-home-user-quietbridge/45ea6e31-5790-5734-a3d8-4ea60ec84630/scratchpad/r4-exp")
SESSIONS = {"A": "2016-03-12", "B": "2017-01-07"}


# ---------------------------------------------------------------- text + metrics
def norm(text: str) -> list[str]:
    """LRS3-style words: upper case, letters and apostrophes."""
    return re.sub(r"[^A-Z' ]+", " ", text.upper().replace("-", " ")).split()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def errors(ref: str, hyp: str) -> tuple[int, int]:
    r = norm(ref)
    return edit_distance(r, norm(hyp)), len(r)


def bootstrap(per_item: list[tuple[float, float]], n_boot: int = 2000, seed: int = 0):
    """Ratio statistic sum(num)/sum(den) with a percentile 95% CI, resampling segments."""
    arr = np.asarray(per_item, dtype=np.float64)
    if not len(arr):
        return {"value": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    num, den = arr[:, 0][idx].sum(1), arr[:, 1][idx].sum(1)
    stats = num / np.maximum(den, 1e-12)
    value = arr[:, 0].sum() / max(arr[:, 1].sum(), 1e-12)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"value": float(value), "lo": float(lo), "hi": float(hi), "n": int(len(arr))}


def fmt(ci: dict, pct: bool = True) -> str:
    if ci["value"] is None:
        return "n/a"
    k = 100 if pct else 1
    return f"{ci['value'] * k:.1f} [{ci['lo'] * k:.1f}, {ci['hi'] * k:.1f}]"


# ---------------------------------------------------------------- data
def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ClipCache:
    def __init__(self, cache_dir: str):
        self.grays = np.load(os.path.join(cache_dir, "grays.npy"), mmap_mode="r")
        self.anchors = np.load(os.path.join(cache_dir, "anchors.npy"))
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.fps = float(self.meta["fps"])

    def frame_range(self, start: float, end: float) -> list[int]:
        a = max(0, int(np.ceil(start * self.fps)))
        b = min(len(self.anchors) - 1, int(np.floor(end * self.fps)))
        return list(range(a, b + 1))


def degrade(idx: list[int], fps_native: float, cond: dict) -> list[int]:
    """Native frame indices that a degraded capture would have delivered."""
    idx = list(idx)
    if cond.get("trim_start"):
        idx = idx[int(round(cond["trim_start"] * fps_native)):]
    if cond.get("trim_end"):
        k = int(round(cond["trim_end"] * fps_native))
        idx = idx[:len(idx) - k] if k else idx
    if cond.get("fps"):
        t = np.asarray(idx) / fps_native
        grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / cond["fps"])
        keep = sorted(set(np.clip(np.searchsorted(t, grid - 0.5 / fps_native), 0, len(idx) - 1).tolist()))
        idx = [idx[i] for i in keep]
    if cond.get("keep"):
        # Irregular delivery: each native frame arrives with probability `keep` (0.67 of 29.97 fps ~ 20 fps
        # on average, intervals of 33/67/100 ms), seeded by the segment's first frame. First/last kept.
        rng = np.random.default_rng(10_000 + idx[0])
        mask = rng.random(len(idx)) < cond["keep"]
        mask[0] = mask[-1] = True
        idx = [i for i, k in zip(idx, mask) if k]
    if cond.get("gap"):
        t = np.asarray(idx) / fps_native
        mid = (t[0] + t[-1]) / 2
        g = cond["gap"] / 2
        idx = [i for i, x in zip(idx, t) if not (mid - g <= x < mid + g)]
    return idx


def rois_for_segment(clip: ClipCache, seg: dict, cond: dict | None = None):
    """Product path: degrade -> resample(ts) -> mouth_rois; crop shift/scale applied in aligned space."""
    from lipflow.face import mouth_rois
    from lipflow.vsr import LipReader
    import cv2

    cond = cond or {}
    idx = degrade(clip.frame_range(seg["start"], seg["end"]), clip.fps, cond)
    ts = [i / clip.fps for i in idx]
    if cond.get("jitter"):
        # Timestamp jitter (ms, Gaussian sd) on otherwise correct frames, as browser/camera clocks report
        # them; kept increasing. Only the resampler sees it.
        rng = np.random.default_rng(20_000 + idx[0])
        t = np.asarray(ts) + rng.normal(0.0, cond["jitter"] / 1000.0, len(ts))
        ts = np.maximum.accumulate(t).tolist()
    if cond.get("resample") == "nearest":  # alternative to the product's searchsorted-left resampling
        t = np.asarray(ts)
        grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / 25)
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        j = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
        sel = [idx[k] for k in j.tolist()]
    else:
        sel = [idx[j] for j in LipReader.resample(ts, len(ts))]
    anchors = [None if not np.isfinite(clip.anchors[i]).all() else clip.anchors[i].copy() for i in sel]
    frames = [np.asarray(clip.grays[i]) for i in sel]
    shift, scale = cond.get("shift", (0, 0)), cond.get("scale", 1.0)
    if shift == (0, 0) and scale == 1.0:
        return mouth_rois(frames, anchors), len(idx)
    big = mouth_rois(frames, anchors, crop=128)
    if big is None:
        return None, len(idx)
    size = int(round(96 * scale))
    cx, cy = 64 + shift[0], 64 + shift[1]
    out = []
    for p in big:
        q = p[cy - size // 2: cy - size // 2 + size, cx - size // 2: cx - size // 2 + size]
        out.append(q if size == 96 else cv2.resize(q, (96, 96), interpolation=cv2.INTER_LINEAR))
    return np.stack(out), len(idx)


def cond_key(cond: dict | None) -> str:
    return "base" if not cond else "_".join(f"{k}{v}" for k, v in sorted(cond.items())).replace(" ", "")


# ---------------------------------------------------------------- model
def make_reader(beam: int = 4, lm_path: str | None = None, lm_mode: str | None = None,
                personal_weight: float | None = None):
    """Product LipReader on CPU. With a personal LM: lm_mode "shared" (current default: the personal LM
    takes a share of the stock LM weight) or "additive" (the behaviour measured in Experiment 5)."""
    import torch
    torch.set_num_threads(int(os.environ.get("QB_EXP_THREADS", "2")))
    from lipflow.vsr import LipReader
    kw = {"beam_size": beam, "personal": False, "device": "cpu"}
    if lm_path:
        kw["personal_lm_path"] = lm_path
    if lm_mode:
        kw["personal_lm_mode"] = lm_mode
    if personal_weight is not None:
        kw["personal_weight"] = personal_weight
    return LipReader(**kw)


def encode_cached(reader, clip: ClipCache, seg: dict, cond: dict | None, cache_dir: str, tag: str = "stock"):
    """Encoder output for (segment, condition), cached on disk. Returns (enc, n_raw_frames, seconds)."""
    import torch
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}__{seg['id']}__{cond_key(cond)}.pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["enc"], d["raw"], d["sec"]
    rois, raw = rois_for_segment(clip, seg, cond)
    if rois is None:
        return None, raw, 0.0
    t0 = time.time()
    enc = reader.encode(rois)
    sec = time.time() - t0
    torch.save({"enc": enc, "raw": raw, "sec": sec, "frames": len(rois)}, path)
    return enc, raw, sec


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"head": run("git", "rev-parse", "HEAD"),
            "dirty": [l for l in run("git", "status", "--short").splitlines()]}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def seeded(seed: int = 0) -> random.Random:
    return random.Random(seed)

"""End-to-end check: the product Runtime with the REAL LipsAdapter, fed real video frames.

A fake browser camera pushes the clip's own frames (640x360 BGR, native 29.97 fps timestamps) into
`Runtime.receive_frame` on a virtual clock; Record/Stop are the runtime's own actions (mode "lips",
400 ms tail as in the product). The saved-phrase list is a profile's `lip_prompts`. Prints the state
fields the page reads (alternatives, phrase_matches, capture_warning, decision) and the text-free
metrics evidence, per segment. The virtual clock makes `release_to_text_ms` meaningless here; real
CPU time is in decode_ms / encode_ms / search_ms / phrase_scoring_ms.

    .venv/bin/python scripts/experiments/e2e_runtime.py --session B --segments B001 B004 B009 --out e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

NOW = [0]


def wait_for(predicate, timeout=600.0, poll=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    raise TimeoutError("runtime did not reach the expected state")


def read_frames(path, start, end):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    first = max(0, int(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for i in range(first, int(end * fps) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((i / fps, frame))
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--session", default="B")
    ap.add_argument("--segments", nargs="+", required=True)
    ap.add_argument("--keep-every", type=int, nargs="*", default=[],
                    help="per segment: deliver every k-th frame (1 = all frames; 2 = ~15 fps), default 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("QB_EXP_THREADS", "3")))
    import quietbridge.runtime as rt_mod
    from quietbridge.profiles import ProfileManager
    from model_offers import variants

    mani = {k: C.load_manifest(os.path.join(args.scratch, "manifest", f"{k}.jsonl")) for k in ("A", "B")}
    other = "A" if args.session == "B" else "B"
    by_id = {s["id"]: s for s in mani[args.session]}
    segs = [by_id[i] for i in args.segments]
    vocab = sorted({w for k in mani for s in mani[k] for w in s["reference"].split() if len(w) > 3})
    # Saved phrases: segment 1's sentence; segment 2 only via ONE one-word variant (sentence not saved);
    # segment 3's sentence AND its variants; plus every sentence of the other session (distractors).
    phrases = [segs[0]["reference"].capitalize()]
    if len(segs) > 1:
        phrases.append(variants(segs[1], vocab)[0][1].capitalize())
    if len(segs) > 2:
        phrases += [segs[2]["reference"].capitalize(), *(v.capitalize() for _, v in variants(segs[2], vocab)[:4])]
    phrases += [s["reference"].capitalize() for s in mani[other]]
    rt_mod.clock_ms = lambda: NOW[0]
    root = tempfile.mkdtemp(prefix="qb-e2e-", dir=os.path.dirname(os.path.abspath(args.out)))
    manager = ProfileManager(root_dir=root)
    profile = manager.create_profile("e2e", "E2E check", lip_prompts=phrases[:50])
    runtime = rt_mod.Runtime(browser_camera=True, profile=profile, profile_manager=manager)
    runtime.cleanup.submit = lambda *a, **k: False
    t0 = time.monotonic()
    runtime.start()
    wait_for(lambda: runtime.snapshot()["lips"] in {"ready", "unavailable"})
    state = runtime.snapshot()
    if state["lips"] != "ready":
        raise SystemExit(f"lips unavailable: {state['lip_error']}")
    report = {"git": C.git_state(), "model_load_s": time.monotonic() - t0, "phrases": len(phrases),
              "lip_model": state["lip_model"], "segments": []}
    runtime.action("mode", {"mode": "lips"})
    base_ms = 10_000
    clip = os.path.join(C.ROOT, segs[0]["clip"])
    for n, seg in enumerate(segs):
        keep = args.keep_every[n] if n < len(args.keep_every) else 1
        frames = read_frames(clip, seg["start"] - 0.4, seg["end"] + 0.45)
        frames = [f for k, f in enumerate(frames) if k % keep == 0]
        offset = base_ms - round(frames[0][0] * 1000)
        revision = runtime.snapshot()["revision"]
        started = stopped = False
        wall = time.monotonic()
        for t, frame in frames:
            ts = round(t * 1000) + offset
            NOW[0] = ts
            if not started and t >= seg["start"]:
                runtime.action("start", {})
                started = True
            if started and not stopped and t > seg["end"]:
                runtime.action("stop", {})
                stopped = True
                release = ts
            runtime.receive_frame(frame, ts)
            wait_for(lambda: runtime.lip_frames.qsize() <= 2, timeout=120, poll=0.002)
        if not stopped:
            runtime.action("stop", {})
            release = NOW[0]
        NOW[0] = release + 450
        wait_for(lambda: runtime.snapshot()["phase"] == "idle" and runtime.snapshot()["revision"] > revision + 1)
        s = runtime.snapshot()
        m = s["metrics"]
        entry = {"id": seg["id"], "reference": seg["reference"], "delivered_every": keep,
                 "reference_saved": any(C.norm(p) == C.norm(seg["reference"]) for p in phrases),
                 "decision": s["decision"], "candidates": s["candidates"], "alternatives": s["alternatives"],
                 "phrase_matches": s["phrase_matches"], "capture_warning": s["capture_warning"],
                 "lip_error": s["lip_error"],
                 "metrics": {k: m.get(k) for k in ("phrase_match", "phrase_scoring_ms", "capture_quality", "decode_ms",
                                                   "roi_ms", "encode_ms", "search_ms", "lip_capture_fps",
                                                   "lip_raw_frames", "lip_max_frame_gap_ms", "lip_dropped_frames")},
                 "wall_s": time.monotonic() - wall}
        report["segments"].append(entry)
        print(json.dumps({k: entry[k] for k in ("id", "candidates", "phrase_matches", "capture_warning")}), flush=True)
        print("   evidence:", json.dumps(entry["metrics"]["phrase_match"]), "scoring_ms", entry["metrics"]["phrase_scoring_ms"],
              "decode_ms", entry["metrics"]["decode_ms"], flush=True)
        base_ms = NOW[0] + 2000
    runtime.close()
    C.write_json(args.out, report)


if __name__ == "__main__":
    main()

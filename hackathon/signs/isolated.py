"""Experimental, local-only TorchScript isolated-sign adapter.

No weights ship with this adapter. A user-supplied manifest and its explicitly
hashed TorchScript artifact are required; scores are softmax scores, not measured
accuracy or calibrated confidence. This is a gesture-window recognizer, not a
continuous sign-language translator.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from hackathon.contracts import SignEvent

MAX_FRAMES = 240
MAX_CLIPS = 16
FRAME_MS = 1000 / 25
WINDOW_MS = 15 * FRAME_MS
STRIDE_MS = 8 * FRAME_MS


class IsolatedSignRecognizer:
    """Classify explicitly delimited BSL/ASL sign windows using local TorchScript."""

    windowed = True

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self._manifest = self._load_manifest(self.manifest_path)
        model_path = (self.manifest_path.parent / self._manifest["model_file"]).resolve()
        if model_path.parent != self.manifest_path.parent.resolve() or not model_path.is_file():
            raise ValueError("model_file must name an existing local file beside the manifest")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != self._manifest["model_sha256"]:
            raise ValueError("TorchScript model SHA-256 does not match manifest")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("isolated sign inference requires locally installed torch") from exc
        self._torch = torch
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()
        # Public shape matches fusion's label->entry contract. Model metadata is
        # not validation evidence, so this backend never promotes verified mappings.
        self.vocabulary = {}
        self._class_to_entry = {}
        for class_name, entry in self._manifest["vocabulary"].items():
            runtime_entry = dict(entry)
            runtime_entry["verified"] = False
            self.vocabulary[runtime_entry["label"]] = runtime_entry
            self._class_to_entry[class_name] = runtime_entry
        self.model_info = {
            "language": self._manifest["language"], "sha256": digest,
            "status": "experimental", "class_count": len(self._manifest["labels"]),
            "enabled_count": sum(bool(v["enabled"]) for v in self.vocabulary.values()),
            "source_url": self._manifest["source_url"], "license": self._manifest["license"],
        }
        self.last_diagnostics = {"outcome": "idle", "reason": "not_started"}
        self.last_tracking = []
        self._last_ts = None
        self._boundary = None
        self._frames = None
        self._raw_count = self._dropped = 0
        self._first_ts = self._last_frame_ts = None
        self._max_gap = 0

    @staticmethod
    def _load_manifest(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read isolated-sign manifest: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON object")
        expected = {"schema_version", "language", "model_file", "model_sha256", "labels", "input", "vocabulary", "threshold", "margin", "source_url", "license"}
        if not expected <= data.keys():
            raise ValueError(f"manifest missing keys: {sorted(expected - data.keys())}")
        if data["schema_version"] != 1 or data["language"] not in ("bsl", "asl"):
            raise ValueError("schema_version must be 1 and language bsl or asl")
        if not isinstance(data["model_file"], str) or Path(data["model_file"]).name != data["model_file"] or data["model_file"] in ("", ".", ".."):
            raise ValueError("model_file must be a relative filename beside the manifest")
        if not isinstance(data["model_sha256"], str) or len(data["model_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in data["model_sha256"]):
            raise ValueError("model_sha256 must be 64 lowercase hex characters")
        inp = data["input"]
        if inp != {"frames": 16, "size": 224, "fps": 25, "mean": [.5, .5, .5], "std": [1, 1, 1]}:
            raise ValueError("unsupported preprocessing; expected fixed I3D 16x224 RGB/25fps normalization")
        labels, vocab = data["labels"], data["vocabulary"]
        if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique nonempty ordered strings")
        if not isinstance(vocab, dict) or not set(vocab) <= set(labels):
            raise ValueError("vocabulary keys must be exact model class strings")
        seen = set()
        for key, item in vocab.items():
            if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"] or item["label"] in seen:
                raise ValueError("each vocabulary entry needs a unique nonempty runtime label")
            seen.add(item["label"])
            if not isinstance(item.get("phrase"), str) or not isinstance(item.get("matches"), list) or not all(isinstance(x, str) for x in item["matches"]):
                raise ValueError("vocabulary needs phrase and string matches")
            if not isinstance(item.get("enabled"), bool) or not isinstance(item.get("verified"), bool):
                raise ValueError("vocabulary enabled and verified must be booleans")
        for key in ("threshold", "margin"):
            if not isinstance(data[key], (int, float)) or not math.isfinite(data[key]) or not 0 <= data[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if not isinstance(data["source_url"], str) or not isinstance(data["license"], str):
            raise ValueError("source_url and license must be strings")
        return data

    def start(self, ts):
        self._ensure_open()
        ts = int(ts)
        if self._last_ts is not None and ts < self._last_ts or self._boundary is not None and ts < self._boundary:
            raise ValueError("start timestamp precedes previous input or window boundary")
        self._boundary, self._frames = ts, []
        self._raw_count = self._dropped = self._max_gap = 0
        self._first_ts = self._last_frame_ts = None
        self.last_diagnostics = {"outcome": "capturing", "reason": "window_started", "raw_frames": 0, "dropped_frames": 0, "input_fps": None, "max_gap_ms": 0, "clips": 0}

    def process(self, frame_bgr, ts):
        self._ensure_open()
        ts = int(ts)
        if self._last_ts is not None and ts <= self._last_ts:
            raise ValueError("frame timestamps must be strictly increasing")
        self._last_ts = ts
        self.last_tracking = []
        if self._frames is not None:
            if ts < self._boundary:
                raise ValueError("frame timestamp precedes active window")
            frame = np.asarray(frame_bgr)
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                raise ValueError("frame must be a nonempty HxWx3 BGR image")
            self._raw_count += 1
            if self._first_ts is None:
                self._first_ts = ts
            if self._last_frame_ts is not None:
                self._max_gap = max(self._max_gap, ts - self._last_frame_ts)
            self._last_frame_ts = ts
            if len(self._frames) < MAX_FRAMES:
                small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                self._frames.append((ts, small))
            else:
                self._dropped += 1
            self.last_diagnostics.update(self._capture_stats())
        return []

    def finish(self, ts):
        self._ensure_open()
        ts = int(ts)
        if self._frames is None:
            return []
        if ts < self._boundary or self._last_ts is not None and ts < self._last_ts:
            raise ValueError("end timestamp precedes window input")
        frames, start = self._frames, self._boundary
        self._frames = None
        self._boundary = ts
        stats = self._capture_stats(frames)
        self.last_diagnostics.update(stats)
        if self._dropped:
            return self._reject("frame_buffer_overflow")
        if len(frames) < 16:
            return self._reject("insufficient_frames")
        if frames[-1][0] - frames[0][0] < WINDOW_MS:
            return self._reject("insufficient_temporal_coverage")
        gaps = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
        if gaps and max(gaps) > 250:
            return self._reject("frame_gap_over_250ms")
        clips = self._make_clips(frames)
        if not clips:
            return self._reject("insufficient_temporal_coverage")
        self.last_diagnostics["clips"] = len(clips)
        logits = []
        for clip in clips[:MAX_CLIPS]:
            x = self._preprocess(clip)
            with self._torch.inference_mode():
                output = self._model(x)
            if isinstance(output, dict):
                output = output.get("logits")
            if not self._torch.is_tensor(output) or output.ndim != 2 or tuple(output.shape) != (1, len(self._manifest["labels"])):
                return self._reject("invalid_model_output")
            if not self._torch.isfinite(output).all():
                return self._reject("nonfinite_model_output")
            logits.append(output.detach().to(device="cpu", dtype=self._torch.float32)[0])
        probs = self._torch.softmax(self._torch.stack(logits).mean(dim=0), dim=0).numpy()
        order = np.argsort(probs)[::-1]
        top = int(order[0]); confidence = float(probs[top])
        classes = self._manifest["labels"]
        candidate = self._class_to_entry.get(classes[top])
        self.last_diagnostics.update({"top_class": classes[top], "score": confidence, "margin": confidence - float(probs[int(order[1])]) if len(order) > 1 else confidence})
        if candidate is None or not candidate["enabled"]:
            return self._reject("top_class_unmapped_or_disabled")
        if confidence < self._manifest["threshold"]:
            return self._reject("below_threshold")
        margin = confidence - float(probs[int(order[1])]) if len(order) > 1 else confidence
        if margin < self._manifest["margin"]:
            return self._reject("below_margin")
        self.last_diagnostics.update({"outcome": "accepted", "reason": "experimental_window_prediction"})
        return [SignEvent(candidate["label"], confidence, start, ts, True)]

    def _make_clips(self, frames):
        result = []
        first, last = frames[0][0], frames[-1][0]
        latest = max(first, last - WINDOW_MS)
        count = min(MAX_CLIPS, max(1, int((latest - first) / STRIDE_MS) + 1))
        onsets = np.linspace(first, latest, count) if count > 1 else [first]
        for onset in onsets:
            chosen = []
            for i in range(16):
                target = onset + i * FRAME_MS
                chosen.append(min(frames, key=lambda item: abs(item[0] - target))[1])
            result.append(chosen)
        return result

    def _preprocess(self, frames):
        arrays = []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            square = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            crop = square[16:240, 16:240].astype(np.float32) / 255.0 - .5
            arrays.append(crop)
        data = np.stack(arrays).transpose(3, 0, 1, 2).copy()
        return self._torch.from_numpy(data).unsqueeze(0).to(dtype=self._torch.float32)

    def _capture_stats(self, frames=None):
        duration = (self._last_frame_ts - self._first_ts) if self._first_ts is not None else 0
        return {
            "raw_frames": self._raw_count,
            "dropped_frames": self._dropped,
            "input_fps": (1000 * (self._raw_count - 1) / duration if duration > 0 else None),
            "max_gap_ms": self._max_gap,
        }

    def _reject(self, reason):
        self.last_diagnostics.update({"outcome": "rejected", "reason": reason})
        return []

    def cancel(self):
        if self._frames is not None:
            self._boundary = self._frames[-1][0] if self._frames else self._boundary
            self._frames = None
            self.last_diagnostics = {"outcome": "cancelled", "reason": "window_cancelled"}

    def _ensure_open(self):
        if self._model is None:
            raise RuntimeError("IsolatedSignRecognizer is closed")

    def close(self):
        self.cancel()
        self._model = None

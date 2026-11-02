"""Measure real mouth-capture timing without changing recognition input.

New recordings may include raw ``mouth_open`` and ``face_present`` arrays. Older
NPZs remain readable; unavailable activity measurements are null, not guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _milliseconds(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional sequence with at least two frames")
    return array


def analyze_capture(timestamps_ms, mouth_open=None, face_present=None) -> dict:
    timestamps = _milliseconds(timestamps_ms, "timestamps_ms")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_ms must be finite and strictly increasing")
    count = len(timestamps)
    duration_ms = float(timestamps[-1] - timestamps[0])
    gaps = np.diff(timestamps)
    result = {
        "raw_frames": count,
        "duration_ms": round(duration_ms, 2),
        "capture_fps": round((count - 1) * 1000 / duration_ms, 3) if duration_ms > 0 else None,
        "max_frame_gap_ms": round(float(gaps.max()), 2),
        "p95_frame_gap_ms": round(float(np.percentile(gaps, 95)), 2),
        "face_ratio": None,
        "leading_no_face_ms": None,
        "trailing_no_face_ms": None,
        "mouth_activity_range": None,
        "leading_low_activity_ms": None,
        "trailing_low_activity_ms": None,
    }
    if face_present is not None:
        face = np.asarray(face_present, dtype=bool)
        if face.shape != (count,):
            raise ValueError("face_present must align with timestamps_ms")
        result["face_ratio"] = round(float(face.mean()), 3)
        present = np.flatnonzero(face)
        if len(present):
            result["leading_no_face_ms"] = round(float(timestamps[present[0]] - timestamps[0]), 2)
            result["trailing_no_face_ms"] = round(float(timestamps[-1] - timestamps[present[-1]]), 2)
        else:
            result["leading_no_face_ms"] = result["trailing_no_face_ms"] = round(duration_ms, 2)
    if mouth_open is not None:
        activity = np.asarray(mouth_open, dtype=np.float64)
        if activity.ndim != 1 or len(activity) != count:
            raise ValueError("mouth_open must align with timestamps_ms")
        if not np.all(np.isfinite(activity)):
            raise ValueError("mouth_open must be finite")
        low, high = float(np.percentile(activity, 20)), float(np.percentile(activity, 80))
        spread = max(high - low, 0.0)
        result["mouth_activity_range"] = round(spread, 5)
        # This is a diagnostic relative threshold, not an automatic trim rule.
        # It helps compare recordings before any endpoint behavior is changed.
        threshold = low + 0.25 * spread
        active = np.flatnonzero(activity >= threshold)
        result["leading_low_activity_ms"] = round(float(timestamps[active[0]] - timestamps[0]), 2)
        result["trailing_low_activity_ms"] = round(float(timestamps[-1] - timestamps[active[-1]]), 2)
    return result


def analyze_npz(path) -> dict:
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        if "timestamps_ms" not in data.files:
            raise ValueError(f"{path}: missing timestamps_ms")
        return {
            "path": str(path),
            "text": str(data["text"].item()) if "text" in data.files else None,
            "session_id": str(data["session_id"].item()) if "session_id" in data.files else None,
            **analyze_capture(
                data["timestamps_ms"],
                data["mouth_open"] if "mouth_open" in data.files else None,
                data["face_present"] if "face_present" in data.files else None,
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps([analyze_npz(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()

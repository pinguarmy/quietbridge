"""Check one recorded try at the moment it finishes, before it is saved.

Few-shot personal models learn from a handful of tries, so one try with the
hands out of view or a frozen camera can poison a template. This module turns
the raw capture into measured values plus kind, specific warnings. It never
drops anything: the person decides "Save anyway" or "Discard and retry", the
measured values are stored with the saved try, and trainers *report* (not
exclude) flagged tries.

All thresholds are engineering defaults, not validated limits. They live in
``THRESHOLDS`` only, and every saved try records the thresholds version and
the measured values so they can be revisited against real recordings.

REST tries have their own expectations: resting hands may leave the view and
need not move, so the "hands out of view", "no movement" and "never returned
to rest" warnings do not apply to them.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

REST_LABEL = "REST"
THRESHOLDS_VERSION = "capture-quality-v1"

#: Engineering defaults; change them here only. Units are in the key names.
THRESHOLDS: dict[str, float] = {
    # Any try: fewer frames or a shorter span gives the model too little motion.
    "min_frames": 8,
    "min_duration_ms": 600.0,
    # A single gap this long means the camera or browser stalled mid-try;
    # resampling cannot recreate the missing motion.
    "max_frame_gap_ms": 300.0,
    # Gesture tries: share of frames with at least one hand visible.
    "min_hand_visible_ratio": 0.6,
    # Gesture tries: largest pose change from the try's median pose, in
    # hand sizes (wrist to middle-finger knuckle). Below this the hand barely moved.
    "min_movement_hand_sizes": 0.25,
    # Gesture tries: still moving at the end. Mean landmark speed over the last
    # window, as a share of the try's peak speed, while hands are still in view.
    "end_window_ms": 300.0,
    "max_end_speed_ratio": 0.5,
    # Lip tries: share of frames with the face (and so the mouth) found.
    "min_face_ratio": 0.8,
}

_WRIST, _MIDDLE_MCP = 0, 9


def _warning(code: str, message: str, measured: Any, limit: Any, severity: str = "warn") -> dict[str, Any]:
    # "warn" asks the person to decide before saving; "info" is shown, never blocks.
    return {"code": code, "severity": severity, "message": message, "measured": measured, "limit": limit}


def _timing(timestamps_ms) -> dict[str, Any]:
    ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    frames = int(ts.size)
    if frames < 2 or not np.all(np.isfinite(ts)):
        return {"frames": frames, "duration_ms": 0.0, "max_frame_gap_ms": None}
    gaps = np.diff(ts)
    return {"frames": frames, "duration_ms": round(float(ts[-1] - ts[0]), 1),
            "max_frame_gap_ms": round(float(gaps.max()), 1) if gaps.size else None}


def _timing_warnings(timing: dict[str, Any], t: dict[str, float]) -> list[dict[str, Any]]:
    warnings = []
    if timing["frames"] < t["min_frames"]:
        warnings.append(_warning(
            "too_few_frames",
            f"Only {timing['frames']} camera frames were captured. Try again and keep going a little longer.",
            timing["frames"], t["min_frames"]))
    elif timing["duration_ms"] < t["min_duration_ms"]:
        warnings.append(_warning(
            "too_short",
            f"This try lasted {timing['duration_ms'] / 1000:.1f} seconds. Leave a moment of stillness before and after.",
            timing["duration_ms"], t["min_duration_ms"]))
    gap = timing["max_frame_gap_ms"]
    if gap is not None and gap > t["max_frame_gap_ms"]:
        warnings.append(_warning(
            "large_frame_gap",
            f"The camera paused for {gap / 1000:.1f} seconds during this try, so part of the movement is missing.",
            gap, t["max_frame_gap_ms"]))
    return warnings


def _hand_scale(landmarks: np.ndarray, present: np.ndarray) -> float:
    sizes = [np.linalg.norm(landmarks[f, h, _WRIST, :2] - landmarks[f, h, _MIDDLE_MCP, :2])
             for f, h in zip(*np.nonzero(present))]
    sizes = [s for s in sizes if np.isfinite(s) and s > 1e-6]
    return float(np.median(sizes)) if sizes else math.nan


def gesture_quality(landmarks, hand_present, timestamps_ms, label: str,
                    thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Measure one gesture (or REST) try. Returns measured values and warnings.

    ``landmarks`` is (N, 2, 21, 3) raw camera-normalised coordinates,
    ``hand_present`` is (N, 2) and ``timestamps_ms`` is (N,).
    """
    t = {**THRESHOLDS, **(thresholds or {})}
    lm = np.asarray(landmarks, dtype=np.float64)
    hp = np.asarray(hand_present, dtype=bool)
    ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if lm.ndim != 4 or lm.shape[1:] != (2, 21, 3) or hp.shape != (lm.shape[0], 2) or ts.shape != (lm.shape[0],):
        raise ValueError("landmarks, hand_present and timestamps_ms must describe the same frames")
    is_rest = label == REST_LABEL
    timing = _timing(ts)
    any_hand = hp.any(axis=1)
    visible_ratio = float(any_hand.mean()) if any_hand.size else 0.0
    measured: dict[str, Any] = {**timing, "hand_visible_ratio": round(visible_ratio, 3),
                                "movement_hand_sizes": None, "end_speed_ratio": None}
    warnings = _timing_warnings(timing, t)

    scale = _hand_scale(lm, hp)
    if any_hand.any() and np.isfinite(scale):
        # Pose per frame: mean landmark position of the visible hands (x, y only).
        movement = []
        for h in range(2):
            idx = np.flatnonzero(hp[:, h])
            if idx.size < 2:
                continue
            xy = lm[idx, h, :, :2]
            median_pose = np.median(xy, axis=0)
            movement.append(float(np.max(np.linalg.norm(xy - median_pose, axis=2).mean(axis=1))) / scale)
        if movement:
            measured["movement_hand_sizes"] = round(max(movement), 3)
        # Speed per frame interval of the hand(s) visible in both frames.
        speeds = np.full(max(len(ts) - 1, 0), np.nan)
        for f in range(len(ts) - 1):
            both = hp[f] & hp[f + 1]
            dt = ts[f + 1] - ts[f]
            if both.any() and dt > 0:
                step = np.linalg.norm(lm[f + 1, both, :, :2] - lm[f, both, :, :2], axis=2).mean()
                speeds[f] = step / scale / (dt / 1000.0)
        finite = speeds[np.isfinite(speeds)]
        if finite.size >= 3 and float(finite.max()) > 0:
            end_start = ts[-1] - t["end_window_ms"]
            tail = speeds[(ts[1:] >= end_start) & np.isfinite(speeds)]
            if tail.size:
                measured["end_speed_ratio"] = round(float(tail.mean() / finite.max()), 3)

    if not is_rest:
        if visible_ratio < t["min_hand_visible_ratio"]:
            warnings.append(_warning(
                "hands_out_of_view",
                f"Your hands were in view for {round(visible_ratio * 100)}% of this try. "
                "Keep the whole movement inside the camera view.",
                round(visible_ratio, 3), t["min_hand_visible_ratio"]))
        moved = measured["movement_hand_sizes"]
        if moved is not None and moved < t["min_movement_hand_sizes"]:
            warnings.append(_warning(
                "no_movement",
                "Your hand hardly moved in this try. Make the movement once, clearly, between two still moments.",
                moved, t["min_movement_hand_sizes"]))
        end_ratio = measured["end_speed_ratio"]
        if end_ratio is not None and end_ratio > t["max_end_speed_ratio"]:
            warnings.append(_warning(
                "no_return_to_rest",
                "Your hand was still moving when the try finished. Finish the movement and pause before saving.",
                end_ratio, t["max_end_speed_ratio"]))
    elif visible_ratio == 0.0:
        # Fine for staying silent, but it teaches the model nothing about rest.
        warnings.append(_warning(
            "rest_no_hands",
            "No hands were visible in this resting try. That is fine for staying silent, but a resting try "
            "with your hands relaxed in view teaches the app more.",
            0.0, None, severity="info"))
    return {"kind": "gesture", "label": label, "rest": is_rest, "version": THRESHOLDS_VERSION,
            "thresholds": t, "measured": measured, "warnings": warnings}


def lip_quality(timestamps_ms, face_present, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Measure one lip try from its capture timestamps and face detections.

    Mouth movement amount is not checked: the opening signal has no validated
    unit across faces and cameras, so a threshold on it would be a guess.
    """
    t = {**THRESHOLDS, **(thresholds or {})}
    ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    face = np.asarray(face_present, dtype=bool).reshape(-1)
    if face.shape != ts.shape:
        raise ValueError("face_present must align with timestamps_ms")
    timing = _timing(ts)
    face_ratio = float(face.mean()) if face.size else 0.0
    measured = {**timing, "face_ratio": round(face_ratio, 3)}
    warnings = _timing_warnings(timing, t)
    if face_ratio < t["min_face_ratio"]:
        warnings.append(_warning(
            "mouth_not_visible",
            f"Your mouth was found in {round(face_ratio * 100)}% of this try. Face the camera with your mouth "
            "uncovered and well lit.",
            round(face_ratio, 3), t["min_face_ratio"]))
    return {"kind": "lips", "version": THRESHOLDS_VERSION, "thresholds": t,
            "measured": measured, "warnings": warnings}


def combine(*reports: dict[str, Any] | None) -> dict[str, Any]:
    """One review for a paired try: every part's warnings, nothing hidden."""
    parts = [r for r in reports if r]
    return {"version": THRESHOLDS_VERSION, "parts": parts,
            "warnings": [w for r in parts for w in r["warnings"]]}


def with_decision(report: dict[str, Any], decision: str) -> dict[str, Any]:
    """Record how the person resolved the review ("no_warnings" or "saved_anyway")."""
    if decision not in ("no_warnings", "saved_anyway", "not_reviewed"):
        raise ValueError(f"unknown quality decision {decision!r}")
    return {**report, "decision": decision}


def to_npz_field(report: dict[str, Any]) -> np.ndarray:
    """Scalar JSON string, loadable with allow_pickle=False."""
    return np.asarray(json.dumps(report, sort_keys=True, allow_nan=False))


def from_npz(data) -> dict[str, Any] | None:
    """Read the stored review from an opened NPZ, or None for older recordings."""
    if "quality_json" not in getattr(data, "files", ()):
        return None
    try:
        value = json.loads(str(data["quality_json"]))
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def needs_review(report: dict[str, Any] | None) -> bool:
    """True when at least one warning asks the person to decide before saving."""
    return any(isinstance(w, dict) and w.get("severity", "warn") == "warn"
               for w in (report or {}).get("warnings", []))


def warning_codes(report: dict[str, Any] | None) -> list[str]:
    return [w.get("code", "") for w in (report or {}).get("warnings", []) if isinstance(w, dict)]

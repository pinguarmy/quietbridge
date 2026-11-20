"""Capture-quality review at the moment a try finishes (synthetic landmark sequences).

Each warning has a sequence built to trigger it and nothing else; a clean
gesture try and a still REST try produce no warning that asks for a decision.
Thresholds are engineering defaults; these tests check the rules, not that the
defaults suit real cameras.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from quietbridge.signs import capture_quality as cq

DT = 40.0  # 25 fps


def _hand(center, size=0.1):
    lm = np.zeros((21, 3))
    lm[:, 0] = center[0] + np.linspace(0, 0.3 * size, 21)
    lm[:, 1] = center[1] + np.linspace(0, size, 21)
    lm[0, :2] = center
    lm[9, :2] = (center[0], center[1] + size)  # hand size = size
    return lm


def gesture_try(n=40, move=0.3, visible=None, settle=True, gap_at=None, gap_ms=0.0):
    """Hold still, move once (``move`` in screen units), and settle at the end."""
    t = np.linspace(0, 1, n)
    if settle:
        phase = np.clip((t - 0.2) / 0.6, 0, 1)  # still for the first and last 20 %
        offset = move * (1 - np.cos(np.pi * phase)) / 2
    else:
        offset = move * t  # still moving at full speed when the try ends
    lm = np.full((n, 2, 21, 3), np.nan)
    hp = np.zeros((n, 2), dtype=bool)
    for f in range(n):
        if visible is not None and not visible[f]:
            continue
        lm[f, 1] = _hand((0.4 + offset[f], 0.5))
        hp[f, 1] = True
    ts = np.arange(n) * DT
    if gap_at is not None:
        ts[gap_at:] += gap_ms
    return lm, hp, ts


def codes(report):
    return [w["code"] for w in report["warnings"]]


def test_clean_gesture_try_has_no_warnings():
    report = cq.gesture_quality(*gesture_try(), "HELLO")
    assert codes(report) == []
    assert not cq.needs_review(report)
    measured = report["measured"]
    assert measured["frames"] == 40 and measured["hand_visible_ratio"] == 1.0
    assert measured["movement_hand_sizes"] > cq.THRESHOLDS["min_movement_hand_sizes"]
    assert measured["end_speed_ratio"] <= cq.THRESHOLDS["max_end_speed_ratio"]
    assert report["version"] == cq.THRESHOLDS_VERSION and report["thresholds"] == cq.THRESHOLDS


def test_too_few_frames():
    report = cq.gesture_quality(*gesture_try(n=5), "HELLO")
    assert "too_few_frames" in codes(report) and cq.needs_review(report)


def test_too_short_but_enough_frames():
    lm, hp, ts = gesture_try(n=12)
    report = cq.gesture_quality(lm, hp, ts / 2, "HELLO")  # 12 frames over 220 ms
    assert "too_short" in codes(report) and "too_few_frames" not in codes(report)


def test_large_frame_gap():
    report = cq.gesture_quality(*gesture_try(gap_at=20, gap_ms=600), "HELLO")
    assert codes(report) == ["large_frame_gap"]
    assert report["warnings"][0]["measured"] == pytest.approx(640.0)


def test_hands_out_of_view():
    visible = np.zeros(40, dtype=bool)
    visible[:15] = True
    report = cq.gesture_quality(*gesture_try(visible=visible), "HELLO")
    assert "hands_out_of_view" in codes(report)
    assert report["measured"]["hand_visible_ratio"] == pytest.approx(15 / 40)


def test_no_movement_in_a_gesture_try():
    report = cq.gesture_quality(*gesture_try(move=0.0), "HELLO")
    assert codes(report) == ["no_movement"]


def test_gesture_that_never_returns_to_rest():
    report = cq.gesture_quality(*gesture_try(settle=False), "HELLO")
    assert codes(report) == ["no_return_to_rest"]


def test_rest_tries_have_their_own_expectations():
    # Still hands are exactly what a resting try should be.
    assert codes(cq.gesture_quality(*gesture_try(move=0.0), cq.REST_LABEL)) == []
    # Hands may leave the view while resting; with none at all, an informational note only.
    visible = np.zeros(40, dtype=bool)
    report = cq.gesture_quality(*gesture_try(visible=visible), cq.REST_LABEL)
    assert codes(report) == ["rest_no_hands"] and not cq.needs_review(report)
    # Timing problems still apply to resting tries.
    assert "too_few_frames" in codes(cq.gesture_quality(*gesture_try(n=4, move=0.0), cq.REST_LABEL))


def test_lip_quality_warnings_and_clean_path():
    ts = np.arange(30) * DT
    assert codes(cq.lip_quality(ts, np.ones(30, dtype=bool))) == []
    face = np.ones(30, dtype=bool)
    face[:10] = False
    report = cq.lip_quality(ts, face)
    assert codes(report) == ["mouth_not_visible"]
    assert report["measured"]["face_ratio"] == pytest.approx(20 / 30, abs=1e-3)
    assert "too_few_frames" in codes(cq.lip_quality(ts[:5], np.ones(5, dtype=bool)))
    gapped = ts.copy()
    gapped[15:] += 500
    assert codes(cq.lip_quality(gapped, np.ones(30, dtype=bool))) == ["large_frame_gap"]


def test_combined_review_keeps_every_part_and_round_trips_through_npz(tmp_path):
    gesture = cq.gesture_quality(*gesture_try(move=0.0), "HELLO")
    lips = cq.lip_quality(np.arange(30) * DT, np.zeros(30, dtype=bool))
    combined = cq.combine(gesture, lips)
    assert codes(combined) == ["no_movement", "mouth_not_visible"]
    stored = cq.with_decision(gesture, "saved_anyway")
    path = tmp_path / "t.npz"
    np.savez(path, quality_json=cq.to_npz_field(stored))
    with np.load(path, allow_pickle=False) as data:
        assert cq.from_npz(data) == json.loads(json.dumps(stored))
    np.savez(path, other=np.array(1))
    with np.load(path, allow_pickle=False) as data:
        assert cq.from_npz(data) is None  # older recordings: unknown, not "clean"
    with pytest.raises(ValueError):
        cq.with_decision(gesture, "silently_dropped")


def test_mismatched_arrays_are_rejected():
    lm, hp, ts = gesture_try()
    with pytest.raises(ValueError):
        cq.gesture_quality(lm, hp[:-1], ts, "HELLO")
    with pytest.raises(ValueError):
        cq.lip_quality(ts, np.ones(3, dtype=bool))

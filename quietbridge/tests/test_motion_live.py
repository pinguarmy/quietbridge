"""Live motion wiring contracts with fakes: no camera or model inference required.

The former sliding-window adapter and its fixed hackathon phrase table were
removed; the custom motion path is SignRecognizer's explicit window (and the
opt-in motion_auto segmenter), covered in quietbridge/signs/tests.
"""
import queue
from unittest.mock import Mock, patch

import numpy as np
import pytest

from quietbridge.contracts import SignEvent
from quietbridge.runtime import Runtime

CUSTOM = {"DDE": {"display_name": "Dee", "phrase": "Dee", "matches": ["Dee"],
                  "enabled": True, "verified": False}}


def hand():
    return {"handedness": "Right", "landmarks": np.full((21, 3), .4).tolist()}


def test_motion_model_without_vocabulary_fails_visibly(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/personal.npz")
    with pytest.raises(ValueError, match="no vocabulary.*--vocabulary"):
        Runtime(enable_lips=False)
    # Explicit constructor argument follows the same rule.
    monkeypatch.delenv("QUIETBRIDGE_MOTION_MODEL")
    with pytest.raises(ValueError, match="no vocabulary"):
        Runtime(enable_lips=False, motion_model="/local/personal.npz")


def test_no_fixed_demo_phrase_table_remains():
    import quietbridge.runtime as runtime_module
    assert not hasattr(runtime_module, "MOTION_PHRASES")
    assert not hasattr(runtime_module, "_MotionWindow")


def test_opt_in_custom_vocabulary_needs_confirmation(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/personal.npz")
    runtime = Runtime(enable_lips=False, vocabulary=CUSTOM)
    try:
        assert runtime.state["gesture_model"] == "personal motion-v2 (experimental)"
        assert "POINTING_UP" not in runtime.vocabulary
        runtime.receive_signs([SignEvent("DDE", .8, 1000, 4000, True)], now=4000)
        assert runtime.state["decision"]["text"] == "Dee"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
        # A label outside the explicit vocabulary contributes no text.
        runtime.receive_signs([SignEvent("HELLO", .9, 4100, 4500, True)], now=4500)
        assert runtime.state["decision"]["text"] == "Dee"
    finally:
        runtime.close()


def test_worker_routes_custom_motion_events(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/personal.npz")
    runtime = Runtime(enable_lips=False, vocabulary=CUSTOM)
    # SignRecognizer owns motion classification in the merged architecture.
    detector = Mock(last_tracking=[hand()])
    detector._motion = object()
    detector.motion_auto = True
    detector._motion_last_boundary_ms = None
    detector.process.side_effect = [[SignEvent("DDE", .8, 1000, 4000, True)], []]
    packets = [(np.zeros((10, 10, 3), np.uint8), ts) for ts in range(1000, 6100, 100)]
    def get(timeout):
        if packets:
            return packets.pop(0)
        runtime.stop.set()
        raise queue.Empty
    runtime.hand_frames.get = get
    runtime.receive_signs = Mock()
    try:
        with patch("quietbridge.signs.recognizer.SignRecognizer", return_value=detector):
            runtime._hands()
        events = [event for call in runtime.receive_signs.call_args_list for event in call.args[0]]
        assert len(events) == 1 and events[0].label == "DDE"
        assert not runtime.state["sign_error"]
    finally:
        runtime.close()

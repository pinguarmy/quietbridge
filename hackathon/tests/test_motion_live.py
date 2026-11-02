"""Live motion wiring tests: no camera or model inference required."""
import queue
from unittest.mock import Mock, patch

import numpy as np

from hackathon.contracts import SignEvent
from hackathon.runtime import Runtime, _MotionWindow


def hand():
    return {"handedness": "Right", "landmarks": np.full((21, 3), .4).tolist()}


def test_window_schema_cadence_bounds_and_rearm():
    model = Mock()
    model.predict.return_value = ("DDE", .8)
    window = _MotionWindow(model)
    emitted = []
    for ts in range(0, 8100, 100):
        emitted.extend(window.update([hand()], ts))
    assert len(emitted) == 1 and emitted[0].label == "DDE"
    assert model.predict.call_count == 6
    trial = model.predict.call_args.args[0]
    assert trial["landmarks"].shape == (41, 2, 21, 3)
    assert trial["hand_present"].shape == (41, 2)
    assert trial["handedness"][0, 0] == "Right"
    assert len(window.frames) <= 240
    assert window.frames[-1][0] - window.frames[0][0] <= 4000
    model.predict.return_value = (None, 0)
    for ts in range(8100, 10200, 100):
        assert not window.update([], ts)
    model.predict.return_value = ("DDE", .8)
    for ts in range(10200, 12300, 100):
        emitted.extend(window.update([hand()], ts))
    assert len(emitted) == 2


def test_low_score_unknown_and_rest_do_not_emit():
    for prediction in [("OPENAI", .74), ("REST", 1), ("POINTING_UP", 1), (None, 0)]:
        model = Mock()
        model.predict.return_value = prediction
        window = _MotionWindow(model)
        assert not [event for ts in range(0, 6000, 100)
                    for event in window.update([hand()], ts)]


def test_opt_in_custom_vocabulary_needs_confirmation(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/personal.npz")
    runtime = Runtime(enable_lips=False)
    try:
        assert runtime.state["gesture_model"] == "personal motion-v2 (experimental)"
        assert "POINTING_UP" not in runtime.vocabulary
        runtime.receive_signs([SignEvent("DDE", .8, 1000, 4000, True)], now=4000)
        assert runtime.state["decision"]["text"] == "DEE"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
    finally:
        runtime.close()


def test_worker_routes_custom_motion_events(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/personal.npz")
    runtime = Runtime(enable_lips=False)
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
        with patch("hackathon.signs.recognizer.SignRecognizer", return_value=detector):
            runtime._hands()
        events = [event for call in runtime.receive_signs.call_args_list for event in call.args[0]]
        assert len(events) == 1 and events[0].label == "DDE"
        assert not runtime.state["sign_error"]
    finally:
        runtime.close()

"""Cross-branch runtime contracts for sessions and explicit motion."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hackathon.tests.test_motion_integration import MotionRecognizer, wait_for

from hackathon.contracts import LipResult, SignEvent
from hackathon.runtime import Runtime

VOCAB = {"CUSTOM": {"phrase": "Custom", "matches": ["Custom"],
                    "enabled": True, "verified": True}}


def test_explicit_profile_vocabulary_wins_over_deployed_motion(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", "/local/model.npz")
    runtime = Runtime(vocabulary=VOCAB)
    try:
        assert runtime.vocabulary == VOCAB
        assert runtime.motion_model_path == "/local/model.npz"
    finally:
        runtime.close()


def test_session_stop_preserves_pending_motion_and_its_window_evidence():
    runtime = Runtime(browser_camera=True, vocabulary=VOCAB)
    runtime.cleanup.submit = Mock(return_value=False)
    try:
        runtime.action("camera_session", {"active": True})
        runtime.state.update(camera="ready", lips="ready", signs="ready", sign_input="motion")
        with patch("hackathon.runtime.clock_ms", return_value=1000):
            runtime.action("start", {})
        runtime.receive_signs([SignEvent("CUSTOM", .9, 1100, 1200, True)], now=1200)
        with patch("hackathon.runtime.clock_ms", return_value=3000):
            runtime.action("camera_session", {"active": False})
        assert len(runtime.events) == 1
        assert list(runtime.hand_control.queue)[-1] == ("finish", 3000, runtime.record_generation)
        with patch("hackathon.runtime.clock_ms", return_value=3000):
            runtime._publish_session(LipResult(["Custom"], 20, raw_frames=40, capture_fps=20))
        assert runtime.state["session_text"] == "Custom"
        assert runtime.state["decision"]["text"] == "Custom"
        assert runtime.state["metrics"]["lip_capture_fps"] == 20
    finally:
        runtime.close()


def test_custom_motion_requires_hold_but_lips_and_motion_auto_remain_automatic():
    runtime = Runtime(browser_camera=True, vocabulary=VOCAB)
    try:
        runtime.action("camera_session", {"active": True})
        runtime.state.update(camera="ready", lips="ready", signs="ready", sign_input="motion")
        runtime._auto_frame(.1, 1000)
        assert runtime.state["phase"] == "idle"
        assert not any(command[0] == "start" for command in runtime.hand_control.queue)
        runtime.action("mode", {"mode": "lips"})
        runtime._auto_frame(.1, 1100)
        assert runtime.state["phase"] == "recording"
        assert not runtime._window_motion
        runtime.action("cancel", {})
        runtime.action("mode", {"mode": "combined"})
        runtime.state["sign_input"] = "motion_auto"
        runtime._auto_frame(.1, 1200)
        assert runtime.state["phase"] == "recording"
        assert not runtime._window_motion
    finally:
        runtime.close()


def test_window_conflict_stays_visible_after_camera_stop():
    vocabulary = {"WATER": {"phrase": "water", "matches": ["water"], "enabled": True,
                              "verified": False, "intent_template": "I need {gesture}"}}
    runtime = Runtime(browser_camera=True, vocabulary=vocabulary)
    runtime.cleanup.submit = Mock(return_value=False)
    try:
        runtime.action("camera_session", {"active": True})
        runtime.state.update(camera="ready", lips="ready", signs="ready", sign_input="motion")
        with patch("hackathon.runtime.clock_ms", return_value=1000):
            runtime.action("start", {})
        runtime.finish_at = 2000
        runtime.state["phase"] = "idle"
        event = SignEvent("WATER", .9, 1100, 1800, True)
        runtime._publish_session(LipResult(["I do not need water", "I need"], 1), [event])
        assert runtime.state["decision"]["reason"] == "conflict"
        assert runtime.state["decision"]["text"] == "I do not need water"
        runtime.action("camera_session", {"active": False})
        assert runtime.state["decision"]["reason"] == "conflict"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
        assert "I need" in runtime.state["alternatives"]
        assert "water" in runtime.state["alternatives"]
        runtime.action("camera_session", {"active": True})
        assert not runtime._session_conflict
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["combined", "signs"])
def test_custom_session_hold_survives_camera_stop(monkeypatch, mode):
    recognizer = MotionRecognizer()
    real_import = __import__("importlib").import_module
    monkeypatch.setattr("hackathon.runtime.importlib.import_module",
                        lambda name: SimpleNamespace(SignRecognizer=lambda: recognizer)
                        if name == "hackathon.signs.recognizer" else real_import(name))
    lip_started = threading.Event()
    class Lips:
        recording = None
        def __init__(self, *args, **kwargs): pass
        def start(self, *args): self.recording = True; lip_started.set()
        def process(self, *args): pass
        def finish(self):
            self.recording = None
            return LipResult(["other", "hello"], 1)
        def close(self): pass
    monkeypatch.setattr("hackathon.lips_adapter.LipsAdapter", Lips)
    vocab = {"HELLO": {"phrase": "Hello.", "matches": ["hello"], "enabled": True, "verified": False}}
    runtime = Runtime(browser_camera=True, vocabulary=vocab)
    for worker in (runtime._hands, runtime._lips):
        thread = threading.Thread(target=worker, daemon=True)
        runtime.threads.append(thread)
        thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion" and runtime.state["lips"] == "ready")
        runtime.action("camera_session", {"active": True})
        runtime.state["camera"] = "ready"
        runtime.action("mode", {"mode": mode})
        runtime.action("start", {})
        wait_for(lambda: bool(recognizer.started))
        if mode == "combined":
            assert lip_started.wait(2)
        runtime.action("stop", {})
        wait_for(lambda: runtime.state["phase"] == "idle" and bool(runtime.state["session_text"]))
        expected = "hello" if mode == "combined" else "Hello."
        assert runtime.state["session_text"] == expected
        if mode == "combined":
            assert "other" in runtime.state["alternatives"]
        runtime.action("camera_session", {"active": False})
        assert runtime.state["decision"]["text"] == expected
        if mode == "combined":
            assert "other" in runtime.state["alternatives"]
    finally:
        runtime.close()

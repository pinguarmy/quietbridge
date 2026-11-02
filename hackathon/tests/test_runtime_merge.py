"""Cross-branch runtime contracts for sessions and explicit motion."""
from unittest.mock import Mock, patch

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
        assert runtime.state["decision"]["reason"] == "agreement"
        assert runtime.state["metrics"]["lip_capture_fps"] == 20
    finally:
        runtime.close()


def test_automatic_lip_chunks_do_not_implicitly_start_motion_window():
    runtime = Runtime(browser_camera=True, vocabulary=VOCAB)
    try:
        runtime.action("camera_session", {"active": True})
        runtime.state.update(camera="ready", lips="ready", signs="ready", sign_input="motion")
        runtime._auto_frame(.1, 1000)
        assert runtime.state["phase"] == "recording"
        assert not runtime._window_motion
        assert not any(command[0] == "start" for command in runtime.hand_control.queue)
    finally:
        runtime.close()

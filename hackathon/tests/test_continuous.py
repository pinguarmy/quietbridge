"""Continuous camera-session boundaries; synthetic timing, no recognition claims."""
import threading
from unittest.mock import patch

import numpy as np
import pytest

from hackathon.contracts import LipResult
from hackathon.runtime import Runtime


@pytest.fixture
def runtime():
    r = Runtime(browser_camera=True, vocabulary={})
    r.state.update(lips="ready", camera="ready")
    yield r
    r.close()


def activate(r):
    r.action("camera_session", {"active": True})


def open_clip(r):
    with patch("hackathon.runtime.clock_ms", return_value=1300):
        for timestamp in (1000, 1100, 1300):
            r._auto_frame(.08, timestamp)
    assert r.state["phase"] == "recording"


def test_session_does_not_start_before_first_ready_camera_frame(runtime):
    runtime.state["camera"] = "waiting"
    activate(runtime)
    for timestamp in (1000, 1200, 1400):
        runtime._auto_frame(.08, timestamp)
    assert runtime.record_start is None
    runtime.state["camera"] = "ready"
    open_clip(runtime)


def test_missing_face_waits_but_visible_closed_mouth_starts_session_capture(runtime):
    activate(runtime)
    runtime._auto_frame(None, 1000)
    assert runtime.state["phase"] == "idle"
    runtime._auto_frame(0, 1100)
    assert runtime.state["phase"] == "recording"
    assert runtime.state["auto_recognition"] is True


def test_quiet_does_not_end_computation_chunk_or_session(runtime):
    activate(runtime)
    open_clip(runtime)
    with patch("hackathon.runtime.clock_ms", return_value=2500):
        runtime._auto_frame(.01, 1800)
        runtime._auto_frame(.01, 2400)  # minimum clip duration not reached
        assert runtime.state["phase"] == "recording"
        runtime._auto_frame(.01, 2500)
    assert runtime.state["phase"] == "recording"
    assert runtime.finish_at is None
    assert runtime.state["auto_recognition"] is True


def test_face_loss_does_not_end_recording_but_six_second_compute_cap_is_finite(runtime):
    activate(runtime)
    open_clip(runtime)
    with patch("hackathon.runtime.clock_ms", return_value=2600):
        runtime._auto_frame(None, 1900)
        runtime._auto_frame(None, 2600)
    assert runtime.state["phase"] == "recording"
    runtime.action("cancel", {})
    open_clip(runtime)
    with patch("hackathon.runtime.clock_ms", return_value=7300):
        runtime._auto_frame(.09, 7300)
    assert runtime.state["phase"] == "tail"


def test_stop_camera_flushes_pending_clip_and_ignores_late_uploads(runtime):
    activate(runtime)
    open_clip(runtime)
    generation = runtime._cleanup_generation
    revision = runtime.revision
    runtime.action("camera_session", {"active": False})
    assert runtime.state["phase"] == "tail"
    assert runtime.record_start is not None
    assert runtime.state["session_finalizing"] is True
    assert runtime.state["auto_recognition"] is False
    assert runtime.state["camera"] == "waiting"
    assert runtime.tracking == {"hands": [], "mouth": None}
    assert runtime._cleanup_generation == generation
    assert not runtime.receive_frame(np.zeros((2, 2, 3), dtype=np.uint8), 8000)
    runtime._auto_frame(.09, 9000)
    assert runtime.state["phase"] == "tail"


def test_manual_protocol_stays_supported_without_camera_session(runtime):
    runtime.action("start", {})
    assert runtime.state["phase"] == "recording"
    assert not runtime.state["auto_recognition"]
    runtime.action("stop", {})
    assert runtime.state["phase"] == "tail"
    runtime.action("cancel", {})
    assert runtime.state["phase"] == "idle"


def test_sessions_only_browser_live_and_boolean(runtime):
    with pytest.raises(ValueError):
        runtime.action("camera_session", {"active": "yes"})
    r = Runtime(rehearsal=True, vocabulary={})
    try:
        with pytest.raises(ValueError):
            r.action("camera_session", {"active": True})
    finally:
        r.close()


def test_normalized_mouth_ring_is_only_a_segmentation_hint():
    inner = [[.5, .5] for _ in range(20)]
    outer = [[.5, .5] for _ in range(20)]
    outer[0], outer[10] = [.4, .5], [.6, .5]
    inner[5], inner[15] = [.5, .49], [.5, .51]
    assert Runtime._mouth_open({"inner_lips": inner, "outer_lips": outer}) == pytest.approx(.075)
    assert Runtime._mouth_open(None) is None


def test_stop_camera_preserves_result_of_already_running_decode(runtime):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Adapter:
        last_tracking = None
        recording = None
        def start(self):
            self.recording = object()
        def process(self, frame, timestamp):
            pass
        def finish(self):
            entered.set()
            assert release.wait(3)
            runtime.stop.set()  # leave the worker after this result
            return LipResult(["late words"], 12)
        def close(self):
            closed.set()
    activate(runtime)
    runtime.action("start", {})
    runtime.action("stop", {})
    runtime.finish_at = 0
    with patch("hackathon.lips_adapter.LipsAdapter", Adapter):
        thread = threading.Thread(target=runtime._lips)
        thread.start()
        try:
            assert entered.wait(3)
            runtime.action("camera_session", {"active": False})
            stopped_revision = runtime.revision
            release.set()
            thread.join(3)
            assert not thread.is_alive()
            assert runtime.revision > stopped_revision
            assert runtime.state["candidates"] == ["late words"]
            assert runtime.state["session_text"] == "late words"
            assert runtime.state["phase"] == "idle"
            assert closed.is_set()
        finally:
            release.set()
            runtime.stop.set()
            thread.join(3)

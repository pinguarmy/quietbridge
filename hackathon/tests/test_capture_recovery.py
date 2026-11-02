"""Deterministic capture-boundary regressions using a gated lip adapter."""
import threading
import time

import pytest

from hackathon.contracts import LipResult
from hackathon.profiles import ProfileManager
from hackathon.runtime import Runtime


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("worker did not reach the expected state")


class FakeAdapter:
    def __init__(self):
        self.recording = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_process = False
        self.last_mouth_open = 0.0
        self.frames = []
        self.windows = []
        self.finished = threading.Event()
        self.cancel_boundaries = []

    def start(self, timestamp=None):
        self.recording = []

    def process(self, frame, timestamp):
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError("test gate timed out")
        if self.fail_process:
            raise RuntimeError("capture failed")
        if self.recording is not None:
            self.recording.append(frame)
        self.frames.append(frame)

    def finish(self):
        self.windows.append(list(self.recording))
        self.recording = None
        self.finished.set()
        return LipResult(["hello"], 1)

    def cancel(self, timestamp_ms=None):
        self.cancel_boundaries.append(timestamp_ms)
        self.recording = None

    def close(self):
        pass


@pytest.fixture
def rig(monkeypatch):
    adapter = FakeAdapter()
    now = [1000]
    monkeypatch.setattr("hackathon.lips_adapter.LipsAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr("hackathon.runtime.clock_ms", lambda: now[0])
    runtime = Runtime(browser_camera=True)
    runtime.state["camera"] = "ready"
    thread = threading.Thread(target=runtime._lips, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    wait_for(lambda: runtime.snapshot()["lips"] == "ready")
    try:
        yield runtime, adapter, now
    finally:
        adapter.release.set()
        runtime.close()
        thread.join(2)
        assert not thread.is_alive()


def test_tail_drains_all_queued_frames_before_decode(rig):
    runtime, adapter, now = rig
    runtime.action("start", {})
    adapter.release.clear()
    runtime.lip_frames.put(("first", 1001))
    assert adapter.entered.wait(2)
    runtime.lip_frames.put(("second", 1002))
    runtime.lip_frames.put(("third", 1003))
    runtime.lip_frames.put(("post-tail", 1401))
    runtime.action("stop", {})
    now[0] = 1400
    adapter.release.set()
    assert adapter.finished.wait(2)
    assert adapter.windows == [["first", "second", "third"]]


def test_camera_stop_drains_queued_capture(rig):
    runtime, adapter, now = rig
    runtime.action("camera_session", {"active": True})
    runtime.action("start", {})
    adapter.release.clear()
    runtime.lip_frames.put(("first", 1001))
    assert adapter.entered.wait(2)
    runtime.lip_frames.put(("second", 1002))
    runtime.lip_frames.put(("third", 1003))
    now[0] = 1003
    runtime.action("camera_session", {"active": False})
    adapter.release.set()
    assert adapter.finished.wait(2)
    assert adapter.windows == [["first", "second", "third"]]


@pytest.mark.parametrize("reset", ["cancel", "camera_session"])
def test_cancel_or_session_reset_discards_queued_old_frames(rig, reset):
    runtime, adapter, now = rig
    runtime.action("start", {})
    adapter.release.clear()
    runtime.lip_frames.put(("in-flight", 1001))
    assert adapter.entered.wait(2)
    runtime.lip_frames.put(("old-1", 1002))
    runtime.lip_frames.put(("old-2", 1003))
    now[0] = 1003
    if reset == "cancel":
        runtime.action("cancel", {})
    else:
        runtime.action("camera_session", {"active": True})
        runtime.state["auto_recognition"] = False  # exercise explicit capture only
    # A new capture starts before the blocked old frame returns.
    now[0] = 1004
    runtime.action("start", {})
    runtime.lip_frames.put(("new", 1004))
    adapter.release.set()
    wait_for(lambda: "new" in adapter.frames)
    runtime.action("stop", {})
    now[0] = 1404
    assert adapter.finished.wait(2)
    assert adapter.windows == [["new"]]
    assert 1003 in adapter.cancel_boundaries
    assert "old-1" not in adapter.frames and "old-2" not in adapter.frames


def test_capture_exception_during_finalizing_recovers_session(rig):
    runtime, adapter, _ = rig
    runtime.action("camera_session", {"active": True})
    runtime.session.append("earlier words")
    runtime.action("start", {})
    adapter.release.clear()
    adapter.fail_process = True
    runtime.lip_frames.put(("broken", 1001))
    assert adapter.entered.wait(2)
    runtime.action("camera_session", {"active": False})
    assert runtime.snapshot()["session_finalizing"]
    adapter.release.set()
    wait_for(lambda: runtime.snapshot()["lips"] == "unavailable")
    state = runtime.snapshot()
    assert state["phase"] == "idle"
    assert not state["session_finalizing"]
    assert state["lip_error"] == "capture failed"
    assert state["session_text"] == "earlier words"
    assert state["last_spoken"] == ""


def test_profile_switch_still_clears_capture_queues(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    default = manager.ensure_default_profile()
    manager.create_profile("other", "Other")
    runtime = Runtime(rehearsal=True, profile=default, profile_manager=manager)
    try:
        for timestamp in (1, 2, 3):
            runtime.lip_frames.put(("old", timestamp))
            runtime.hand_frames.put(("old", timestamp))
        runtime.switch_profile("other")
        assert runtime.snapshot()["profile_id"] == "other"
        assert runtime.lip_frames.empty()
        assert runtime.hand_frames.empty()
        assert runtime.snapshot()["phase"] == "idle"
    finally:
        runtime.close()

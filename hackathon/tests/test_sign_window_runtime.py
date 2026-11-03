"""Worker integration with an explicitly selected standard-sign backend."""
import threading
from unittest.mock import Mock

import pytest

from hackathon.contracts import LipResult, SignEvent
from hackathon.runtime import Runtime
from hackathon.tests.test_motion_integration import wait_for


class IsolatedBackend:
    windowed = True
    vocabulary = {"BSL_WATER": {"phrase": "water", "matches": ["water"], "enabled": True, "verified": False}}
    model_info = {"language": "bsl", "sha256": "a" * 64, "status": "experimental"}
    last_tracking = []
    last_diagnostics = {"outcome": "accepted"}

    def __init__(self):
        self.frames = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.started = threading.Event()
        self.finished_frames = None

    def start(self, ts):
        self.begin = ts
        self.started.set()

    def process(self, frame, ts):
        self.entered.set()
        assert self.release.wait(3)
        self.frames.append(ts)
        return []

    def finish(self, ts):
        self.finished_frames = list(self.frames)
        return [SignEvent("BSL_WATER", .95, self.begin, ts, True)]

    def cancel(self):
        pass

    def close(self):
        pass


def test_stop_drains_standard_sign_window_before_publishing(monkeypatch):
    backend = IsolatedBackend()
    now = [1000]
    monkeypatch.setattr("hackathon.signs.isolated.IsolatedSignRecognizer", lambda path: backend)
    monkeypatch.setattr("hackathon.runtime.clock_ms", lambda: now[0])
    runtime = Runtime(browser_camera=True, enable_lips=False, sign_model="explicit.json")
    runtime.cleanup.submit = Mock(return_value=False)
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "isolated")
        assert list(runtime.vocabulary) == ["BSL_WATER"]
        runtime.action("camera_session", {"active": True})
        runtime.state["camera"] = "ready"
        runtime.action("start", {})
        assert backend.started.wait(2)
        runtime.hand_frames.put((None, 1040))
        assert backend.entered.wait(2)
        for ts in (1080, 1120, 1160):
            runtime.hand_frames.put((None, ts))
        now[0] = 1200
        runtime.action("camera_session", {"active": False})
        backend.release.set()
        wait_for(lambda: runtime.state["decision"]["text"] == "water")
        assert backend.finished_frames == [1040, 1080, 1120, 1160]
        assert not runtime.state["session_finalizing"]
        assert runtime.state["sign_model"]["language"] == "bsl"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
        assert runtime.state["sign_diagnostics"]["outcome"] == "accepted"
        runtime.speaker.speak = Mock()
        runtime.action("confirm", {"revision": runtime.revision, "text": "water"})
        runtime.speaker.speak.assert_not_called()
    finally:
        backend.release.set()
        runtime.close()


def test_sign_worker_load_error_does_not_fall_back_to_stock(monkeypatch):
    def fail(path):
        raise ValueError("model hash mismatch")
    monkeypatch.setattr("hackathon.signs.isolated.IsolatedSignRecognizer", fail)
    runtime = Runtime(browser_camera=True, enable_lips=False, sign_model="missing.json")
    try:
        runtime._hands()
        assert runtime.state["signs"] == "unavailable"
        assert runtime.vocabulary == {}
        assert "hash mismatch" in runtime.state["sign_error"]
    finally:
        runtime.close()


def test_worker_stop_event_belongs_to_thread_creation(monkeypatch):
    runtime = Runtime(browser_camera=True, enable_lips=False)
    callbacks, observed = [], []
    class DelayedThread:
        def __init__(self, *, target, **kwargs): callbacks.append(target)
        def start(self): pass
        def join(self, timeout=None): pass
    runtime._hands = lambda generation, stop: observed.append(stop)
    try:
        with monkeypatch.context() as m:
            m.setattr("hackathon.runtime.threading.Thread", DelayedThread)
            runtime._start_workers()
            original = runtime._worker_stop
            runtime._worker_stop = threading.Event()
            callbacks[0]()
        assert observed == [original]
    finally:
        runtime.close()


def test_standard_sign_window_participates_in_lip_candidate_selection():
    runtime = Runtime(browser_camera=True, sign_model="explicit.json")
    try:
        runtime.vocabulary = dict(IsolatedBackend.vocabulary)
        runtime._fusion_vocabulary = runtime.vocabulary
        runtime.state.update(sign_input="isolated", lips="ready", signs="ready", camera="ready")
        runtime.action("camera_session", {"active": True})
        runtime.state["camera"] = "ready"
        runtime.action("start", {})
        runtime._publish_session(LipResult(["tea", "water"], 1),
                                 [SignEvent("BSL_WATER", .95, runtime.record_start, runtime.record_start, True)])
        assert runtime.state["session_text"] == "water"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
    finally:
        runtime.close()

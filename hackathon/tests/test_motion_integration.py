import importlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hackathon.contracts import LipResult, SignEvent
from hackathon.runtime import Runtime


def wait_for(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


class MotionRecognizer:
    def __init__(self):
        self._motion = object()
        self.last_tracking = []
        self.started = []
        self.finished = []
        self.cancelled = 0
        self.processed = []
        self.finish_delay = 0.0
        self.finish_started = threading.Event()
        self.finish_release = None
        self.in_window = False
        self._motion_last_boundary_ms = None

    def start(self, timestamp_ms):
        self.started.append(timestamp_ms)
        self.in_window = True
        self._motion_last_boundary_ms = timestamp_ms

    def process(self, frame, timestamp_ms):
        self.last_tracking = []
        if self._motion_last_boundary_ms is not None and timestamp_ms < self._motion_last_boundary_ms:
            raise AssertionError("stale frame crossed the motion boundary")
        if self.in_window:
            self.processed.append(timestamp_ms)
        return []

    def finish(self, timestamp_ms):
        self.finished.append(timestamp_ms)
        self.in_window = False
        self._motion_last_boundary_ms = timestamp_ms
        self.finish_started.set()
        if self.finish_release is not None:
            assert self.finish_release.wait(1)
        elif self.finish_delay:
            time.sleep(self.finish_delay)
        return [SignEvent("HELLO", 0.5, self.started[-1], timestamp_ms, True)]

    def cancel(self):
        self.cancelled += 1
        self.in_window = False

    def close(self):
        pass


def make_runtime(monkeypatch, with_lips=False):
    recognizer = MotionRecognizer()
    real_import = importlib.import_module
    monkeypatch.setattr("hackathon.runtime.importlib.import_module",
                        lambda name: SimpleNamespace(SignRecognizer=lambda: recognizer)
                        if name == "hackathon.signs.recognizer" else real_import(name))
    vocabulary = {"HELLO": {"phrase": "Hello.", "matches": ["hello"],
                            "enabled": True, "verified": False}}
    runtime = Runtime(enable_lips=with_lips, vocabulary=vocabulary)
    runtime.state["camera"] = "ready"
    return runtime, recognizer


def test_motion_record_release_publishes_one_reviewed_event(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch)
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion")
        runtime.action("start", {})
        start = runtime.record_start
        runtime.hand_frames.put((np.zeros((2, 2, 3), dtype=np.uint8), start + 1))
        runtime.action("stop", {})
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello.")
        assert recognizer.started == [start]
        assert len(recognizer.finished) == 1
        assert runtime.state["phase"] == "idle"
        assert runtime.state["decision"]["status"] == "confirmation_needed"
        assert runtime.state["decision"]["matched_signs"] == ["HELLO"]
    finally:
        runtime.close()
        thread.join(1)


def test_motion_cancel_discards_window_without_a_decision(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch)
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion")
        runtime.action("start", {})
        runtime.action("cancel", {})
        wait_for(lambda: recognizer.cancelled == 1)
        assert recognizer.finished == []
        assert runtime.state["phase"] == "idle"
        assert runtime.state["decision"]["status"] == "unavailable"
    finally:
        runtime.close()
        thread.join(1)


def test_stock_sign_only_does_not_open_a_motion_window():
    runtime = Runtime(enable_lips=False, vocabulary={})
    runtime.state.update(camera="ready", signs="ready", sign_input="stock")
    try:
        with pytest.raises(ValueError, match="automatic"):
            runtime.action("start", {})
    finally:
        runtime.close()


def test_stale_and_post_tail_frames_never_enter_motion_window(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch)
    runtime.state.update(signs="ready", sign_input="motion")
    runtime.action("start", {})
    start = runtime.record_start
    runtime.hand_frames.put((np.zeros((2, 2, 3), dtype=np.uint8), start - 5))
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: bool(recognizer.started))
        runtime.action("stop", {})
        end = runtime.finish_at
        runtime.hand_frames.put((np.zeros((2, 2, 3), dtype=np.uint8), end + 100))
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello.")
        assert recognizer.processed == [], (start, end, recognizer.processed)
        assert runtime.state["signs"] == "ready"
    finally:
        runtime.close()
        thread.join(1)


def test_sign_only_window_never_starts_lip_capture(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch, with_lips=True)

    class Lips:
        def __init__(self):
            self.recording = None
            self.last_tracking = None
            self.starts = 0

        def start(self):
            self.starts += 1

        def process(self, frame, timestamp_ms):
            pass

        def finish(self):
            return LipResult(["Hello"], 1)

        def close(self):
            pass

    lips = Lips()
    monkeypatch.setattr("hackathon.lips_adapter.LipsAdapter", lambda: lips)
    hands = threading.Thread(target=runtime._hands, daemon=True)
    lip_thread = threading.Thread(target=runtime._lips, daemon=True)
    runtime.threads.extend([hands, lip_thread])
    hands.start(); lip_thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion" and runtime.state["lips"] == "ready")
        runtime.action("mode", {"mode": "signs"})
        runtime.action("start", {})
        runtime.action("stop", {})
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello.")
        assert lips.starts == 0
        assert runtime.state["decision"]["reason"] == "sign_only"
    finally:
        runtime.close()
        hands.join(1); lip_thread.join(1)


def test_combined_waits_for_slow_motion_before_fusion(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch, with_lips=True)
    recognizer.finish_delay = 0.2

    class Lips:
        def __init__(self):
            self.recording = None
            self.last_tracking = None
            self.starts = 0

        def start(self):
            self.starts += 1
            self.recording = object()

        def process(self, frame, timestamp_ms):
            pass

        def finish(self):
            return LipResult(["Hello"], 1)

        def close(self):
            pass

    lips = Lips()
    monkeypatch.setattr("hackathon.lips_adapter.LipsAdapter", lambda: lips)
    hands = threading.Thread(target=runtime._hands, daemon=True)
    lip_thread = threading.Thread(target=runtime._lips, daemon=True)
    runtime.threads.extend([hands, lip_thread])
    hands.start(); lip_thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion" and runtime.state["lips"] == "ready")
        runtime.action("start", {})
        runtime.action("stop", {})
        time.sleep(0.08)
        assert runtime.state["phase"] != "idle"
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello")
        assert runtime.state["decision"]["reason"] == "agreement"
        assert runtime.state["decision"]["matched_signs"] == ["HELLO"]
        assert lips.starts == 1
    finally:
        runtime.close()
        hands.join(1); lip_thread.join(1)


def test_cancel_during_motion_finish_cannot_publish_or_pollute_next_window(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch)
    recognizer.finish_release = threading.Event()
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion")
        runtime.action("start", {})
        runtime.action("stop", {})
        wait_for(lambda: recognizer.finish_started.is_set())
        runtime.action("cancel", {})
        recognizer.finish_release.set()
        wait_for(lambda: recognizer.cancelled == 1)
        assert runtime.state["decision"]["status"] == "unavailable"
        assert runtime.state["decision"]["text"] == ""
        runtime.action("start", {})
        recognizer.finish_release = None
        runtime.action("stop", {})
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello.")
        assert len(recognizer.started) == 2
    finally:
        runtime.close()
        thread.join(1)


def test_frame_queued_during_finish_cannot_cross_completed_boundary(monkeypatch):
    runtime, recognizer = make_runtime(monkeypatch)
    recognizer.finish_release = threading.Event()
    thread = threading.Thread(target=runtime._hands, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    try:
        wait_for(lambda: runtime.state["sign_input"] == "motion")
        runtime.action("start", {})
        runtime.action("stop", {})
        end = runtime.finish_at
        wait_for(lambda: recognizer.finish_started.is_set())
        runtime.hand_frames.put((np.zeros((2, 2, 3), dtype=np.uint8), end - 1))
        recognizer.finish_release.set()
        wait_for(lambda: runtime.state["decision"]["text"] == "Hello.")
        time.sleep(0.1)
        assert runtime.state["signs"] == "ready"
        assert recognizer.processed == []
    finally:
        runtime.close()
        thread.join(1)

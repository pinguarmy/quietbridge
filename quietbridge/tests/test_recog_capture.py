"""Capture-boundary regressions on synthetic frame streams with known
timestamps (fake tracker/decoder). They prove which frames reach a clip, not
camera or recognition accuracy."""
from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from quietbridge.contracts import LipResult
from quietbridge.lip_activity import ActivityAction, ActivityState, LipActivityConfig, LipActivityDetector
from quietbridge.lips_adapter import LipsAdapter
from quietbridge.profiles import ProfileManager
from quietbridge.runtime import Runtime

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("worker did not reach the expected state")


# --- lip activity detector ----------------------------------------------------

def test_out_of_order_quiet_frame_cannot_start_the_quiet_timer_early():
    detector = LipActivityDetector(LipActivityConfig(min_duration_ms=1000, quiet_ms=700))
    assert detector.update(0.08, 1000) == ActivityAction.START
    assert detector.update(0.08, 3000) == ActivityAction.NONE
    # A late frame stamped 2000 must not count as quiet since 2000.
    assert detector.update(0.01, 2000) == ActivityAction.NONE
    assert detector.quiet_since is None
    assert detector.update(0.01, 3010) == ActivityAction.NONE  # 10 ms of real quiet
    assert detector.state == ActivityState.RECORDING
    assert detector.update(0.01, 3710) == ActivityAction.STOP   # well-ordered endpoint unchanged


def test_repeated_timestamp_is_ignored_and_reset_accepts_a_fresh_clock():
    detector = LipActivityDetector()
    detector.update(0.01, 500)
    assert detector.update(0.08, 500) == ActivityAction.NONE
    detector.reset()
    assert detector.update(0.08, 10) == ActivityAction.START


def test_unseen_time_is_not_quiet_when_face_loss_waits():
    detector = LipActivityDetector(LipActivityConfig(min_duration_ms=1000, quiet_ms=700, max_chunk_ms=9000))
    detector.update(0.08, 1000)
    detector.update(0.01, 1500)              # quiet starts
    detector.update(None, 1600)              # face lost for 2 s
    detector.update(None, 3600)
    assert detector.update(0.01, 3700) == ActivityAction.NONE  # 0 ms observed quiet, not 2200
    assert detector.update(0.01, 4399) == ActivityAction.NONE
    assert detector.update(0.01, 4400) == ActivityAction.STOP


def test_profile_switch_resets_detector_with_the_frame_clock(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    manager.create_profile("bob", "Bob")
    runtime = Runtime(rehearsal=True, profile_manager=manager)
    try:
        runtime._lip_activity.update(0.08, 50_000)
        runtime._last_frame_timestamp_ms = 50_000
        runtime.switch_profile("bob")
        assert runtime._last_frame_timestamp_ms is None
        assert runtime._lip_activity.last_ts is None
        assert runtime._lip_activity.state == ActivityState.IDLE
    finally:
        runtime.close()


# --- lips adapter ---------------------------------------------------------------

def _obs():
    pts = np.array([[60.0 + i, 50.0 + (i % 5)] for i in range(40)], dtype=np.float32)
    return SimpleNamespace(pts=pts, outer_lips=pts[:20], inner_lips=pts[20:], anchors=pts[:5], mouth_open=0.05)


class Gate:
    def __init__(self):
        self.open = threading.Event()
        self.open.set()
        self.entered = threading.Event()

    def detect(self, frame, ts):
        self.entered.set()
        assert self.open.wait(3)
        return _obs()

    def close(self):
        pass


class Reader:
    def encode(self, rois):
        return "enc"

    def beam_search_details(self, enc, nbest):
        return [{"text": "OK", "score": -1.0}]


def _adapter(monkeypatch, tracker=None):
    monkeypatch.setattr("lipflow.camera.face_crop", lambda gray, obs, rec: ("crop", (0, 0)))
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((12, 96, 96), dtype=np.uint8))
    adapter = object.__new__(LipsAdapter)
    adapter.tracker = tracker or Gate()
    adapter.reader = Reader()
    adapter.model_info = {"mode": "stock"}
    adapter.recording = None
    adapter.last_tracking = adapter.last_mouth_open = adapter.record_start_ms = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None
    adapter.last_metrics = {}
    return adapter


def test_recording_never_duplicates_or_reorders_frames(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter.start(1000)
    for ts in (1000, 1040, 1040, 1020, 1080):
        adapter.process(FRAME, ts)
    assert [round(t * 1000) for t in adapter.recording.ts] == [1000, 1040, 1080]


def test_pre_start_frame_processed_after_start_is_bounded_pre_roll(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter._last_window_end_ms = 600
    adapter.start(1000)
    for ts in (500, 650, 750, 1010):  # 500 is before the pre-roll, 650 after the last window
        adapter.process(FRAME, ts)
    assert [round(t * 1000) for t in adapter.recording.ts] == [750, 1010]
    adapter = _adapter(monkeypatch)
    adapter._last_window_end_ms = 900
    adapter.start(1000)
    adapter.process(FRAME, 850)       # belongs to the previous window
    assert adapter.recording.ts == []


def test_resample_metrics_report_repeated_and_unused_frames():
    from lipflow.vsr import LipReader
    from quietbridge.lips_adapter import _resample_metrics
    gap = _resample_metrics(LipReader.resample, [0, .04, .08, .2, .24])
    assert gap == {"lip_model_frames": 7, "lip_repeated_frames": 2, "lip_unused_frames": 0}
    fast = _resample_metrics(LipReader.resample, [0, .033, .067, .1, .133, .167, .2])
    assert fast == {"lip_model_frames": 6, "lip_repeated_frames": 0, "lip_unused_frames": 1}
    assert _resample_metrics(None, [0, .04]) == {} and _resample_metrics(LipReader.resample, []) == {}


# --- lips worker -------------------------------------------------------------------

@pytest.fixture
def worker(monkeypatch):
    gate = Gate()
    adapter = _adapter(monkeypatch, gate)
    now = [1000]
    monkeypatch.setattr("quietbridge.lips_adapter.LipsAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr("quietbridge.runtime.clock_ms", lambda: now[0])
    runtime = Runtime(browser_camera=True, vocabulary={})
    runtime.cleanup.submit = lambda *args: False
    runtime.state["camera"] = "ready"
    thread = threading.Thread(target=runtime._lips, daemon=True)
    runtime.threads.append(thread)
    thread.start()
    wait_for(lambda: runtime.snapshot()["lips"] == "ready")
    try:
        yield runtime, adapter, gate, now
    finally:
        gate.open.set()
        runtime.close()
        thread.join(3)


def _finish(runtime, now, release=1200):
    now[0] = release
    runtime.action("stop", {})
    now[0] = release + 400
    wait_for(lambda: runtime.snapshot()["phase"] == "idle" and runtime.snapshot()["candidates"])
    return runtime.snapshot()


def test_manual_start_keeps_a_queued_pre_start_frame(worker):
    runtime, adapter, gate, now = worker
    gate.open.clear()
    gate.entered.clear()
    runtime.lip_frames.put((FRAME, 900))      # idle frame being tracked
    assert gate.entered.wait(3)
    runtime.lip_frames.put((FRAME, 950))      # queued before the start...
    runtime.action("start", {})               # ...record_start = 1000
    gate.open.set()
    runtime.lip_frames.put((FRAME, 1050))
    wait_for(lambda: adapter.recording is not None and len(adapter.recording.ts) == 3)
    state = _finish(runtime, now)
    assert state["metrics"]["lip_raw_frames"] == 3  # 900 (pre-roll), 950, 1050
    assert state["metrics"]["lip_start_ms"] == 900 and state["metrics"]["lip_end_ms"] == 1050


def test_overflow_drops_are_reported_per_window(worker):
    runtime, adapter, gate, now = worker
    runtime.capture_dropped_frames = 7  # earlier windows
    runtime.action("start", {})
    gate.open.clear()
    gate.entered.clear()
    runtime.receive_frame(FRAME, 1001)
    assert gate.entered.wait(3)
    for ts in range(1002, 1002 + runtime.lip_frames.maxsize + 2):
        runtime.receive_frame(FRAME, ts)
    gate.open.set()
    state = _finish(runtime, now, release=1200)
    assert state["metrics"]["lip_dropped_frames"] == 2
    assert state["metrics"]["capture_dropped_frames"] == 10  # either queue, cumulative

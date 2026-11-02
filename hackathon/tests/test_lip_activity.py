"""Synthetic state machine, pre-roll, and endpointing tests for lip activity.

Tests:
- Zero-dependency state machine transitions
- Activity onset vs missing face vs closed/still mouth
- Minimum utterance duration and quiet endpointing (600-800ms)
- Maximum chunk cap (6000ms)
- Cancellation and reset isolation
- Pre-roll bounded memory, no disk persistence, and window boundary isolation
- Preservation of manual protocol and motion window independence
"""
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hackathon.contracts import LipResult
from hackathon.lip_activity import (
    ActivityAction,
    ActivityState,
    LipActivityConfig,
    LipActivityDetector,
)
from hackathon.lips_adapter import LipsAdapter
from hackathon.runtime import Runtime


# ---------------------------------------------------------------------------
# Unit tests for LipActivityDetector
# ---------------------------------------------------------------------------

def test_config_defaults_and_immutability():
    cfg = LipActivityConfig()
    assert cfg.open_threshold == 0.04
    assert 600 <= cfg.quiet_ms <= 800
    assert cfg.quiet_ms == 700
    assert cfg.min_duration_ms == 1200
    assert cfg.max_chunk_ms == 6000
    assert cfg.pre_roll_ms == 300
    assert cfg.min_active_frames == 1
    assert cfg.face_loss_quiet is False
    with pytest.raises(Exception):
        cfg.open_threshold = 0.1  # frozen dataclass


def test_missing_face_and_still_mouth_do_not_start_when_idle():
    detector = LipActivityDetector()
    assert detector.state == ActivityState.IDLE

    # No face (None)
    for ts in (100, 200, 300):
        action = detector.update(None, ts)
        assert action == ActivityAction.NONE
        assert detector.state == ActivityState.IDLE

    # Closed/still mouth (0.0, 0.01, 0.03 < 0.04 threshold)
    for opening in (0.0, 0.01, 0.02, 0.039):
        action = detector.update(opening, 400)
        assert action == ActivityAction.NONE
        assert detector.state == ActivityState.IDLE


def test_activity_onset_triggers_start():
    detector = LipActivityDetector()
    assert detector.update(0.01, 100) == ActivityAction.NONE
    assert detector.state == ActivityState.IDLE

    # Activity above threshold (e.g. 0.05)
    action = detector.update(0.05, 200)
    assert action == ActivityAction.START
    assert detector.state == ActivityState.RECORDING
    assert detector.start_ts == 200


def test_continuous_activity_does_not_stop_until_max_chunk():
    cfg = LipActivityConfig(max_chunk_ms=3000)
    detector = LipActivityDetector(cfg)

    assert detector.update(0.08, 1000) == ActivityAction.START
    assert detector.state == ActivityState.RECORDING

    # Active frames every 100ms
    for ts in range(1100, 3900, 100):
        assert detector.update(0.08, ts) == ActivityAction.NONE
        assert detector.state == ActivityState.RECORDING

    # Reaching max chunk at 4000 (1000 + 3000)
    action = detector.update(0.08, 4000)
    assert action == ActivityAction.STOP
    assert detector.state == ActivityState.IDLE


def test_quiet_triggers_stop_after_min_duration_and_quiet_timeout():
    # min_duration=1000ms, quiet_ms=700ms
    cfg = LipActivityConfig(min_duration_ms=1000, quiet_ms=700)
    detector = LipActivityDetector(cfg)

    # Start at 1000
    assert detector.update(0.08, 1000) == ActivityAction.START

    # Active until 1500
    assert detector.update(0.08, 1500) == ActivityAction.NONE

    # Mouth becomes quiet at 1600
    assert detector.update(0.01, 1600) == ActivityAction.NONE
    assert detector.quiet_since == 1600

    # At 2000: quiet elapsed 400ms (< 700ms)
    assert detector.update(0.01, 2000) == ActivityAction.NONE
    assert detector.state == ActivityState.RECORDING

    # At 2290: quiet elapsed 690ms (< 700ms)
    assert detector.update(0.01, 2290) == ActivityAction.NONE
    assert detector.state == ActivityState.RECORDING

    # At 2300: quiet elapsed 700ms (>= 700ms) and total elapsed 1300ms (>= 1000ms)
    assert detector.update(0.01, 2300) == ActivityAction.STOP
    assert detector.state == ActivityState.IDLE


def test_quiet_respects_min_duration_even_if_quiet_persists():
    # min_duration=1200ms, quiet_ms=600ms
    cfg = LipActivityConfig(min_duration_ms=1200, quiet_ms=600)
    detector = LipActivityDetector(cfg)

    # Brief 100ms utterance
    assert detector.update(0.08, 1000) == ActivityAction.START
    assert detector.update(0.01, 1100) == ActivityAction.NONE  # quiet starts at 1100

    # At 1700: quiet elapsed 600ms (>= quiet_ms), but total elapsed is 700ms (< 1200ms)
    assert detector.update(0.01, 1700) == ActivityAction.NONE
    assert detector.state == ActivityState.RECORDING

    # At 2200: total elapsed is 1200ms (>= min_duration_ms) and quiet elapsed 1100ms (>= quiet_ms)
    assert detector.update(0.01, 2200) == ActivityAction.STOP
    assert detector.state == ActivityState.IDLE


def test_activity_resets_quiet_timer():
    cfg = LipActivityConfig(min_duration_ms=1000, quiet_ms=700)
    detector = LipActivityDetector(cfg)

    assert detector.update(0.08, 1000) == ActivityAction.START

    # Quiet from 1500 to 2000 (500ms quiet)
    assert detector.update(0.01, 1500) == ActivityAction.NONE
    assert detector.quiet_since == 1500
    assert detector.update(0.01, 2000) == ActivityAction.NONE

    # User speaks again at 2100 -> quiet timer must be cancelled
    assert detector.update(0.09, 2100) == ActivityAction.NONE
    assert detector.quiet_since is None

    # User goes quiet at 2200
    assert detector.update(0.01, 2200) == ActivityAction.NONE
    assert detector.quiet_since == 2200

    # At 2800 (600ms after 2200) -> not stopped
    assert detector.update(0.01, 2800) == ActivityAction.NONE
    assert detector.state == ActivityState.RECORDING

    # At 2900 (700ms after 2200) -> stopped
    assert detector.update(0.01, 2900) == ActivityAction.STOP
    assert detector.state == ActivityState.IDLE


def test_face_loss_behavior_default_vs_opt_in():
    # Default: face loss (None) does not start quiet timer
    detector = LipActivityDetector(LipActivityConfig(face_loss_quiet=False, max_chunk_ms=4000))
    detector.update(0.08, 1000)
    assert detector.state == ActivityState.RECORDING

    # Face lost for 1000ms
    assert detector.update(None, 1500) == ActivityAction.NONE
    assert detector.update(None, 2500) == ActivityAction.NONE
    assert detector.state == ActivityState.RECORDING
    assert detector.quiet_since is None

    # Reaching max chunk stops
    assert detector.update(None, 5000) == ActivityAction.STOP

    # Opt-in: face_loss_quiet=True
    opt_in = LipActivityDetector(LipActivityConfig(face_loss_quiet=True, min_duration_ms=1000, quiet_ms=700))
    opt_in.update(0.08, 1000)
    assert opt_in.update(None, 1200) == ActivityAction.NONE
    assert opt_in.quiet_since == 1200
    # At 2000 (elapsed 1000 >= min, quiet 800 >= 700)
    assert opt_in.update(None, 2000) == ActivityAction.STOP
    assert opt_in.state == ActivityState.IDLE


def test_reset_and_cancel_and_sync_recording():
    detector = LipActivityDetector()
    detector.update(0.08, 1000)
    detector.update(0.01, 1500)
    assert detector.state == ActivityState.RECORDING
    assert detector.start_ts == 1000
    assert detector.quiet_since == 1500

    detector.cancel()
    assert detector.state == ActivityState.IDLE
    assert detector.start_ts is None
    assert detector.quiet_since is None

    detector.sync_recording(3000)
    assert detector.state == ActivityState.RECORDING
    assert detector.start_ts == 3000
    assert detector.quiet_since is None


# ---------------------------------------------------------------------------
# Pre-roll and Window Isolation tests on LipsAdapter
# ---------------------------------------------------------------------------

def test_lips_adapter_preroll_buffers_and_populates_start(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    outer = np.array([[40, 20], [80, 40], [60, 60]], dtype=np.float32)
    inner = np.array([[50, 35], [70, 45]], dtype=np.float32)
    obs = SimpleNamespace(outer_lips=outer, inner_lips=inner,
                          anchors="anchors", mouth_open=0.08)

    class FakeTracker:
        def detect(self, f, ts):
            return obs
        def close(self):
            pass

    monkeypatch.setattr("lipflow.camera.face_crop", lambda gray, o, rec: "crop")

    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.tracker = FakeTracker()
    adapter.recording = None
    adapter.last_tracking = None
    adapter.last_mouth_open = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None

    # Idle frames arrive at 700, 800, 900, 1000
    for ts in (700, 800, 900, 1000):
        adapter.process(frame, ts)
        assert adapter.last_mouth_open == 0.08

    assert len(adapter._preroll) == 4
    assert adapter.recording is None

    # Call start at 1000 (pre_roll covers 1000 - 300 = 700 to 1000)
    adapter.start(1000)
    assert adapter.recording is not None
    # All 4 pre-roll frames are transferred into recording
    assert len(adapter.recording.ts) == 4
    assert adapter.recording.ts == [0.7, 0.8, 0.9, 1.0]
    assert adapter.recording.grays == ["crop", "crop", "crop", "crop"]
    # Pre-roll deque is now emptied
    assert len(adapter._preroll) == 0

    # Subsequent recording frames are appended directly
    adapter.process(frame, 1050)
    assert adapter.recording.ts == [0.7, 0.8, 0.9, 1.0, 1.05]


def test_preroll_does_not_leak_frames_from_previous_window(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    obs = SimpleNamespace(outer_lips=np.zeros((3, 2)), inner_lips=np.zeros((2, 2)),
                          anchors="anchors", mouth_open=0.08)

    class FakeTracker:
        def detect(self, f, ts):
            return obs

    class FakeReader:
        def encode(self, rois):
            return "enc"
        def beam_search(self, enc, nbest):
            return ["OK"]

    monkeypatch.setattr("lipflow.camera.face_crop", lambda gray, o, rec: "crop")
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((12, 96, 96), dtype=np.uint8))

    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.reader = FakeReader()
    adapter.tracker = FakeTracker()
    adapter.recording = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None

    # Window 1 runs from 1000 to 2000
    adapter.start(1000)
    adapter.process(frame, 1500)
    adapter.process(frame, 2000)
    assert adapter.recording.ts == [1.5, 2.0]

    # Finish window 1 (sets last_window_end_ms = 2000)
    adapter.finish()
    assert adapter.recording is None
    assert adapter._last_window_end_ms == 2000

    # Frames after window 1: 1900 (should be discarded because <= 2000), 2050, 2150
    adapter.process(frame, 2050)
    adapter.process(frame, 2150)

    # Start Window 2 at 2200
    adapter.start(2200)
    # Window 2 must only have frames from 2050 and 2150; nothing from <= 2000
    assert all(t > 2.0 for t in adapter.recording.ts)
    assert adapter.recording.ts == [2.05, 2.15]


def test_preroll_cleared_on_cancel():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    obs = SimpleNamespace(outer_lips=np.zeros((3, 2)), inner_lips=np.zeros((2, 2)),
                          anchors="anchors", mouth_open=0.05)

    class FakeTracker:
        def detect(self, f, ts):
            return obs

    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.tracker = FakeTracker()
    adapter.recording = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None

    adapter.process(frame, 500)
    adapter.process(frame, 600)
    assert len(adapter._preroll) == 2

    # Cancel clears preroll and sets boundary
    adapter.cancel(timestamp_ms=600)
    assert len(adapter._preroll) == 0
    assert adapter.recording is None
    assert adapter._last_window_end_ms == 600

    # Frame before boundary is rejected
    adapter.process(frame, 550)
    assert len(adapter._preroll) == 0

    # Frame after boundary is buffered
    adapter.process(frame, 700)
    assert len(adapter._preroll) == 1


# ---------------------------------------------------------------------------
# Integration tests with Runtime
# ---------------------------------------------------------------------------

def test_runtime_continuous_session_accumulates_multiple_chunks():
    """Verify continuous session accumulates chunks driven by lip activity."""
    runtime = Runtime(browser_camera=True, vocabulary={})
    runtime.state.update(lips="ready", camera="ready")
    try:
        runtime.action("camera_session", {"active": True})
        assert runtime.state["session_active"] is True
        assert runtime.state["session_chunks"] == []

        # Chunk 1: Activity at 1000
        with patch("hackathon.runtime.clock_ms", return_value=1000):
            runtime._auto_frame(0.08, 1000)
        assert runtime.state["phase"] == "recording"
        assert runtime.record_start == 1000

        # Quiet at 1800, stops at 2500 (1800 + 700 = 2500, elapsed 1500 >= 1200)
        runtime._auto_frame(0.01, 1800)
        runtime._auto_frame(0.01, 2500)
        assert runtime.state["phase"] == "tail"
        assert runtime.state["session_active"] is True

        # Simulate decoder publish for chunk 1
        runtime._publish_session(LipResult(["First chunk"], 15))
        assert runtime.state["session_chunks"] == ["First chunk"]
        assert runtime.state["session_text"] == "First chunk"

        # Return to idle
        runtime._auto_reset()
        runtime.state.update(phase="idle", recording=False)
        runtime.record_start = None

        # Chunk 2: Activity at 4000
        with patch("hackathon.runtime.clock_ms", return_value=4000):
            runtime._auto_frame(0.09, 4000)
        assert runtime.state["phase"] == "recording"
        assert runtime.record_start == 4000

        # Quiet at 4600, stops at 5300 (4600 + 700 = 5300, elapsed 1300 >= 1200)
        runtime._auto_frame(0.01, 4600)
        runtime._auto_frame(0.01, 5300)
        assert runtime.state["phase"] == "tail"

        # Simulate decoder publish for chunk 2
        runtime._publish_session(LipResult(["Second chunk"], 15))
        assert runtime.state["session_chunks"] == ["First chunk", "Second chunk"]
        assert runtime.state["session_text"] == "First chunk Second chunk"
        assert runtime.state["session_active"] is True

        # End camera session -> finalizes
        runtime.action("camera_session", {"active": False})
        assert runtime.state["session_active"] is False
    finally:
        runtime.close()


def test_manual_record_is_unaffected_by_auto_frame():
    """Manual Record protocol is completely independent of lip activity auto_frame."""
    runtime = Runtime(browser_camera=True, vocabulary={})
    runtime.state.update(lips="ready", camera="ready")
    try:
        # Without camera_session, auto_recognition is False
        assert not runtime.state["auto_recognition"]
        runtime._auto_frame(0.08, 1000)
        assert runtime.state["phase"] == "idle"

        # User starts manually
        runtime.action("start", {})
        assert runtime.state["phase"] == "recording"
        assert not runtime.state["auto_recognition"]

        # Quiet in auto_frame does not stop manual recording
        runtime._auto_frame(0.01, 2500)
        assert runtime.state["phase"] == "recording"

        # User stops manually
        runtime.action("stop", {})
        assert runtime.state["phase"] == "tail"
    finally:
        runtime.close()

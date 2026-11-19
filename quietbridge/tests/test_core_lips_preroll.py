"""Pre-roll keeps recording arrays aligned and never fabricates frames (fake tracker)."""
from collections import deque
from types import SimpleNamespace

import numpy as np

from quietbridge.lips_adapter import LipsAdapter


def _adapter():
    pts = np.array([[60.0 + i, 50.0 + (i % 5)] for i in range(40)], dtype=np.float32)
    obs = SimpleNamespace(pts=pts, outer_lips=pts[:20], inner_lips=pts[20:], anchors=pts[:5], mouth_open=0.05)
    adapter = object.__new__(LipsAdapter)
    adapter.tracker = SimpleNamespace(detect=lambda frame, ts: obs)
    adapter.recording = None
    adapter.last_tracking = adapter.last_mouth_open = adapter.record_start_ms = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None
    return adapter


def test_unconvertible_preroll_frame_is_skipped_not_stored_as_none():
    adapter = _adapter()
    good = np.zeros((120, 160, 3), dtype=np.uint8)
    adapter.process(good, 1000)
    adapter._preroll.append((np.zeros((120, 160, 7), dtype=np.uint8), 1033, adapter._preroll[0][2]))
    adapter.process(good, 1066)
    adapter.start(1066)
    rec = adapter.recording
    assert rec.ts == [1.0, 1.066]
    assert len(rec.grays) == len(rec.anchors) == len(rec.mouth_open) == 2
    assert all(gray is not None for gray in rec.grays)


def test_preroll_is_bounded_and_chronological():
    adapter = _adapter()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for ts in range(0, 2000, 33):
        adapter.process(frame, ts)
    adapter.start(1980)
    times = [round(t * 1000) for t in adapter.recording.ts]
    assert times == sorted(times)
    assert times and times[0] >= 1980 - adapter.pre_roll_ms and times[-1] <= 1980

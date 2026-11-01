"""Tests for hackathon.signs.lip_lab.

The LipsAdapter is faked at the LipLab boundary; no real model, camera or
recognition accuracy is exercised here. Synchronization uses
``threading.Event`` / the lab's condition variable, not sleep-polling.
"""
import threading

import pytest

from hackathon.contracts import LipResult
from hackathon.signs.lip_lab import LipLab


class FakeAdapter:
    """Stands in for LipsAdapter; records calls and returns fixed candidates."""

    def __init__(self, candidates=("hello world",), decode_ms=7,
                 finish_gate: threading.Event | None = None,
                 finish_error: Exception | None = None):
        self.started = threading.Event()
        self.closed = threading.Event()
        self.finish_called = threading.Event()
        self.processed = threading.Event()  # set per process() call; tests clear it
        self.frames = []
        self.start_calls = 0
        self.close_calls = 0
        self._candidates = list(candidates)
        self._decode_ms = decode_ms
        self._finish_gate = finish_gate
        self._finish_error = finish_error

    def start(self):
        self.start_calls += 1
        self.started.set()

    def process(self, frame_bgr, timestamp_ms: int):
        self.frames.append((frame_bgr, timestamp_ms))
        self.processed.set()

    def finish(self) -> LipResult:
        self.finish_called.set()
        if self._finish_gate is not None:
            self._finish_gate.wait(timeout=5)
        if self._finish_error is not None:
            raise self._finish_error
        return LipResult(list(self._candidates), self._decode_ms)

    def close(self):
        self.close_calls += 1
        self.closed.set()


def wait_for(lab: LipLab, **state):
    """Block until private state matches, e.g. wait_for(lab, status='ready')."""
    with lab._cond:
        ok = lab._cond.wait_for(
            lambda: all(getattr(lab, f"_{k}") == v for k, v in state.items()),
            timeout=5)
    assert ok, f"timed out waiting for {state}: {lab.snapshot()}"


def make_lab(adapter=None, **kwargs):
    holder = []

    def factory():
        a = adapter if adapter is not None else FakeAdapter()
        holder.append(a)
        return a

    lab = LipLab(adapter_factory=factory, **kwargs)
    return lab, holder


def test_disabled_by_default_and_no_auto_load():
    lab, holder = make_lab()
    snap = lab.snapshot()
    assert snap["status"] == "disabled"
    assert snap["phase"] == "idle"
    assert snap["recording"] is False
    assert snap["candidates"] == []
    assert snap["mouth_box"] is None
    lab.feed(object(), 1)  # no-op while disabled
    with pytest.raises(RuntimeError):
        lab.start()
    assert holder == []  # adapter never constructed without enable()
    lab.close()


def test_enable_loads_once_and_reenable_is_noop():
    lab, holder = make_lab()
    lab.enable()
    wait_for(lab, status="ready")
    lab.enable()  # must not spawn a second worker or reload the model
    wait_for(lab, status="ready")
    assert len(holder) == 1
    lab.close()
    assert holder[0].closed.wait(timeout=2)
    assert holder[0].close_calls == 1


def test_record_stop_publishes_candidates():
    adapter = FakeAdapter(candidates=("hello world", "hello"), decode_ms=12)
    lab, _ = make_lab(adapter, tail_seconds=0)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    assert lab.snapshot()["recording"] is True
    lab.feed("f1", 1000)
    assert adapter.processed.wait(timeout=5)
    adapter.processed.clear()
    lab.feed("f2", 1001)
    assert adapter.processed.wait(timeout=5)
    adapter.processed.clear()
    # Out-of-order timestamp must be dropped: f3 proves the worker moved past
    # the stale frame (queue keeps only the latest, in FIFO order).
    lab.feed("stale", 1000)
    lab.feed("f3", 1002)
    assert adapter.processed.wait(timeout=5)
    assert [ts for _, ts in adapter.frames] == [1000, 1001, 1002]
    assert adapter.frames[0][0] == "f1"  # frame passed through, not copied
    lab.stop()
    wait_for(lab, phase="idle")
    assert adapter.finish_called.is_set()
    snap = lab.snapshot()
    assert snap["candidates"] == ["hello world", "hello"]
    assert snap["decode_ms"] == 12
    assert isinstance(snap["release_to_text_ms"], int) and snap["release_to_text_ms"] >= 0
    assert snap["recording"] is False
    assert snap["error"] == ""
    lab.close()


def test_stop_tail_keeps_processing_until_deadline():
    adapter = FakeAdapter()
    lab, _ = make_lab(adapter, tail_seconds=0.3)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    lab.feed("f1", 1000)
    assert adapter.processed.wait(timeout=5)
    adapter.processed.clear()
    lab.stop()
    wait_for(lab, phase="tail")
    lab.feed("tail-frame", 1001)  # tail frames still reach the adapter
    assert adapter.processed.wait(timeout=5)
    wait_for(lab, phase="idle")
    snap = lab.snapshot()
    assert snap["candidates"] == ["hello world"]
    # release_to_text starts at release and includes the tail, not capture time
    assert snap["release_to_text_ms"] >= 300
    lab.close()


def test_max_utterance_auto_stops():
    adapter = FakeAdapter()
    lab, _ = make_lab(adapter, tail_seconds=0, max_utterance_seconds=0.05)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    assert adapter.finish_called.wait(timeout=5)  # auto stop+finish, no stop() call
    wait_for(lab, phase="idle")
    assert lab.snapshot()["candidates"] == ["hello world"]
    lab.close()


def test_cancel_during_recording_discards_result():
    adapter = FakeAdapter()
    lab, _ = make_lab(adapter, tail_seconds=0)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    lab.feed("f1", 1000)
    assert adapter.processed.wait(timeout=5)
    lab.cancel()
    wait_for(lab, phase="idle")
    snap = lab.snapshot()
    assert snap["candidates"] == []
    assert snap["release_to_text_ms"] is None
    assert not adapter.finish_called.is_set()  # cancel drops the buffer without decoding
    lab.close()


def test_cancel_during_decode_drops_stale_result():
    gate = threading.Event()  # holds finish() so cancel lands mid-decode
    adapter = FakeAdapter(finish_gate=gate)
    lab, _ = make_lab(adapter, tail_seconds=0)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    lab.feed("f1", 1000)
    lab.stop()
    assert adapter.finish_called.wait(timeout=5)
    wait_for(lab, phase="decoding")
    lab.cancel()
    gate.set()  # finish returns only after the cancel
    wait_for(lab, phase="idle")
    snap = lab.snapshot()
    assert snap["candidates"] == []  # stale result must not be published
    assert snap["release_to_text_ms"] is None
    lab.close()


def test_missing_model_reports_unavailable():
    def bad_factory():
        raise RuntimeError("weights missing")

    lab = LipLab(adapter_factory=bad_factory)
    lab.enable()
    wait_for(lab, status="unavailable")
    snap = lab.snapshot()
    assert "weights missing" in snap["error"]
    assert snap["candidates"] == []
    with pytest.raises(RuntimeError):
        lab.start()
    lab.close()


def test_finish_error_surfaces_without_fake_candidates():
    adapter = FakeAdapter(finish_error=ValueError("Not enough visible mouth frames"))
    lab, _ = make_lab(adapter, tail_seconds=0)
    lab.enable()
    wait_for(lab, status="ready")
    lab.start()
    wait_for(lab, phase="recording")
    lab.feed("f1", 1000)
    lab.stop()
    wait_for(lab, phase="idle")
    snap = lab.snapshot()
    assert snap["status"] == "ready"
    assert "Not enough visible mouth frames" in snap["error"]
    assert snap["candidates"] == []
    lab.close()


def test_control_validates_state():
    lab, _ = make_lab()
    with pytest.raises(RuntimeError):
        lab.stop()
    with pytest.raises(RuntimeError):
        lab.cancel()
    lab.enable()
    wait_for(lab, status="ready")
    with pytest.raises(RuntimeError):
        lab.stop()  # idle, nothing recording
    lab.start()
    wait_for(lab, phase="recording")
    with pytest.raises(RuntimeError):
        lab.start()  # double start
    lab.cancel()
    wait_for(lab, phase="idle")
    lab.close()

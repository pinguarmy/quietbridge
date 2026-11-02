"""Standalone lip-reading lab for the gesture page.

Optional and lazy: no model is loaded until ``enable()`` is called. A single
daemon worker thread owns the ``LipsAdapter`` so every adapter call
(``start``/``process``/``finish``/``close``) stays on that one thread.
Control methods (``start``/``stop``/``cancel``) only validate state and
enqueue a command; they never block on the model.

Mouth boxes are intentionally not reported: the adapter's stored anchors are
face-alignment points, not a precise lip region, so ``mouth_box`` stays None.
"""
from __future__ import annotations

import queue
import threading
import time

from hackathon.lips_adapter import LipsAdapter

MAX_UTTERANCE_S = 30.0
TAIL_S = 0.4


class LipLab:
    def __init__(self, adapter_factory=None, tail_seconds: float = TAIL_S,
                 max_utterance_seconds: float = MAX_UTTERANCE_S):
        self._adapter_factory = adapter_factory or LipsAdapter
        self._tail_seconds = tail_seconds
        self._max_utterance = max_utterance_seconds
        self._cond = threading.Condition()
        self._status = "disabled"  # disabled | warming | ready | unavailable
        self._phase = "idle"  # idle | recording | tail | decoding
        self._error = ""
        self._candidates: list[str] = []
        self._decode_ms = None
        self._release_to_text_ms = None
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._control: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Worker-owned below; _gen is also bumped by cancel() to invalidate
        # any in-flight finish.
        self._gen = 0
        self._record_start = 0.0
        self._tail_deadline = 0.0
        self._released_at = 0.0
        self._last_ts = -1

    # ---- HTTP-facing control; all non-blocking ----

    def enable(self):
        """Start the worker that loads the model. No-op if already running."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._status = "warming"
            self._error = ""
            self._thread = threading.Thread(target=self._run, name="lip-lab", daemon=True)
            self._thread.start()
            self._cond.notify_all()

    def start(self):
        with self._cond:
            if self._status != "ready":
                raise RuntimeError(f"lip lab not ready (status={self._status})")
            if self._phase != "idle":
                raise RuntimeError(f"already recording (phase={self._phase})")
        self._control.put("start")

    def stop(self):
        with self._cond:
            if self._phase != "recording":
                raise RuntimeError(f"not recording (phase={self._phase})")
        self._control.put("stop")

    def cancel(self):
        with self._cond:
            if self._status != "ready" or self._phase == "idle":
                raise RuntimeError(
                    f"nothing to cancel (status={self._status}, phase={self._phase})")
            # Invalidate any result that an in-flight finish may return later.
            self._gen += 1
            self._candidates = []
            self._cond.notify_all()
        self._control.put("cancel")

    def feed(self, frame_bgr, timestamp_ms: int):
        """Offer the latest BGR frame; kept only while recording or in tail."""
        with self._cond:
            if self._status != "ready" or self._phase not in ("recording", "tail"):
                return
        try:
            self._frames.get_nowait()  # drop the older frame, keep only latest
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait((frame_bgr, timestamp_ms))
        except queue.Full:
            pass

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "status": self._status,
                "phase": self._phase,
                "candidates": list(self._candidates),
                "decode_ms": self._decode_ms,
                "error": self._error,
                "mouth_box": None,
                "release_to_text_ms": self._release_to_text_ms,
                "recording": self._phase in ("recording", "tail"),
            }

    def close(self):
        """Ask the worker to exit; the worker closes the adapter exactly once."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    # ---- worker thread: sole owner of the adapter ----

    def _run(self):
        adapter = None
        try:
            try:
                adapter = self._adapter_factory()
            except Exception as exc:  # missing model etc: explicit error, no fallback
                with self._cond:
                    self._status = "unavailable"
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._cond.notify_all()
                return
            with self._cond:
                self._status = "ready"
                self._cond.notify_all()
            while not self._stop_event.is_set():
                self._drain_control(adapter)
                now = time.perf_counter()
                if self._phase == "recording" and now - self._record_start >= self._max_utterance:
                    self._begin_tail(now)
                if self._phase == "tail" and now >= self._tail_deadline:
                    self._finish(adapter)
                    continue
                timeout = 0.05 if self._phase in ("recording", "tail") else 0.1
                try:
                    frame, ts = self._frames.get(timeout=timeout)
                except queue.Empty:
                    continue
                if self._phase not in ("recording", "tail") or ts <= self._last_ts:
                    continue  # stale or out-of-order frame; timestamps must move forward
                self._last_ts = ts
                try:
                    adapter.process(frame, ts)
                except Exception as exc:
                    self._abort(adapter, f"{type(exc).__name__}: {exc}")
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass

    def _drain_control(self, adapter):
        while True:
            try:
                cmd = self._control.get_nowait()
            except queue.Empty:
                return
            if cmd == "start":
                self._begin_recording(adapter)
            elif cmd == "stop":
                if self._phase == "recording":
                    self._begin_tail(time.perf_counter())
            elif cmd == "cancel":
                self._abort(adapter, "")

    def _begin_recording(self, adapter):
        with self._cond:
            if self._phase != "idle":
                return
            self._gen += 1
            self._candidates = []
            self._decode_ms = None
            self._release_to_text_ms = None
            self._record_start = time.perf_counter()
            self._phase = "recording"
            self._cond.notify_all()
        adapter.start()

    def _begin_tail(self, now: float):
        with self._cond:
            self._released_at = now
            self._tail_deadline = now + self._tail_seconds
            self._phase = "tail"
            self._cond.notify_all()

    def _finish(self, adapter):
        with self._cond:
            gen = self._gen
            self._phase = "decoding"
            self._cond.notify_all()
        try:
            result = adapter.finish()
            error = ""
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        with self._cond:
            if gen == self._gen:  # not cancelled while decoding
                if result is not None:
                    self._candidates = list(result.candidates)
                    self._decode_ms = result.decode_ms
                    self._release_to_text_ms = round(
                        (time.perf_counter() - self._released_at) * 1000)
                    self._error = ""
                else:
                    self._error = error
            self._phase = "idle"
            self._cond.notify_all()

    def _abort(self, adapter, error: str):
        # This adapter's buffer is public; dropping it does not require an
        # expensive decode. Cancel must never run a new recognition attempt.
        adapter.recording = None
        with self._cond:
            if error:
                self._error = error
            self._candidates = []
            self._phase = "idle"
            self._cond.notify_all()

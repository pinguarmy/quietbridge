"""One camera, independent latest-frame workers, revision-bound confirmation."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import importlib
import queue
import threading
import time

from .contracts import Decision, LipResult, SignEvent
from .fusion import fuse, load_vocabulary, usable_signs
from .speech import Speaker


def clock_ms():
    return int(time.perf_counter() * 1000)


def put_latest(q, value):
    try:
        q.put_nowait(value)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(value)
        except queue.Full:
            pass


class Runtime:
    def __init__(self, *, rehearsal=False, camera=0, enable_lips=True, vocabulary=None):
        self.rehearsal = rehearsal
        self.camera_source = camera
        self.enable_lips = enable_lips
        self.vocabulary = load_vocabulary() if vocabulary is None else vocabulary
        self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                   if rehearsal else self.vocabulary)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.hand_frames = queue.Queue(maxsize=1)
        self.lip_frames = queue.Queue(maxsize=1)
        self.events = deque(maxlen=100)
        self.seen = deque(maxlen=200)
        self.jpeg = None
        self.speaker = Speaker()
        self.threads = []
        self.record_start = None
        self.release_ms = None
        self.finish_at = None
        self.revision = 0
        self.confirmed_revision = None
        self.state = {
            "mode": "combined" if enable_lips else "signs", "phase": "idle",
            "camera": "rehearsal" if rehearsal else "starting",
            "lips": "rehearsal" if rehearsal else ("warming" if enable_lips else "disabled"),
            "signs": "rehearsal" if rehearsal else "loading",
            "camera_error": "", "lip_error": "", "sign_error": "", "error": "",
            "decision": asdict(Decision("", "unavailable", "sign_only", "local")),
            "candidates": [], "alternatives": [], "last_sign": None,
            "metrics": {}, "last_spoken": "", "recording": False,
        }

    def start(self):
        if self.rehearsal:
            return
        for name, target in [("capture", self._capture), ("hands", self._hands),
                             *([("lips", self._lips)] if self.enable_lips else [])]:
            thread = threading.Thread(target=target, name=f"demo-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def snapshot(self):
        import copy
        with self.lock:
            return copy.deepcopy({**self.state, "revision": self.revision,
                                  "confirmed": self.confirmed_revision == self.revision,
                                  "rehearsal": self.rehearsal, "speech_available": self.speaker.available,
                                  "vocabulary": self.vocabulary})

    def _publish(self, lips, events, now, since=None):
        start = time.perf_counter()
        decision = fuse(lips, events, self._fusion_vocabulary, now, since)
        self.revision += 1
        self.confirmed_revision = None
        self.state["decision"] = asdict(decision)
        self.state["candidates"] = lips.candidates if lips else []
        choices = self.state["candidates"] + [self._fusion_vocabulary[e.label]["phrase"] for e in
                    usable_signs(events, self._fusion_vocabulary, now, since)]
        self.state["alternatives"] = list(dict.fromkeys(c for c in [decision.text, *choices] if c))
        for metric in ("release_to_text_ms", "decode_ms"):
            self.state["metrics"].pop(metric, None)
        self.state["metrics"]["fusion_ms"] = round((time.perf_counter() - start) * 1000, 2)

    def receive_signs(self, events, now=None):
        now = clock_ms() if now is None else now
        with self.lock:
            for event in events:
                if not isinstance(event, SignEvent):
                    raise TypeError("Recognizer must return hackathon.contracts.SignEvent instances")
                key = (event.label, event.start_ms, event.end_ms)
                if key in self.seen:
                    continue
                valid = usable_signs([event], self._fusion_vocabulary, now)
                if not valid:
                    continue
                self.seen.append(key)
                self.events.append(event)
                self.state["last_sign"] = asdict(event)
                self.state["metrics"]["sign_event_age_ms"] = now - event.end_ms
                if self.state["mode"] != "lips" and self.state["phase"] == "idle":
                    self._publish(None, [event], now)

    def action(self, name, data):
        with self.lock:
            self.state["error"] = ""
            if name == "mode":
                if self.state["phase"] != "idle":
                    raise ValueError("Finish the current recording before changing mode")
                if data.get("mode") not in {"signs", "lips", "combined"}:
                    raise ValueError("Unknown input mode")
                self.state["mode"] = data["mode"]
                self.events.clear()
                self.state["last_sign"] = None
                self._publish(None, [], clock_ms())
            elif name == "start":
                if self.rehearsal:
                    raise ValueError("Use synthetic input controls in rehearsal mode")
                if self.state["mode"] == "signs":
                    raise ValueError("Select lip or combined mode to record lips")
                if self.state["phase"] != "idle":
                    raise ValueError("A recording or decode is already in progress")
                if not self.rehearsal and (self.state["lips"] != "ready" or self.state["camera"] != "ready"):
                    raise ValueError("Wait for the camera and lip model to be ready")
                self.record_start = clock_ms()
                self.release_ms = self.finish_at = None
                self._publish(None, [], self.record_start)
                self.state.update(phase="recording", recording=True, lip_error="")
            elif name == "stop":
                if self.state["phase"] != "recording":
                    return  # A late key/pointer release after timeout or cancel is harmless.
                self.release_ms = clock_ms()
                self.finish_at = self.release_ms + 400
                self.state.update(phase="tail", recording=False)
            elif name == "cancel":
                self.record_start = self.release_ms = self.finish_at = None
                self.state.update(phase="idle", recording=False)
                self._publish(None, [], clock_ms())
            elif name == "confirm":
                self._check_revision(data)
                text = data.get("text", "")
                if not text or text not in self.state["alternatives"]:
                    raise ValueError("Select one of the displayed alternatives")
                self.state["decision"]["text"] = text
                self.state["decision"]["status"] = "accepted"
                self.confirmed_revision = self.revision
            elif name == "speak":
                self._check_revision(data)
                decision = self.state["decision"]
                if decision["status"] != "accepted" or not decision["text"]:
                    raise ValueError("Confirm a displayed alternative before speaking")
                if data.get("text") != decision["text"]:
                    raise ValueError("The message changed. Check it again before speaking")
                started = time.perf_counter()
                self.speaker.speak(decision["text"])
                self.state["last_spoken"] = decision["text"]
                self.state["metrics"]["tts_dispatch_ms"] = round((time.perf_counter() - started) * 1000, 2)
            elif name == "rehearse":
                if not self.rehearsal:
                    raise ValueError("Synthetic inputs are available only with --rehearsal")
                text = data.get("text", "")
                label = data.get("label", "")
                if not isinstance(text, str) or len(text) > 2000 or label not in {"", *self.vocabulary}:
                    raise ValueError("Invalid rehearsal input")
                now = clock_ms()
                events = [SignEvent(label, 0.95, now - 500, now, True)] if label else []
                self.state["last_sign"] = asdict(events[0]) if events else None
                self._publish(LipResult([text], 0) if text.strip() else None, events, now)
            else:
                raise ValueError("Unknown action")

    def _check_revision(self, data):
        if data.get("revision") != self.revision:
            raise ValueError("The result changed; review the current text first")

    def _capture(self):
        cap = None
        try:
            import cv2
            import sys
            source = self.camera_source
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(source, int) else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                raise RuntimeError("Camera unavailable; allow Camera access and close other camera apps")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 30)
            last_preview = 0
            last_frame = time.perf_counter()
            while not self.stop.is_set():
                ok, frame = cap.read()
                now = clock_ms()
                if not ok:
                    if time.perf_counter() - last_frame > 2:
                        raise RuntimeError("Camera stopped delivering frames")
                    self.stop.wait(0.01)
                    continue
                last_frame = time.perf_counter()
                # Frames are read-only for consumers; preview processing uses a new array.
                put_latest(self.hand_frames, (frame, now))
                put_latest(self.lip_frames, (frame, now))
                with self.lock:
                    self.state["camera"] = "ready"
                if now - last_preview >= 100:
                    ok, encoded = cv2.imencode(".jpg", cv2.flip(frame, 1), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self.lock:
                            self.jpeg = encoded.tobytes()
                    last_preview = now
        except Exception as exc:
            with self.lock:
                self.state.update(camera="unavailable", camera_error=str(exc))
        finally:
            if cap is not None:
                cap.release()

    def _hands(self):
        recognizer = None
        try:
            module = importlib.import_module("hackathon.signs.recognizer")
            recognizer = module.SignRecognizer()
            with self.lock:
                self.state["signs"] = "ready"
            last = 0
            while not self.stop.is_set():
                try:
                    frame, ts = self.hand_frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if ts - last < 66:
                    continue
                last = ts
                started = time.perf_counter()
                events = recognizer.process(frame, ts)
                self.receive_signs(events)
                with self.lock:
                    self.state["metrics"]["hand_process_ms"] = round((time.perf_counter() - started) * 1000, 2)
        except Exception as exc:
            with self.lock:
                self.state.update(signs="unavailable", sign_error=str(exc))
        finally:
            if recognizer is not None:
                recognizer.close()

    def _lips(self):
        adapter = None
        try:
            from .lips_adapter import LipsAdapter
            adapter = LipsAdapter()
            with self.lock:
                self.state["lips"] = "ready"
            active_start = None
            while not self.stop.is_set():
                packet = None
                try:
                    packet = self.lip_frames.get(timeout=0.05)
                except queue.Empty:
                    pass
                with self.lock:
                    start, phase, end = self.record_start, self.state["phase"], self.finish_at
                if phase in {"recording", "tail"} and start != active_start:
                    adapter.start()
                    active_start = start
                if phase == "idle":
                    adapter.recording = None
                    active_start = None
                    continue
                if active_start is None:
                    continue
                if packet and packet[1] >= active_start and (end is None or packet[1] <= end):
                    adapter.process(*packet)
                now = clock_ms()
                with self.lock:
                    if self.record_start != active_start:
                        continue
                    if self.state["phase"] == "recording" and now - active_start >= 10000:
                        self.release_ms = now
                        self.finish_at = now + 400
                        self.state.update(phase="tail", recording=False)
                    if self.state["phase"] != "tail" or now < self.finish_at:
                        continue
                    self.state["phase"] = "decoding"
                    released = self.release_ms
                try:
                    result = adapter.finish()
                    with self.lock:
                        if self.record_start != active_start or self.state["phase"] != "decoding":
                            continue  # cancelled/changed while inference ran
                        events = list(self.events) if self.state["mode"] == "combined" else []
                        self._publish(result, events, self.finish_at, active_start)
                        self.state["metrics"].update(decode_ms=result.decode_ms,
                            release_to_text_ms=clock_ms() - released)
                except Exception as exc:
                    with self.lock:
                        if self.record_start == active_start:
                            self.state["lip_error"] = str(exc)
                            events = list(self.events) if self.state["mode"] == "combined" else []
                            self._publish(None, events, self.finish_at, active_start)
                finally:
                    with self.lock:
                        if self.record_start == active_start:
                            self.state.update(phase="idle", recording=False)
                            self.record_start = None
                    active_start = None
        except Exception as exc:
            with self.lock:
                self.state.update(lips="unavailable", lip_error=str(exc), phase="idle", recording=False)
        finally:
            if adapter is not None:
                adapter.close()

    def close(self):
        self.stop.set()
        self.speaker.close()
        for thread in self.threads:
            thread.join(timeout=1)

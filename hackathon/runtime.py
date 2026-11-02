"""One camera, independent latest-frame workers, revision-bound confirmation."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import importlib
import queue
import threading
import time

from .cleanup import CleanupClient
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
    def __init__(self, *, rehearsal=False, browser_camera=False, camera=0, enable_lips=True, vocabulary=None,
                 lip_model=None, lip_language_model=None, motion_model=None, profile_name=None,
                 configuration_warning="", max_utterance_seconds=30.0):
        if rehearsal and browser_camera:
            raise ValueError("Browser camera and rehearsal modes are mutually exclusive")
        self.rehearsal = rehearsal
        self.browser_camera = browser_camera
        self.camera_source = camera
        self.enable_lips = enable_lips
        self.lip_model_path = lip_model
        self.lip_language_model_path = lip_language_model
        self.motion_model_path = motion_model
        self.max_utterance_ms = round(float(max_utterance_seconds) * 1000)
        if self.max_utterance_ms < 1000:
            raise ValueError("Maximum utterance length must be at least one second")
        self.vocabulary = load_vocabulary() if vocabulary is None else vocabulary
        self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                   if rehearsal else self.vocabulary)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.hand_frames = queue.Queue(maxsize=1)
        self.lip_frames = queue.Queue(maxsize=1)
        self.hand_control = queue.Queue()
        self.frame_times = deque()
        self.events = deque(maxlen=100)
        self.seen = deque(maxlen=200)
        self.jpeg = None
        self._last_frame_timestamp_ms = None
        self.tracking = {"hands": [], "mouth": None}
        self.speaker = Speaker()
        self.threads = []
        self.record_start = None
        self.release_ms = None
        self.finish_at = None
        self.window_generation = 0
        self.record_generation = None
        self._window_motion = False
        self._window_motion_done = False
        self.revision = 0
        self.confirmed_revision = None
        self._cleanup_generation = 0
        self.cleanup = CleanupClient()
        self.state = {
            "mode": "combined" if enable_lips else "signs", "phase": "idle",
            "profile_name": profile_name or "Stock configuration",
            "profile_warning": configuration_warning,
            "camera": "rehearsal" if rehearsal else ("waiting" if browser_camera else "starting"),
            "lips": "rehearsal" if rehearsal else ("warming" if enable_lips else "disabled"),
            "signs": "rehearsal" if rehearsal else "loading",
            "sign_input": "rehearsal" if rehearsal else "loading",
            "lip_model": {"mode": "stock", "status": "rehearsal" if rehearsal else "loading"},
            "camera_error": "", "lip_error": "", "sign_error": "", "error": "",
            "decision": asdict(Decision("", "unavailable", "sign_only", "local")),
            "candidates": [], "candidate_scores": [], "alternatives": [], "last_sign": None,
            "metrics": {}, "last_spoken": "", "recording": False,
            "cleanup": {"status": "disabled", "suggestion": "", "latency_ms": 0,
                        "error": "", "source_revision": 0},
        }

    def start(self):
        if self.rehearsal:
            return
        workers = [] if self.browser_camera else [("capture", self._capture)]
        workers += [("hands", self._hands), *([("lips", self._lips)] if self.enable_lips else [])]
        for name, target in workers:
            thread = threading.Thread(target=target, name=f"demo-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({**self.state, "revision": self.revision,
                                  "confirmed": self.confirmed_revision == self.revision,
                                  "rehearsal": self.rehearsal, "browser_camera": self.browser_camera,
                                  "speech_available": self.speaker.available, "vocabulary": self.vocabulary})

    def tracking_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.tracking)

    def _publish(self, lips, events, now, since=None):
        start = time.perf_counter()
        decision = fuse(lips, events, self._fusion_vocabulary, now, since)
        self.revision += 1
        self.confirmed_revision = None
        self.state["decision"] = asdict(decision)
        self.state["candidates"] = lips.candidates if lips else []
        self.state["candidate_scores"] = lips.candidate_scores if lips else []
        choices = self.state["candidates"] + [self._fusion_vocabulary[e.label]["phrase"] for e in
                    usable_signs(events, self._fusion_vocabulary, now, since)]
        self.state["alternatives"] = list(dict.fromkeys(c for c in [decision.text, *choices] if c))
        self._cleanup_generation += 1
        generation, revision = self._cleanup_generation, self.revision
        enabled = self.cleanup.public_config()["enabled"]
        self.state["cleanup"] = {"status": "idle" if enabled else "disabled", "suggestion": "",
                                 "latency_ms": 0, "error": "", "source_revision": revision}
        candidates = self.state["alternatives"]
        labels = [event.label for event in usable_signs(events, self._fusion_vocabulary, now, since)]
        if candidates and enabled:
            self.state["cleanup"]["status"] = "pending"
            if not self.cleanup.submit(generation, revision, candidates, labels, self._cleanup_result):
                self.state["cleanup"]["status"] = "idle"
        for metric in ("release_to_text_ms", "decode_ms", "lip_capture_fps",
                       "lip_raw_frames", "lip_max_frame_gap_ms"):
            self.state["metrics"].pop(metric, None)
        if lips is not None:
            self.state["metrics"].update(
                lip_capture_fps=lips.capture_fps,
                lip_raw_frames=lips.raw_frames,
                lip_max_frame_gap_ms=lips.max_frame_gap_ms,
            )
        self.state["metrics"]["fusion_ms"] = round((time.perf_counter() - start) * 1000, 2)

    def _cleanup_result(self, generation, revision, result):
        with self.lock:
            if self.stop.is_set() or generation != self._cleanup_generation or revision != self.revision:
                return
            self.state["cleanup"] = {**result, "source_revision": revision}
            suggestion = result.get("suggestion", "")
            if result.get("status") == "ready" and suggestion and suggestion not in self.state["alternatives"]:
                self.state["alternatives"].append(suggestion)

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
                mode = self.state["mode"]
                motion = self.state["sign_input"] == "motion"
                if mode == "signs" and not motion:
                    raise ValueError("This gesture mode is automatic; only custom window mode uses Record")
                if self.state["phase"] != "idle":
                    raise ValueError("A recording or decode is already in progress")
                if self.state["camera"] != "ready":
                    raise ValueError("Wait for the camera to be ready")
                if mode in {"lips", "combined"} and self.state["lips"] != "ready":
                    raise ValueError("Wait for the lip model to be ready")
                if mode == "signs" and self.state["signs"] != "ready":
                    raise ValueError("Wait for the custom gesture model to be ready")
                self.window_generation += 1
                self.record_generation = self.window_generation
                self.record_start = clock_ms()
                self.release_ms = self.finish_at = None
                self._window_motion = motion and mode != "lips"
                self._window_motion_done = not self._window_motion
                self.events.clear()
                self._publish(None, [], self.record_start)
                self.state.update(phase="recording", recording=True, lip_error="", sign_error="")
                if self._window_motion:
                    self.hand_control.put(("start", self.record_start, self.record_generation))
            elif name == "stop":
                if self.state["phase"] != "recording":
                    return  # A late key/pointer release after timeout or cancel is harmless.
                self.release_ms = clock_ms()
                self.finish_at = self.release_ms + 400
                self.state.update(phase="tail", recording=False)
                if self._window_motion:
                    self.hand_control.put(("finish", self.finish_at, self.record_generation))
            elif name == "cancel":
                self.window_generation += 1
                cancelled_generation = self.record_generation
                self.record_generation = None
                if self.state["sign_input"] == "motion":
                    self.hand_control.put(("cancel", None, cancelled_generation))
                self.record_start = self.release_ms = self.finish_at = None
                self._window_motion = False
                self._window_motion_done = False
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

    def _note_frame_time(self, now):
        self.frame_times.append(now)
        while self.frame_times and self.frame_times[0] < now - 5000:
            self.frame_times.popleft()
        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            if elapsed > 0:
                self.state["metrics"]["camera_input_fps"] = round(
                    (len(self.frame_times) - 1) * 1000 / elapsed, 2)

    def receive_frame(self, frame_bgr, timestamp_ms=None):
        """Ingest one browser frame; return False for a repeated timestamp."""
        import cv2

        with self.lock:
            now = clock_ms() if timestamp_ms is None else timestamp_ms
            if self._last_frame_timestamp_ms is not None and now <= self._last_frame_timestamp_ms:
                return False
            self._last_frame_timestamp_ms = now
            self._note_frame_time(now)
            # Frames are read-only for consumers; preview processing uses a new array.
            put_latest(self.hand_frames, (frame_bgr, now))
            put_latest(self.lip_frames, (frame_bgr, now))
            ok, encoded = cv2.imencode(".jpg", cv2.flip(frame_bgr, 1),
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self.jpeg = encoded.tobytes()
            self.state["camera"] = "ready"
            return True

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
                    self._note_frame_time(now)
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
        active_generation = None
        try:
            module = importlib.import_module("hackathon.signs.recognizer")
            if self.motion_model_path is not None:
                recognizer = module.SignRecognizer(motion_model_path=self.motion_model_path)
            else:
                recognizer = module.SignRecognizer()
            motion = getattr(recognizer, "_motion", None) is not None
            motion_auto = motion and getattr(recognizer, "motion_auto", False)
            with self.lock:
                self.state.update(signs="ready",
                                  sign_input="motion_auto" if motion_auto else "motion" if motion else "stock")
            last = 0
            active_start = None
            active_generation = None
            pending_finish = None
            while not self.stop.is_set():
                while True:
                    try:
                        command, value, generation = self.hand_control.get_nowait()
                    except queue.Empty:
                        break
                    if command == "start":
                        with self.lock:
                            if generation != self.record_generation:
                                continue
                        recognizer.start(value)
                        active_start = value
                        active_generation = generation
                        pending_finish = None
                    elif command == "cancel":
                        recognizer.cancel()
                        active_start = active_generation = pending_finish = None
                    elif command == "finish" and generation == active_generation:
                        pending_finish = value

                packet = None
                try:
                    packet = self.hand_frames.get(timeout=0.05)
                except queue.Empty:
                    pass
                if packet is not None:
                    frame, ts = packet
                    with self.lock:
                        shared_end = (self.finish_at
                                      if self.record_generation == active_generation else None)
                    window_end = pending_finish if pending_finish is not None else shared_end
                    motion_boundary = getattr(recognizer, "_motion_last_boundary_ms", None)
                    if motion and ((motion_boundary is not None and ts < motion_boundary)
                                   or (active_start is not None and (ts < active_start
                                       or (window_end is not None and ts > window_end)))):
                        packet = None
                    if packet is not None and ts - last >= (40 if motion else 66):
                        last = ts
                        started = time.perf_counter()
                        events = recognizer.process(frame, ts)
                        with self.lock:
                            if hasattr(recognizer, "last_tracking"):
                                self.tracking["hands"] = recognizer.last_tracking
                                self.tracking["hands_timestamp_ms"] = ts
                        self.receive_signs(events)
                        with self.lock:
                            self.state["metrics"]["hand_process_ms"] = round(
                                (time.perf_counter() - started) * 1000, 2)

                now = clock_ms()
                with self.lock:
                    phase, end = self.state["phase"], self.finish_at
                    current = (active_generation is not None
                               and self.record_generation == active_generation)
                    if motion and current and phase == "recording" \
                            and now - active_start >= self.max_utterance_ms:
                        self.release_ms = now
                        self.finish_at = end = now + 400
                        self.state.update(phase="tail", recording=False)
                        pending_finish = end
                    if motion and current and phase == "tail" and pending_finish is None:
                        pending_finish = end
                if pending_finish is None or now < pending_finish:
                    continue
                events = recognizer.finish(pending_finish)
                with self.lock:
                    valid = (self.record_generation == active_generation
                             and self.record_start == active_start
                             and self.state["phase"] in {"tail", "decoding"})
                    if valid:
                        self._window_motion_done = True
                        # Keep the runtime lock through event storage and sign-only
                        # publication so cancel cannot turn a late finish into idle input.
                        self.receive_signs(events)
                        if self.state["mode"] == "signs":
                            self.state.update(phase="idle", recording=False)
                            self._publish(None, events, pending_finish, active_start)
                            self.record_start = self.release_ms = self.finish_at = None
                            self.record_generation = None
                active_start = active_generation = pending_finish = None
        except Exception as exc:
            with self.lock:
                self.tracking["hands"] = []
                self.tracking.pop("hands_timestamp_ms", None)
                self.state.update(signs="unavailable", sign_error=str(exc))
                if active_generation is not None and self.record_generation == active_generation:
                    # Let a combined lip result finish without this modality.
                    self._window_motion_done = True
                    if self.state["mode"] == "signs":
                        since = self.record_start
                        self.state.update(phase="idle", recording=False)
                        self._publish(None, [], clock_ms(), since)
                        self.record_start = self.release_ms = self.finish_at = None
                        self.record_generation = None
        finally:
            if recognizer is not None:
                recognizer.close()

    def _lips(self):
        adapter = None
        active_generation = None
        try:
            from .lips_adapter import LipsAdapter
            if self.lip_model_path is not None or self.lip_language_model_path is not None:
                adapter = LipsAdapter(model_path=self.lip_model_path,
                                      language_model_path=self.lip_language_model_path)
            else:
                adapter = LipsAdapter()
            with self.lock:
                self.state["lips"] = "ready"
                self.state["lip_model"] = {**getattr(adapter, "model_info", {"mode": "stock"}),
                                           "status": "ready"}
            active_start = None
            active_generation = None
            last_processed_ts = None
            while not self.stop.is_set():
                packet = None
                try:
                    packet = self.lip_frames.get(timeout=0.05)
                except queue.Empty:
                    pass
                with self.lock:
                    start, phase, end = self.record_start, self.state["phase"], self.finish_at
                    generation, mode = self.record_generation, self.state["mode"]
                if mode != "signs" and phase in {"recording", "tail"} and start != active_start:
                    adapter.start()
                    active_start = start
                    active_generation = generation
                if phase == "idle":
                    adapter.recording = None
                    active_start = active_generation = None
                if packet and packet[1] != last_processed_ts:
                    in_capture = (active_start is not None and packet[1] >= active_start
                                  and (end is None or packet[1] <= end))
                    if active_start is None or in_capture:
                        adapter.process(*packet)
                        last_processed_ts = packet[1]
                        with self.lock:
                            if hasattr(adapter, "last_tracking"):
                                self.tracking["mouth"] = adapter.last_tracking
                                self.tracking["mouth_timestamp_ms"] = packet[1]
                if active_start is None:
                    continue
                now = clock_ms()
                with self.lock:
                    if self.record_start != active_start or self.record_generation != active_generation:
                        continue
                    if self.state["phase"] == "recording" and now - active_start >= self.max_utterance_ms:
                        self.release_ms = now
                        self.finish_at = now + 400
                        self.state.update(phase="tail", recording=False)
                    if self.state["phase"] != "tail" or now < self.finish_at:
                        continue
                    if self._window_motion and not self._window_motion_done:
                        continue  # fusion waits for this window's motion result
                    self.state["phase"] = "decoding"
                    released = self.release_ms
                try:
                    result = adapter.finish()
                    with self.lock:
                        if (self.record_start != active_start
                                or self.record_generation != active_generation
                                or self.state["phase"] != "decoding"):
                            continue  # cancelled/changed while inference ran
                        events = list(self.events) if self.state["mode"] == "combined" else []
                        self._publish(result, events, self.finish_at, active_start)
                        self.state["metrics"].update(decode_ms=result.decode_ms,
                            release_to_text_ms=clock_ms() - released)
                except Exception as exc:
                    with self.lock:
                        if (self.record_start == active_start
                                and self.record_generation == active_generation):
                            self.state["lip_error"] = str(exc)
                            events = list(self.events) if self.state["mode"] == "combined" else []
                            self._publish(None, events, self.finish_at, active_start)
                finally:
                    with self.lock:
                        if (self.record_start == active_start
                                and self.record_generation == active_generation):
                            self.state.update(phase="idle", recording=False)
                            self.record_start = None
                            self.record_generation = None
                    active_start = active_generation = None
        except Exception as exc:
            with self.lock:
                self.tracking["mouth"] = None
                self.tracking.pop("mouth_timestamp_ms", None)
                self.state.update(lips="unavailable", lip_error=str(exc), phase="idle", recording=False)
                if active_generation is not None and self.record_generation == active_generation:
                    if self._window_motion:
                        self.hand_control.put(("cancel", None, active_generation))
                    self.record_start = self.release_ms = self.finish_at = None
                    self.record_generation = None
                    self._window_motion = False
        finally:
            if adapter is not None:
                adapter.close()

    def close(self):
        self.stop.set()
        self.speaker.close()
        self.cleanup.close()
        for thread in self.threads:
            thread.join(timeout=1)

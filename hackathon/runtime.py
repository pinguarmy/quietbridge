"""One camera, independent latest-frame workers, revision-bound confirmation."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import importlib
import os
import queue
import threading
import time

from .cleanup import CleanupClient
from .contracts import Decision, LipResult, SignEvent
from .fusion import fuse, load_vocabulary, usable_signs
from .lip_activity import LipActivityConfig, LipActivityDetector, ActivityState, ActivityAction
from .speech import Speaker
from .session_transcript import SessionTranscript


def clock_ms():
    return int(time.perf_counter() * 1000)


def put_latest(q, value):
    dropped = False
    try:
        q.put_nowait(value)
    except queue.Full:
        dropped = True
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(value)
        except queue.Full:
            pass
    return dropped


MOTION_PHRASES = {"HELLO": "Hi", "WELCOME": "Welcome", "OPENAI": "OpenAI",
                  "DDE": "DEE", "HACKATHON": "Hackathon"}


class _MotionWindow:
    """Experimental sliding-window adapter; no images or training samples saved."""
    def __init__(self, model):
        self.model = model
        self.frames = deque(maxlen=240)
        self.last_check = None
        self.latched = None
        self.quiet_since = None
        self.candidate = None
        self.streak = 0

    def update(self, hands, timestamp_ms):
        import numpy as np
        landmarks = np.zeros((2, 21, 3), dtype=np.float32)
        present = np.zeros(2, dtype=bool)
        handedness = np.full(2, "", dtype="U5")
        for slot, hand in enumerate(hands[:2]):
            points = np.asarray(hand.get("landmarks", []), dtype=np.float32)
            if points.shape == (21, 3):
                landmarks[slot] = points
                present[slot] = True
                handedness[slot] = hand.get("handedness", "")
        self.frames.append((timestamp_ms, landmarks, present, handedness))
        while self.frames and timestamp_ms - self.frames[0][0] > 4000:
            self.frames.popleft()
        if (len(self.frames) < 20 or timestamp_ms - self.frames[0][0] < 3000
                or (self.last_check is not None and timestamp_ms - self.last_check < 1000)):
            return []
        self.last_check = timestamp_ms
        trial = {"timestamps_ms": np.array([f[0] for f in self.frames]),
                 "landmarks": np.stack([f[1] for f in self.frames]),
                 "hand_present": np.stack([f[2] for f in self.frames]),
                 "handedness": np.stack([f[3] for f in self.frames])}
        label, score = self.model.predict(trial)
        if label not in MOTION_PHRASES or not np.isfinite(score) or score < .75:
            self.candidate, self.streak = None, 0
            if self.quiet_since is None:
                self.quiet_since = timestamp_ms
            elif timestamp_ms - self.quiet_since >= 1000:
                self.latched = None
            return []
        self.quiet_since = None
        self.streak = self.streak + 1 if label == self.candidate else 1
        self.candidate = label
        if self.streak < 2 or self.latched is not None:
            return []
        self.latched = label
        return [SignEvent(label, float(score), self.frames[0][0], timestamp_ms, True)]


class Runtime:
    def __init__(self, *, rehearsal=False, browser_camera=False, camera=0, enable_lips=True, vocabulary=None,
                 lip_model=None, lip_language_model=None, motion_model=None, profile_name=None,
                 profile=None, configuration_warning="", max_utterance_seconds=30.0,
                 lip_activity_config=None):
        if rehearsal and browser_camera:
            raise ValueError("Browser camera and rehearsal modes are mutually exclusive")
        self.rehearsal = rehearsal
        self.browser_camera = browser_camera
        self.camera_source = camera
        self.profile = profile
        self.enable_lips = enable_lips
        self.lip_model_path = lip_model
        self.lip_language_model_path = lip_language_model
        self.motion_model_path = motion_model if motion_model is not None else (os.environ.get("QUIETBRIDGE_MOTION_MODEL", "").strip() or None)
        self.max_utterance_ms = round(float(max_utterance_seconds) * 1000)
        if self.max_utterance_ms < 1000:
            raise ValueError("Maximum utterance length must be at least one second")
        self.lip_activity_config = lip_activity_config or LipActivityConfig()
        self._lip_activity = LipActivityDetector(self.lip_activity_config)
        self.vocabulary = load_vocabulary() if vocabulary is None else vocabulary
        if self.motion_model_path and not rehearsal and vocabulary is None:
            # Stock pose meanings must never leak into the custom motion session.
            self.vocabulary = {label: {"display_name": phrase, "phrase": phrase,
                               "matches": [phrase], "enabled": True, "verified": False}
                               for label, phrase in MOTION_PHRASES.items()}
        self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                   if rehearsal else self.vocabulary)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        # Idle preview keeps only the latest frame. Active capture uses a bounded
        # chronological buffer (90 decoded frames) and reports every overflow.
        self.hand_frames = queue.Queue(maxsize=90)
        self.lip_frames = queue.Queue(maxsize=90)
        self.capture_dropped_frames = 0
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
        self._camera_session_active = None  # None preserves the manual protocol.
        self._camera_generation = 0
        self._record_generation = 0
        self._auto_open_since = self._auto_quiet_since = None
        self._auto_open_frames = 0
        self._starting_auto_chunk = False
        self.cleanup = CleanupClient()
        self.session = SessionTranscript()
        self.session_signs = []  # Stable events validated when received in this session.
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
            "metrics": {}, "last_spoken": "", "last_correction": "", "recording": False,
            "auto_recognition": False,
            "session_active": False, "session_finalizing": False,
            "session_text": "", "session_chunks": [], "session_gestures": [],
            "gesture_model": "personal motion-v2 (experimental)" if self.motion_model_path else "stock",
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

    def cleanup_config(self):
        return self.cleanup.public_config()

    def configure_cleanup(self, data):
        with self.lock:
            config = self.cleanup.configure(data)
            self._cleanup_generation += 1
            self.state["cleanup"] = {"status": "idle" if config["enabled"] else "disabled",
                                     "suggestion": "", "latency_ms": 0, "error": "",
                                     "source_revision": self.revision}
            return config

    def test_cleanup(self):
        # Deliberately outside the runtime lock: camera and state remain responsive.
        return self.cleanup.test()

    def _publish(self, lips, events, now, since=None, cleanup_labels=None):
        if self.state["session_active"] and lips is None and self.session.text:
            lips = LipResult([self.session.text], 0)
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
        self._cleanup_generation += 1
        labels = list(dict.fromkeys(e.label for e in usable_signs(events, self._fusion_vocabulary, now, since)))
        if cleanup_labels is not None:
            labels = list(dict.fromkeys([*labels, *cleanup_labels]))
        candidates = list(lips.candidates) if lips is not None else []
        pending = bool(candidates) and not self.state["session_active"] and self.cleanup.submit(
            self._cleanup_generation, self.revision, candidates, labels, self._cleanup_result)
        self.state["cleanup"] = {
            "status": "pending" if pending else ("idle" if self.cleanup.public_config()["enabled"] else "disabled"),
            "suggestion": "", "latency_ms": 0, "error": "", "source_revision": self.revision,
        }

    def _cleanup_result(self, generation, revision, result):
        with self.lock:
            if (self.stop.is_set() or generation != self._cleanup_generation or revision != self.revision
                    or self.confirmed_revision == revision):
                return
            self.state["cleanup"] = {**result, "source_revision": revision}
            suggestion = result.get("suggestion", "")
            if result.get("status") == "ready" and suggestion and suggestion not in self.state["alternatives"]:
                self.state["alternatives"].append(suggestion)

    def receive_signs(self, events, now=None):
        now = clock_ms() if now is None else now
        with self.lock:
            if (self.browser_camera and self._camera_session_active is False
                    and not self.state["session_finalizing"]):
                return
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
                if self.state["session_active"]:
                    self.session_signs.append(event)
                    self.state["session_gestures"] = list(dict.fromkeys(e.label for e in self.session_signs))
                self.state["last_sign"] = asdict(event)
                self.state["metrics"]["sign_event_age_ms"] = now - event.end_ms
                if self.state["mode"] != "lips" and self.state["phase"] == "idle":
                    self._publish(None, [event], now)

    def action(self, name, data):
        with self.lock:
            self.state["error"] = ""
            if name == "camera_session":
                if self.rehearsal or not self.browser_camera:
                    raise ValueError("Camera sessions require live browser camera mode")
                active = data.get("active")
                if not isinstance(active, bool):
                    raise ValueError("active must be a boolean")
                if self._camera_session_active is active:
                    return
                self._camera_generation += 1
                self._camera_session_active = active
                self.state["auto_recognition"] = active
                self._auto_reset()
                if active:
                    # Only an explicit new camera session clears accumulated words.
                    self.session.reset()
                    self.session_signs.clear()
                    self.state["session_gestures"] = []
                    self.action("cancel", {})
                    self.state.update(session_active=True, session_finalizing=False,
                                      session_text="", session_chunks=[])
                else:
                    self.state["session_active"] = False
                    self.state["session_finalizing"] = self.state["phase"] in {"recording", "tail", "decoding"}
                    # Stop is finalization, not cancellation: decode the buffered tail.
                    if self.state["phase"] in {"recording", "tail"}:
                        self.release_ms = self.finish_at = clock_ms()
                        self.state.update(phase="tail", recording=False)
                        if self._window_motion:
                            self.hand_control.put(("finish", self.finish_at, self.record_generation))
                    elif self.state["phase"] == "idle":
                        self._publish_session(None)
                    self.state["camera"] = "waiting"
                    self.jpeg = None
                    self.tracking = {"hands": [], "mouth": None}
                    if not (self.state["session_finalizing"] and self._window_motion):
                        self.events.clear()
                    self.state["last_sign"] = None
                    for frames in (self.hand_frames, self.lip_frames):
                        try:
                            frames.get_nowait()
                        except queue.Empty:
                            pass
            elif name == "mode":
                if self.state["phase"] != "idle":
                    raise ValueError("Finish the current recording before changing mode")
                if data.get("mode") not in {"signs", "lips", "combined"}:
                    raise ValueError("Unknown input mode")
                self.state["mode"] = data["mode"]
                self._camera_generation += 1  # reset experimental motion windows
                self.events.clear()
                self.state["last_sign"] = None
                self._publish(None, [], clock_ms())
            elif name == "start":
                if self.rehearsal:
                    raise ValueError("Use synthetic input controls in rehearsal mode")
                mode = self.state["mode"]
                motion = self.state["sign_input"] == "motion" and not self._starting_auto_chunk
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
                self._record_generation += 1
                self.release_ms = self.finish_at = None
                self._window_motion = motion and mode != "lips"
                self._window_motion_done = not self._window_motion
                self.events.clear()
                if not self.state["session_active"]:
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
                self._record_generation += 1
                self.state["session_finalizing"] = False
                self._auto_reset()
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
            elif name == "correct":
                self._check_revision(data)
                if self.state["session_active"] or self.state["session_finalizing"] or self.state["phase"] != "idle":
                    raise ValueError("Finish the current input before correcting text")
                corrected = data.get("text", "")
                if not isinstance(corrected, str) or not corrected.strip() or len(corrected) > 2000:
                    raise ValueError("Corrected text must contain 1–2000 characters")
                corrected = corrected.strip()
                original = self.state["decision"]["text"]
                if corrected == original:
                    raise ValueError("The correction does not change the text")
                learn = data.get("learn", False)
                if not isinstance(learn, bool):
                    raise ValueError("learn must be a boolean")
                if learn:
                    if self.profile is None:
                        raise ValueError("Create and select a profile before learning from corrections")
                    from .corrections import CorrectionStore
                    CorrectionStore(self.profile).record(
                        original=original, corrected=corrected,
                        candidates=list(self.state["alternatives"]), learn=True)
                if corrected not in self.state["alternatives"]:
                    self.state["alternatives"].append(corrected)
                self.state["decision"].update(text=corrected, status="accepted", reason="corrected",
                                              matched_signs=[])
                self.revision += 1
                self.confirmed_revision = self.revision
                self.state["last_correction"] = "Correction saved for future training" if learn else "Correction applied locally"
                self._cleanup_generation += 1
                self.state["cleanup"].update(status="idle" if self.cleanup.public_config()["enabled"] else "disabled",
                                             suggestion="", error="")
            elif name == "confirm":
                self._check_revision(data)
                text = data.get("text", "")
                if not text or text not in self.state["alternatives"]:
                    raise ValueError("Select one of the displayed alternatives")
                self.state["decision"]["text"] = text
                self.state["decision"]["status"] = "accepted"
                self.confirmed_revision = self.revision
                self._cleanup_generation += 1
                self.state["cleanup"].update(status="idle" if self.cleanup.public_config()["enabled"] else "disabled",
                                             suggestion="", error="")
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
                self._cleanup_generation += 1
                self.state["cleanup"].update(status="idle" if self.cleanup.public_config()["enabled"] else "disabled",
                                             suggestion="", error="")
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

    def _auto_reset(self):
        self._auto_open_since = self._auto_quiet_since = None
        self._auto_open_frames = 0
        if hasattr(self, "_lip_activity"):
            self._lip_activity.reset()

    @staticmethod
    def _mouth_open(tracking, aspect=4 / 3):
        """Geometric segmentation hint, not speech detection or confidence."""
        if not tracking:
            return None
        inner, outer = tracking.get("inner_lips", []), tracking.get("outer_lips", [])
        if len(inner) <= 15 or len(outer) <= 10:
            return None
        def distance(a, b):
            return (((a[0] - b[0]) * aspect) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        width = distance(outer[0], outer[10])
        return distance(inner[5], inner[15]) / width if width > 0 else None

    def _publish_session(self, result):
        """Append raw computed chunks; only Stop permits optional text cleanup."""
        if result is not None and result.candidates:
            self.session.append(result.candidates[0])
        text = self.session.text
        self.state["session_text"] = text
        self.state["session_chunks"] = list(self.session.chunks)
        combined = self.state["mode"] == "combined"
        # Historical session labels are optional text context, not freshly timed
        # local fusion evidence. Keep original timestamps and the global TTL.
        context = list(dict.fromkeys(e.label for e in self.session_signs)) if combined else []
        session_lips = (LipResult([text], result.decode_ms if result else 0,
                                  capture_fps=result.capture_fps, raw_frames=result.raw_frames,
                                  max_frame_gap_ms=result.max_frame_gap_ms) if text and result else
                        LipResult([text], 0) if text else None)
        self._publish(session_lips, list(self.events) if combined else [], clock_ms(),
                      self.record_start if self._window_motion else None, cleanup_labels=context)
        if not self.state["session_active"]:
            self.state["session_finalizing"] = False

    def _auto_frame(self, opening, timestamp_ms):
        """Lip activity endpoint state machine for continuous camera sessions."""
        with self.lock:
            if (not self.state["auto_recognition"] or self._camera_session_active is not True
                    or self.state["mode"] == "signs" or self.state["lips"] != "ready"
                    or self.state["camera"] != "ready"):
                return
            phase = self.state["phase"]
            if phase == "idle" and self._lip_activity.state != ActivityState.IDLE:
                self._lip_activity.reset()
            elif phase == "recording" and self._lip_activity.state != ActivityState.RECORDING:
                self._lip_activity.sync_recording(self.record_start or timestamp_ms)

            action = self._lip_activity.update(opening, timestamp_ms)
            if action == ActivityAction.START and self.state["phase"] == "idle":
                self._starting_auto_chunk = True
                try:
                    self.action("start", {})
                finally:
                    self._starting_auto_chunk = False
            elif action == ActivityAction.STOP and self.state["phase"] == "recording":
                self.action("stop", {})

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

    def _queue_camera_frame(self, frame_bgr, now):
        preserve = self.state["phase"] in {"recording", "tail"}
        if not preserve:
            for frames in (self.hand_frames, self.lip_frames):
                try:
                    while True:
                        frames.get_nowait()
                except queue.Empty:
                    pass
        dropped = [put_latest(frames, (frame_bgr, now))
                   for frames in (self.hand_frames, self.lip_frames)]
        if preserve and any(dropped):
            self.capture_dropped_frames += 1
        if self.capture_dropped_frames:
            self.state["metrics"]["capture_dropped_frames"] = self.capture_dropped_frames

    def receive_frame(self, frame_bgr, timestamp_ms=None):
        """Ingest one browser frame; return False for a repeated timestamp."""
        import cv2

        with self.lock:
            if self.browser_camera and self._camera_session_active is False:
                return False
            now = clock_ms() if timestamp_ms is None else timestamp_ms
            if self._last_frame_timestamp_ms is not None and now <= self._last_frame_timestamp_ms:
                return False
            self._last_frame_timestamp_ms = now
            self._note_frame_time(now)
            # Frames are read-only for consumers; preview processing uses a new array.
            self._queue_camera_frame(frame_bgr, now)
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
                with self.lock:
                    self._queue_camera_frame(frame, now)
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
                        camera_generation = self._camera_generation
                        camera_stopped = self.browser_camera and self._camera_session_active is False
                        shared_end = (self.finish_at
                                      if self.record_generation == active_generation else None)
                    if camera_stopped:
                        packet = None
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
                            if camera_generation == self._camera_generation:
                                if hasattr(recognizer, "last_tracking"):
                                    self.tracking["hands"] = recognizer.last_tracking
                                    self.tracking["hands_timestamp_ms"] = ts
                                self.receive_signs(events)
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
                    camera_generation = self._camera_generation
                    if (self.browser_camera and self._camera_session_active is False
                            and not self.state["session_finalizing"]):
                        adapter.recording = None
                        active_start = active_generation = None
                        continue
                if mode != "signs" and phase in {"recording", "tail"} and start != active_start:
                    try:
                        adapter.start(start)
                    except TypeError:
                        adapter.start()
                    active_start = start
                    active_generation = generation
                if phase == "idle":
                    if hasattr(adapter, "cancel") and adapter.recording is not None:
                        adapter.cancel()
                    else:
                        adapter.recording = None
                    active_start = active_generation = None
                if packet and packet[1] != last_processed_ts:
                    in_capture = (active_start is not None and packet[1] >= active_start
                                  and (end is None or packet[1] <= end))
                    if active_start is None or in_capture:
                        adapter.process(*packet)
                        last_processed_ts = packet[1]
                        with self.lock:
                            if camera_generation != self._camera_generation:
                                continue
                            if hasattr(adapter, "last_tracking"):
                                self.tracking["mouth"] = adapter.last_tracking
                                self.tracking["mouth_timestamp_ms"] = packet[1]
                            if self.state["auto_recognition"]:
                                opening = getattr(adapter, "last_mouth_open", None)
                                if opening is None:
                                    aspect = packet[0].shape[1] / packet[0].shape[0]
                                    opening = self._mouth_open(getattr(adapter, "last_tracking", None), aspect)
                                self._auto_frame(opening, packet[1])
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
                    record_generation = self._record_generation
                try:
                    result = adapter.finish()
                    with self.lock:
                        if (self.record_start != active_start
                                or self.record_generation != active_generation
                                or self.state["phase"] != "decoding"):
                            continue  # cancelled/changed while inference ran
                        if self.state["session_active"] or self.state["session_finalizing"]:
                            self._publish_session(result)
                        else:
                            events = list(self.events) if self.state["mode"] == "combined" else []
                            self._publish(result, events, self.finish_at, active_start)
                        self.state["metrics"].update(decode_ms=result.decode_ms,
                            release_to_text_ms=clock_ms() - released,
                            **getattr(adapter, "last_metrics", {}))
                except Exception as exc:
                    with self.lock:
                        if (self.record_start == active_start
                                and self.record_generation == active_generation):
                            self.state["lip_error"] = str(exc)
                            if self.state["session_active"] or self.state["session_finalizing"]:
                                self._publish_session(None)
                            else:
                                events = list(self.events) if self.state["mode"] == "combined" else []
                                self._publish(None, events, self.finish_at, active_start)
                finally:
                    with self.lock:
                        if (self.record_start == active_start
                                and self.record_generation == active_generation):
                            self._auto_reset()
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
        with self.lock:
            self._cleanup_generation += 1
        self.cleanup.close()
        self.speaker.close()
        for thread in self.threads:
            thread.join(timeout=1)

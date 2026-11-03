"""One camera, independent latest-frame workers, revision-bound confirmation."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import importlib
import os
from pathlib import Path
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


class ProfileSwitchConflict(RuntimeError):
    """Raised when profile switch is attempted while runtime is busy."""
    pass


class Runtime:
    def __init__(self, *, rehearsal=False, browser_camera=False, camera=0, enable_lips=True, vocabulary=None,
                 lip_model=None, lip_language_model=None, motion_model=None, profile_name=None,
                 profile=None, configuration_warning="", max_utterance_seconds=30.0,
                 lip_activity_config=None, profile_manager=None,
                 explicit_lip_model=None, explicit_lip_language_model=None,
                 explicit_motion_model=None, explicit_vocabulary_path=None, sign_model=None):
        if rehearsal and browser_camera:
            raise ValueError("Browser camera and rehearsal modes are mutually exclusive")
        self.rehearsal = rehearsal
        self.browser_camera = browser_camera
        self.camera_source = camera
        # Standard-language checkpoints are an explicit backend, never an alias
        # for the stock poses or a personal motion model.
        self.isolated_sign_model_path = Path(sign_model).expanduser() if sign_model is not None else None
        self.profile = profile
        if profile_manager is not None:
            self.profile_manager = profile_manager
        elif profile is not None:
            from .profiles import ProfileManager
            self.profile_manager = ProfileManager(root_dir=profile.root_dir)
        else:
            from .profiles import ProfileManager
            self.profile_manager = ProfileManager()
        self.enable_lips = enable_lips

        # Track explicit CLI / external overrides so profile switching respects precedence
        self._explicit_lip_model = explicit_lip_model if explicit_lip_model is not None else (lip_model if profile is None and lip_model is not None else None)
        self._explicit_lip_language_model = explicit_lip_language_model if explicit_lip_language_model is not None else (lip_language_model if profile is None and lip_language_model is not None else None)
        self._explicit_motion_model = explicit_motion_model if explicit_motion_model is not None else (motion_model if profile is None and motion_model is not None else None)
        self._explicit_vocabulary_path = explicit_vocabulary_path
        self._explicit_vocabulary = vocabulary if profile is None and vocabulary is not None else None

        self.lip_model_path = lip_model if lip_model is not None else self._explicit_lip_model
        self.lip_language_model_path = lip_language_model if lip_language_model is not None else self._explicit_lip_language_model
        self.motion_model_path = (motion_model if motion_model is not None else self._explicit_motion_model) or (os.environ.get("QUIETBRIDGE_MOTION_MODEL", "").strip() or None)
        self.max_utterance_ms = round(float(max_utterance_seconds) * 1000)
        if self.max_utterance_ms < 1000:
            raise ValueError("Maximum utterance length must be at least one second")
        self.lip_activity_config = lip_activity_config or LipActivityConfig()
        self._lip_activity = LipActivityDetector(self.lip_activity_config)
        self.vocabulary = {} if self.isolated_sign_model_path else (load_vocabulary() if vocabulary is None else vocabulary)
        if self.motion_model_path and not self.isolated_sign_model_path and not rehearsal and vocabulary is None:
            # Stock pose meanings must never leak into the custom motion session.
            self.vocabulary = {label: {"display_name": phrase, "phrase": phrase,
                               "matches": [phrase], "enabled": True, "verified": False}
                               for label, phrase in MOTION_PHRASES.items()}
        self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                   if rehearsal else self.vocabulary)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self._worker_stop = threading.Event()
        self._worker_generation = 0
        self._switching_profile = False
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
        self._lip_cancel_boundary = None
        self._lip_cancel_generation = 0
        self._auto_open_since = self._auto_quiet_since = None
        self._auto_open_frames = 0
        self._starting_auto_chunk = False
        self.cleanup = CleanupClient()
        self.session = SessionTranscript()
        self._session_windows = []
        self._session_conflict = False
        self.session_signs = []  # Stable events validated when received in this session.
        self.state = {
            "mode": "combined" if enable_lips else "signs", "phase": "idle",
            "profile_name": profile_name or (profile.name if profile else "Stock configuration"),
            "profile_id": profile.id if profile else "default",
            "profile_warning": configuration_warning,
            "camera": "rehearsal" if rehearsal else ("waiting" if browser_camera else "starting"),
            "lips": "rehearsal" if rehearsal else ("warming" if enable_lips else "disabled"),
            "signs": "rehearsal" if rehearsal else "loading",
            "sign_input": "rehearsal" if rehearsal else "loading",
            "lip_model": self._rehearsal_lip_model_info() if rehearsal else {"mode": "stock", "status": "loading"},
            "camera_error": "", "lip_error": "", "sign_error": "", "error": "",
            "decision": asdict(Decision("", "unavailable", "sign_only", "local")),
            "candidates": [], "candidate_scores": [], "alternatives": [], "last_sign": None,
            "metrics": {}, "last_spoken": "", "last_correction": "", "recording": False,
            "auto_recognition": False,
            "session_active": False, "session_finalizing": False,
            "session_text": "", "session_chunks": [], "session_gestures": [],
            "gesture_model": "isolated sign model (experimental)" if self.isolated_sign_model_path else "personal motion-v2 (experimental)" if self.motion_model_path else "stock",
            "sign_model": {"status": "loading"} if self.isolated_sign_model_path else None,
            "cleanup": {"status": "disabled", "suggestion": "", "latency_ms": 0,
                        "error": "", "source_revision": 0},
        }

    def _rehearsal_lip_model_info(self):
        if self.lip_model_path or self.lip_language_model_path:
            from .lips_adapter import _file_sha256
            info = {"mode": "personal" if self.lip_model_path else "stock", "status": "rehearsal"}
            if self.lip_model_path:
                info["vsr_path"] = str(self.lip_model_path)
                try:
                    info["vsr_sha256"] = _file_sha256(self.lip_model_path)
                except Exception:
                    pass
            if self.lip_language_model_path:
                info["language_model_path"] = str(self.lip_language_model_path)
                try:
                    info["language_model_sha256"] = _file_sha256(self.lip_language_model_path)
                except Exception:
                    pass
            return info
        return {"mode": "stock", "status": "rehearsal"}

    def start(self):
        if self.rehearsal:
            return
        self._start_workers()

    def _start_workers(self):
        if self.rehearsal:
            return
        self._worker_stop = threading.Event()
        worker_gen = self._worker_generation
        workers = [] if self.browser_camera else [("capture", self._capture)]
        workers += [("hands", self._hands), *([("lips", self._lips)] if self.enable_lips else [])]
        for name, target in workers:
            def run_worker(t=target, worker_stop=self._worker_stop):
                try:
                    t(worker_gen, worker_stop)
                except TypeError:
                    t()
            thread = threading.Thread(target=run_worker, name=f"demo-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def _stop_workers(self, timeout=2.0):
        if hasattr(self, "_worker_stop") and self._worker_stop is not None:
            self._worker_stop.set()
        threads = list(self.threads)
        self.threads.clear()
        for thread in threads:
            thread.join(timeout=timeout)

    def snapshot(self):
        with self.lock:
            prof_id = self.profile.id if self.profile else "default"
            return copy.deepcopy({**self.state, "profile_id": prof_id, "active_profile": prof_id,
                                  "revision": self.revision,
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
            previous = self.state["cleanup"].get("suggestion", "")
            if previous and self.confirmed_revision != self.revision and previous != self.state["decision"]["text"]:
                self.state["alternatives"] = [text for text in self.state["alternatives"] if text != previous]
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
        if self.state["session_active"] and self._session_windows:
            self._set_session_alternatives()
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
        candidates = [decision.text] if decision.text else []
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
            if getattr(self, "_switching_profile", False):
                raise ProfileSwitchConflict("A profile switch is in progress")
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
                    self._session_windows.clear()
                    self._session_conflict = False
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
                    # The active hand window still needs its queued tail. Idle
                    # preview frames can be discarded by the worker instead.
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
                motion = self.state["sign_input"] in {"motion", "isolated"} and not self._starting_auto_chunk
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
                boundary = clock_ms()
                self._lip_cancel_boundary = boundary
                self._lip_cancel_generation += 1
                # Producers hold the runtime lock, so removing the cancelled
                # window here cannot race a newly accepted camera frame.
                with self.lip_frames.mutex:
                    retained = [packet for packet in self.lip_frames.queue if packet[1] > boundary]
                    self.lip_frames.queue.clear()
                    self.lip_frames.queue.extend(retained)
                self._record_generation += 1
                self.state["session_finalizing"] = False
                self._auto_reset()
                self.window_generation += 1
                cancelled_generation = self.record_generation
                self.record_generation = None
                if self.state["sign_input"] in {"motion", "isolated"}:
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

    def _set_session_alternatives(self):
        # Fusion may propose complete text different from the raw accumulated
        # chunks. Its displayed decision must remain an allowed confirmation.
        alternatives = [text for text in (self.state["decision"]["text"], self.session.text) if text]
        for index, (_, candidates) in enumerate(self._session_windows):
            for candidate in candidates:
                candidate = candidate.strip()
                if candidate:
                    chunks = [candidate if i == index else selected
                              for i, (selected, _) in enumerate(self._session_windows)]
                    alternatives.append(" ".join(chunk for chunk in chunks if chunk))
        self.state["alternatives"] = list(dict.fromkeys(alternatives))

    def _publish_session(self, result, window_events=None):
        """Append a completed chunk; fuse explicit motion at its own boundary."""
        if window_events is not None:
            now = self.finish_at if self.finish_at is not None else clock_ms()
            decision = fuse(result, window_events, self._fusion_vocabulary, now, self.record_start)
            self._session_conflict |= decision.reason in {"conflict", "partial_agreement"}
            gesture_text = " ".join(self._fusion_vocabulary[event.label]["phrase"].strip() for event in
                                    usable_signs(window_events, self._fusion_vocabulary, now, self.record_start))
            candidates = [*(result.candidates if result else []), gesture_text]
            selected = decision.text.strip()
            if selected or any(c.strip() for c in candidates):
                self._session_windows.append((selected, candidates))
            self.session.append(selected)
            self.events.clear()  # Window evidence must not rerank a later session chunk.
        elif result is not None and result.candidates:
            selected = result.candidates[0].strip()
            self._session_windows.append((selected, result.candidates))
            self.session.append(selected)
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
        self._publish(session_lips, list(self.events) if combined and window_events is None else [], clock_ms(),
                      self.record_start if self._window_motion else None, cleanup_labels=context)
        # Standalone gesture choices are not complete accumulated-session text.
        if self._session_windows:
            self._set_session_alternatives()
        if self._session_conflict and text:
            self.state["decision"].update(status="confirmation_needed", reason="conflict", matched_signs=[])
        if not self.state["session_active"]:
            self.state["session_finalizing"] = False

    def _auto_frame(self, opening, timestamp_ms):
        """Lip activity endpoint state machine for continuous camera sessions."""
        with self.lock:
            if (not self.state["auto_recognition"] or self._camera_session_active is not True
                    or (self.state["mode"] != "lips" and self.state["sign_input"] in {"motion", "isolated"})
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
            if getattr(self, "_switching_profile", False):
                return False
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

    def _capture(self, worker_gen=0, worker_stop=None):
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
            while not self.stop.is_set() and (worker_stop is None or not worker_stop.is_set()):
                if worker_gen != self._worker_generation:
                    break
                ok, frame = cap.read()
                now = clock_ms()
                if not ok:
                    if time.perf_counter() - last_frame > 2:
                        raise RuntimeError("Camera stopped delivering frames")
                    if worker_stop is not None:
                        worker_stop.wait(0.01)
                    else:
                        self.stop.wait(0.01)
                    continue
                last_frame = time.perf_counter()
                # Frames are read-only for consumers; preview processing uses a new array.
                with self.lock:
                    if worker_gen == self._worker_generation:
                        self._queue_camera_frame(frame, now)
                        self.state["camera"] = "ready"
                        self._note_frame_time(now)
                if now - last_preview >= 100:
                    ok, encoded = cv2.imencode(".jpg", cv2.flip(frame, 1), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self.lock:
                            if worker_gen == self._worker_generation:
                                self.jpeg = encoded.tobytes()
                    last_preview = now
        except Exception as exc:
            with self.lock:
                if worker_gen == self._worker_generation:
                    self.state.update(camera="unavailable", camera_error=str(exc))
        finally:
            if cap is not None:
                cap.release()

    def _hands(self, worker_gen=0, worker_stop=None):
        recognizer = None
        active_generation = None
        try:
            if self.isolated_sign_model_path is not None:
                from .signs.isolated import IsolatedSignRecognizer
                recognizer = IsolatedSignRecognizer(self.isolated_sign_model_path)
            else:
                module = importlib.import_module("hackathon.signs.recognizer")
                recognizer = module.SignRecognizer(motion_model_path=self.motion_model_path) if self.motion_model_path is not None else module.SignRecognizer()
            isolated = self.isolated_sign_model_path is not None
            motion = isolated or getattr(recognizer, "_motion", None) is not None
            motion_auto = motion and getattr(recognizer, "motion_auto", False)
            with self.lock:
                if worker_gen == self._worker_generation:
                    if isolated:
                        self.vocabulary = {label: entry for label, entry in recognizer.vocabulary.items() if entry["enabled"]}
                        self._fusion_vocabulary = self.vocabulary
                        self.state["sign_model"] = recognizer.model_info
                    self.state.update(signs="ready",
                                      sign_input="isolated" if isolated else "motion_auto" if motion_auto else "motion" if motion else "stock")
            last = 0
            active_start = None
            active_generation = None
            pending_finish = None
            while not self.stop.is_set() and (worker_stop is None or not worker_stop.is_set()):
                if worker_gen != self._worker_generation:
                    break
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
                    if camera_stopped and not (motion and active_start is not None):
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
                            if camera_generation == self._camera_generation and worker_gen == self._worker_generation:
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
                # Finish only after all chronological frames through this window
                # have drained, including frames uploaded just before camera stop.
                with self.hand_frames.mutex:
                    if self.hand_frames.queue and self.hand_frames.queue[0][1] <= pending_finish:
                        continue
                events = recognizer.finish(pending_finish)
                with self.lock:
                    valid = (worker_gen == self._worker_generation
                             and self.record_generation == active_generation
                             and self.record_start == active_start
                             and self.state["phase"] in {"tail", "decoding"})
                    if valid:
                        if isolated:
                            self.state["sign_diagnostics"] = recognizer.last_diagnostics
                        self._window_motion_done = True
                        # Keep the runtime lock through event storage and sign-only
                        # publication so cancel cannot turn a late finish into idle input.
                        self.receive_signs(events)
                        if self.state["mode"] == "signs":
                            self.state.update(phase="idle", recording=False)
                            if self.state["session_active"] or self.state["session_finalizing"]:
                                self._publish_session(None, events)
                            else:
                                self._publish(None, events, pending_finish, active_start)
                            self.record_start = self.release_ms = self.finish_at = None
                            self.record_generation = None
                active_start = active_generation = pending_finish = None
        except Exception as exc:
            with self.lock:
                if worker_gen == self._worker_generation:
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

    def _lips(self, worker_gen=0, worker_stop=None):
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
                if worker_gen == self._worker_generation:
                    self.state["lips"] = "ready"
                    self.state["lip_model"] = {**getattr(adapter, "model_info", {"mode": "stock"}),
                                               "status": "ready"}
            active_start = None
            active_generation = None
            last_processed_ts = None
            applied_cancel_generation = None
            while not self.stop.is_set() and (worker_stop is None or not worker_stop.is_set()):
                if worker_gen != self._worker_generation:
                    break
                packet = None
                try:
                    packet = self.lip_frames.get(timeout=0.05)
                except queue.Empty:
                    pass
                with self.lock:
                    start, phase, end = self.record_start, self.state["phase"], self.finish_at
                    generation, mode = self.record_generation, self.state["mode"]
                    camera_generation = self._camera_generation
                    cancel_boundary = self._lip_cancel_boundary
                    if cancel_boundary is not None and self._lip_cancel_generation != applied_cancel_generation:
                        if hasattr(adapter, "cancel"):
                            adapter.cancel(cancel_boundary)
                        else:
                            adapter.recording = None
                        applied_cancel_generation = self._lip_cancel_generation
                        active_start = active_generation = None
                    if packet is not None and cancel_boundary is not None and packet[1] <= cancel_boundary:
                        packet = None
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
                            if (camera_generation != self._camera_generation or worker_gen != self._worker_generation
                                    or (self._lip_cancel_boundary is not None
                                        and packet[1] <= self._lip_cancel_boundary)):
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
                    # Producers enqueue chronologically under this same lock. Drain
                    # every queued in-window frame before decoding, skipping later
                    # frames rather than letting them hold up finalization.
                    with self.lip_frames.mutex:
                        pending_capture = any(active_start <= ts <= self.finish_at
                                              for _, ts in self.lip_frames.queue)
                    if pending_capture:
                        continue
                    self.state["phase"] = "decoding"
                    released = self.release_ms
                    record_generation = self._record_generation
                try:
                    result = adapter.finish()
                    with self.lock:
                        if (worker_gen != self._worker_generation
                                or self.record_start != active_start
                                or self.record_generation != active_generation
                                or self.state["phase"] != "decoding"):
                            continue  # cancelled/changed while inference ran
                        if self.state["session_active"] or self.state["session_finalizing"]:
                            self._publish_session(result, list(self.events) if self._window_motion else None)
                        else:
                            events = list(self.events) if self.state["mode"] == "combined" else []
                            self._publish(result, events, self.finish_at, active_start)
                        self.state["metrics"].update(decode_ms=result.decode_ms,
                            release_to_text_ms=clock_ms() - released,
                            **getattr(adapter, "last_metrics", {}))
                except Exception as exc:
                    with self.lock:
                        if (worker_gen == self._worker_generation
                                and self.record_start == active_start
                                and self.record_generation == active_generation):
                            self.state["lip_error"] = str(exc)
                            if self.state["session_active"] or self.state["session_finalizing"]:
                                self._publish_session(None, list(self.events) if self._window_motion else None)
                            else:
                                events = list(self.events) if self.state["mode"] == "combined" else []
                                self._publish(None, events, self.finish_at, active_start)
                finally:
                    with self.lock:
                        if (worker_gen == self._worker_generation
                                and self.record_start == active_start
                                and self.record_generation == active_generation):
                            self._auto_reset()
                            self.state.update(phase="idle", recording=False)
                            self.record_start = None
                            self.record_generation = None
                    active_start = active_generation = None
        except Exception as exc:
            with self.lock:
                if worker_gen == self._worker_generation:
                    self.tracking["mouth"] = None
                    self.tracking.pop("mouth_timestamp_ms", None)
                    self.state.update(lips="unavailable", lip_error=str(exc), phase="idle", recording=False)
                    if self.state["session_finalizing"]:
                        self._publish_session(None)
                    if active_generation is not None and self.record_generation == active_generation:
                        if self._window_motion:
                            self.hand_control.put(("cancel", None, active_generation))
                        self.record_start = self.release_ms = self.finish_at = None
                        self.record_generation = None
                        self._window_motion = False
        finally:
            if adapter is not None:
                adapter.close()

    def switch_profile(self, profile_id: str):
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("Profile id must be a non-empty string")
        profile_id = profile_id.strip()

        with self.lock:
            if getattr(self, "_switching_profile", False):
                raise ProfileSwitchConflict("A profile switch is already in progress")
            if self.browser_camera and self._camera_session_active is True:
                raise ProfileSwitchConflict("Cannot switch profile while browser camera is active. Stop camera first.")
            if self.state["phase"] != "idle" or self.state.get("recording"):
                raise ProfileSwitchConflict("Cannot switch profile while recording or decoding is in progress")
            if (self.state.get("session_active")
                    or self.state.get("session_finalizing")
                    or self.state.get("auto_recognition")):
                raise ProfileSwitchConflict("Cannot switch profile while continuous session is active or finalizing")

            if self.profile_manager is None:
                from .profiles import ProfileManager
                self.profile_manager = ProfileManager(root_dir=self.profile.root_dir if self.profile else None)
            new_profile = self.profile_manager.get_profile(profile_id)

            from .profiles import (profile_configuration, profile_language_model,
                                   profile_lip_model, profile_motion_model)

            if self._explicit_vocabulary_path is not None:
                new_vocab, new_motion, new_warning = profile_configuration(new_profile, self._explicit_vocabulary_path)
            elif self._explicit_vocabulary is not None:
                new_vocab = copy.deepcopy(self._explicit_vocabulary)
                new_motion = profile_motion_model(new_profile) if profile_motion_model(new_profile).is_file() else None
                new_warning = ""
            else:
                new_vocab, new_motion, new_warning = profile_configuration(new_profile)

            if new_vocab is None:
                new_vocab = load_vocabulary()

            # Motion model resolution
            if self._explicit_motion_model is not None:
                target_motion = self._explicit_motion_model
            else:
                target_motion = new_motion

            if target_motion is not None and not Path(target_motion).is_file():
                raise FileNotFoundError(f"Motion model not found: {target_motion}")

            # Lip models resolution
            if self._explicit_lip_model is not None:
                target_lip = self._explicit_lip_model
            else:
                p_lip = profile_lip_model(new_profile)
                target_lip = p_lip if p_lip.is_file() else None

            if target_lip is not None and not Path(target_lip).is_file():
                raise FileNotFoundError(f"Lip model not found: {target_lip}")

            if self._explicit_lip_language_model is not None:
                target_lm = self._explicit_lip_language_model
            else:
                p_lm = profile_language_model(new_profile)
                target_lm = p_lm if p_lm.is_file() else None

            if target_lm is not None and not Path(target_lm).is_file():
                raise FileNotFoundError(f"Language model not found: {target_lm}")

            self._switching_profile = True

        try:
            self._stop_workers()

            with self.lock:
                self._worker_generation += 1

                for q in (self.hand_frames, self.lip_frames, self.hand_control):
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                self.frame_times.clear()
                self.capture_dropped_frames = 0
                self.events.clear()
                self.seen.clear()
                self.session.reset()
                self._session_windows.clear()
                self._session_conflict = False
                self.session_signs.clear()
                self.tracking = {"hands": [], "mouth": None}
                self.jpeg = None
                self._last_frame_timestamp_ms = None
                self._camera_generation += 1

                self.profile = new_profile
                self.motion_model_path = target_motion
                self.lip_model_path = target_lip
                self.lip_language_model_path = target_lm
                self.vocabulary = {} if self.isolated_sign_model_path else new_vocab
                self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                           if self.rehearsal else self.vocabulary)

                self.revision += 1
                self.confirmed_revision = None
                self.record_start = self.release_ms = self.finish_at = None
                self.record_generation = None
                self._window_motion = False
                self._window_motion_done = False
                self._auto_reset()
                self._cleanup_generation += 1

                self.state.update({
                    "profile_name": new_profile.name,
                    "profile_id": new_profile.id,
                    "profile_warning": new_warning,
                    "camera": "rehearsal" if self.rehearsal else ("waiting" if self.browser_camera else "starting"),
                    "lips": "rehearsal" if self.rehearsal else ("warming" if self.enable_lips else "disabled"),
                    "signs": "rehearsal" if self.rehearsal else "loading",
                    "sign_input": "rehearsal" if self.rehearsal else "loading",
                    "lip_model": self._rehearsal_lip_model_info() if self.rehearsal else {"mode": "stock", "status": "loading"},
                    "gesture_model": "isolated sign model (experimental)" if self.isolated_sign_model_path else "personal motion-v2 (experimental)" if self.motion_model_path else "stock",
                    "sign_model": {"status": "loading"} if self.isolated_sign_model_path else None,
                    "sign_diagnostics": None,
                    "camera_error": "", "lip_error": "", "sign_error": "", "error": "",
                    "decision": asdict(Decision("", "unavailable", "sign_only", "local")),
                    "candidates": [], "candidate_scores": [], "alternatives": [], "last_sign": None,
                    "metrics": {}, "last_spoken": "", "last_correction": "", "recording": False,
                    "auto_recognition": False,
                    "session_active": False, "session_finalizing": False,
                    "session_text": "", "session_chunks": [], "session_gestures": [],
                    "cleanup": {"status": "idle" if self.cleanup.public_config()["enabled"] else "disabled",
                                "suggestion": "", "latency_ms": 0, "error": "", "source_revision": self.revision},
                })

                self.profile_manager.set_active_profile(new_profile.id)

                if not self.rehearsal and not self.stop.is_set():
                    self._start_workers()

                return new_profile
        finally:
            with self.lock:
                self._switching_profile = False

    def close(self):
        self.stop.set()
        self._stop_workers()
        with self.lock:
            self._cleanup_generation += 1
        self.cleanup.close()
        self.speaker.close()

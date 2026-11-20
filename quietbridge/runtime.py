"""One camera, independent latest-frame workers, revision-bound confirmation."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import stat
import threading
import time

from . import phrase_match
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


def _motion_vocabulary_error(path):
    return ValueError(
        f"Custom motion model {path} has no vocabulary. Select the profile that owns it "
        "(--profile) or pass --vocabulary with its label-to-phrase mapping; stock gesture "
        "meanings and fixed phrases are never applied to a custom motion model.")


# Cleared on every published result so values never describe an older one.
_PER_RESULT_METRICS = (
    "release_to_text_ms", "decode_ms", "lip_capture_fps", "lip_raw_frames", "lip_max_frame_gap_ms",
    "lip_start_ms", "lip_end_ms", "lip_candidate_count", "roi_ms", "encode_ms", "search_ms",
    "lip_model_frames", "lip_repeated_frames", "lip_unused_frames", "lip_dropped_frames",
    "accepted_source", "accepted_candidate_index", "session_chunk_evidence")

MAX_QUICK_PHRASES = 50
MAX_PHRASE_CHARS = 2000
# State is polled several times per second; promoted model files and the
# profile's phrase list are stat-checked at most this often.
MODEL_CHECK_INTERVAL_S = 1.0
_MODEL_SLOT_NAMES = {"motion": "gesture model", "lip": "lip-reading model",
                     "lm": "lip-reading language model"}


def quick_phrases(prompts):
    """The profile's own practice phrases as typing shortcuts: trimmed, unique, bounded."""
    phrases = []
    for prompt in prompts if isinstance(prompts, list) else []:
        if not isinstance(prompt, str):
            continue
        text = prompt.strip()
        if text and len(text) <= MAX_PHRASE_CHARS and text not in phrases:
            phrases.append(text)
            if len(phrases) == MAX_QUICK_PHRASES:
                break
    return phrases


MAX_LEARNED_PHRASES = 50


def _learned_phrases(corrections_path):
    """Corrected texts the person explicitly chose to learn from, newest last.

    Uses the corrections module's own audit, which already excludes records
    whose sources lie in a validation or test split. Damaged files give [].
    """
    if corrections_path is None or _stat_signature(corrections_path) is None:
        return []
    try:
        from .corrections import learning_texts
        texts = learning_texts(Path(corrections_path).parent)
    except Exception:
        return []
    return quick_phrases(texts[-MAX_LEARNED_PHRASES:])


def _stat_signature(path):
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_size, info.st_mtime_ns


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class _ProfileFileWatch:
    """Notice promoted profile models and phrase edits made by another process.

    Never reloads anything: it only reports. File I/O happens outside the
    runtime lock, at most once per interval, by one caller at a time; content
    is hashed only when a file's size or modification time changed (plus one
    lazy baseline hash per loaded file).
    """

    def __init__(self, interval=MODEL_CHECK_INTERVAL_S, clock=time.monotonic):
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()        # guards the fields below; never held for I/O
        self._check_lock = threading.Lock()  # one checker at a time
        self._generation = 0
        self._slots = {}
        self._config_path = None
        self._config_sig = None
        self._phrases = []
        self._corrections_path = None
        self._corrections_sig = None
        self._learned = []
        self._result = (False, "")
        self._next_check = 0.0

    def reset(self, slots, config_path, phrases, corrections_path=None):
        """``slots`` maps a slot name to ``(path, loaded_signature)``."""
        corrections_sig = _stat_signature(corrections_path) if corrections_path is not None else None
        learned = _learned_phrases(corrections_path)
        with self._lock:
            self._generation += 1
            self._slots = {name: {"path": path, "loaded": loaded, "sha256": None}
                           for name, (path, loaded) in slots.items()}
            self._config_path = config_path
            self._config_sig = _stat_signature(config_path) if config_path is not None else None
            self._phrases = list(phrases)
            self._corrections_path = corrections_path
            self._corrections_sig = corrections_sig
            self._learned = learned
            self._result = (False, "")
            self._next_check = 0.0

    def suggestion_phrases(self):
        """Saved phrases, then learned corrections; cached, never does I/O."""
        with self._lock:
            return list(dict.fromkeys([*self._phrases, *self._learned]))

    def note_learned(self, text):
        """A correction this process just stored for learning."""
        with self._lock:
            if text not in self._learned:
                self._learned = [*self._learned, text][-MAX_LEARNED_PHRASES:]

    def check(self):
        """Return ``(reload_needed, detail, phrases)``; cached between checks."""
        with self._lock:
            cached = (*self._result, list(self._phrases))
            if self._clock() < self._next_check:
                return cached
        if not self._check_lock.acquire(blocking=False):
            return cached
        try:
            with self._lock:
                generation = self._generation
                slots = {name: dict(slot) for name, slot in self._slots.items()}
                config_path, config_sig = self._config_path, self._config_sig
                corrections_path, corrections_sig = self._corrections_path, self._corrections_sig
            learned = None
            current_corrections = (_stat_signature(corrections_path)
                                   if corrections_path is not None else None)
            if current_corrections != corrections_sig:
                learned = _learned_phrases(corrections_path)
            changes = []
            for name, slot in slots.items():
                current = _stat_signature(slot["path"])
                if current == slot["loaded"]:
                    if current is not None and slot["sha256"] is None:
                        digest = _sha256(slot["path"])
                        if _stat_signature(slot["path"]) == current:
                            slot["sha256"] = digest
                    continue
                if current is None:
                    changes.append((name, "removed"))
                elif slot["loaded"] is None:
                    changes.append((name, "saved"))
                elif slot["sha256"] is not None and _sha256(slot["path"]) == slot["sha256"]:
                    slot["loaded"] = current  # touched, same content: nothing to reload
                else:
                    changes.append((name, "saved"))
            phrases = None
            current_config = _stat_signature(config_path) if config_path is not None else None
            if current_config is not None and current_config != config_sig:
                try:
                    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
                    phrases = quick_phrases(data.get("lip_prompts")) if isinstance(data, dict) else None
                except (OSError, ValueError):
                    phrases = None  # mid-write or damaged: keep the last good list
                if phrases is None:
                    current_config = config_sig
            detail = " ".join(
                f"A newer {_MODEL_SLOT_NAMES[name]} was saved for this profile." if kind == "saved"
                else f"The {_MODEL_SLOT_NAMES[name]} this profile was using was removed."
                for name, kind in changes)
            with self._lock:
                if generation == self._generation:
                    self._slots = slots
                    self._result = (bool(changes), detail)
                    if phrases is not None:
                        self._phrases = phrases
                        self._config_sig = current_config
                    if learned is not None:
                        self._learned = learned
                        self._corrections_sig = current_corrections
                    self._next_check = self._clock() + self._interval
                return (*self._result, list(self._phrases))
        finally:
            self._check_lock.release()


class ProfileSwitchConflict(RuntimeError):
    """Raised when profile switch is attempted while runtime is busy."""
    pass


class Runtime:
    def __init__(self, *, rehearsal=False, browser_camera=False, camera=0, enable_lips=True, vocabulary=None,
                 lip_model=None, lip_language_model=None, motion_model=None, profile_name=None,
                 profile=None, configuration_warning="", max_utterance_seconds=30.0,
                 lip_activity_config=None, profile_manager=None,
                 explicit_lip_model=None, explicit_lip_language_model=None,
                 explicit_motion_model=None, explicit_vocabulary_path=None, sign_model=None,
                 phrase_suggestions=None):
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
        # Personal phrase suggestions beside the lip n-best (see phrase_match).
        # None follows the QUIETBRIDGE_PHRASE_SUGGESTIONS environment switch.
        self.phrase_suggestions = (phrase_match.enabled_by_environment() if phrase_suggestions is None
                                   else bool(phrase_suggestions))

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
            # Stock pose meanings must never leak into the custom motion session,
            # and a model without its own label-to-phrase mapping fails visibly.
            raise _motion_vocabulary_error(self.motion_model_path)
        self._fusion_vocabulary = ({k: {**v, "verified": True} for k, v in self.vocabulary.items()}
                                   if rehearsal else self.vocabulary)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self._worker_stop = threading.Event()
        self._worker_generation = 0
        self._switching_profile = False
        # Incremented whenever the displayed message context is replaced (profile switch).
        self._message_epoch = 0
        # True while a speak action has been validated and the voice is starting.
        self._speech_dispatching = False
        # Idle preview keeps only the latest frame. Active capture uses a bounded
        # chronological buffer (90 decoded frames) and reports every overflow.
        self.hand_frames = queue.Queue(maxsize=90)
        self.lip_frames = queue.Queue(maxsize=90)
        self.capture_dropped_frames = 0
        self._window_dropped_frames = 0  # lip-queue overflow drops since the current window started
        self.hand_control = queue.Queue()
        self.frame_times = deque()
        self.events = deque(maxlen=100)
        self.seen = deque(maxlen=200)
        # Preview: keep only the latest frame reference; encode lazily on request.
        self._preview_frame = None
        self._preview_seq = 0
        self._preview_cache = None  # (sequence, mirrored JPEG bytes)
        self._preview_encode_lock = threading.Lock()
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
        self._session_evidence = []  # text-free per-chunk decoder/capture evidence
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
            "candidates": [], "candidate_scores": [], "alternatives": [], "phrase_matches": [],
            "last_sign": None,
            "metrics": {}, "last_spoken": "", "last_correction": "", "recording": False,
            "auto_recognition": False,
            "session_active": False, "session_finalizing": False,
            "session_text": "", "session_chunks": [], "session_gestures": [],
            "gesture_model": "isolated sign model (experimental)" if self.isolated_sign_model_path else "personal motion-v2 (experimental)" if self.motion_model_path else "stock",
            "sign_model": {"status": "loading"} if self.isolated_sign_model_path else None,
            "cleanup": {"status": "disabled", "suggestion": "", "latency_ms": 0,
                        "error": "", "source_revision": 0},
        }
        self._model_watch = _ProfileFileWatch()
        self._reset_model_watch()

    def _model_watch_slots(self):
        """Promoted-model slots this process could reload, with what it loaded.

        A slot pinned by an explicit path (command line, environment or an
        isolated sign model) or unused (lips disabled) is not watched.
        """
        if self.profile is None:
            return {}
        from .profiles import profile_language_model, profile_lip_model, profile_motion_model
        slots = {}
        for name, path, loaded, pinned in (
                ("motion", profile_motion_model(self.profile), self.motion_model_path,
                 self.isolated_sign_model_path is not None or self._explicit_motion_model is not None),
                ("lip", profile_lip_model(self.profile), self.lip_model_path,
                 not self.enable_lips or self._explicit_lip_model is not None),
                ("lm", profile_language_model(self.profile), self.lip_language_model_path,
                 not self.enable_lips or self._explicit_lip_language_model is not None)):
            if pinned or (loaded is not None and Path(loaded).expanduser() != path):
                continue
            slots[name] = (path, _stat_signature(path) if loaded is not None else None)
        return slots

    def _reset_model_watch(self):
        profile = self.profile
        self._model_watch.reset(self._model_watch_slots(),
                                profile.config_path if profile is not None else None,
                                quick_phrases(profile.lip_prompts) if profile is not None else [],
                                (Path(profile.profile_dir) / "corrections.jsonl"
                                 if getattr(profile, "profile_dir", None) is not None else None))

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
            # Every worker takes (generation, stop event). A TypeError raised
            # inside a worker must never re-run it with default arguments.
            def run_worker(t=target, generation=worker_gen, worker_stop=self._worker_stop):
                t(generation, worker_stop)
            thread = threading.Thread(target=run_worker, name=f"demo-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def _stop_workers(self, timeout=2.0):
        if self._worker_stop is not None:
            self._worker_stop.set()
        threads = list(self.threads)
        self.threads.clear()
        for thread in threads:
            thread.join(timeout=timeout)

    def snapshot(self):
        # Stat checks (rarely a hash) run before taking the runtime lock.
        reload_needed, reload_detail, phrases = self._model_watch.check()
        with self.lock:
            prof_id = self.profile.id if self.profile else "default"
            return copy.deepcopy({**self.state, "profile_id": prof_id, "active_profile": prof_id,
                                  "revision": self.revision,
                                  "confirmed": self.confirmed_revision == self.revision,
                                  "rehearsal": self.rehearsal, "browser_camera": self.browser_camera,
                                  "speech_available": self.speaker.available,
                                  "speaking": self._speech_busy(),
                                  "speech_backend": self._speaker_backend(),
                                  "speech_error": self._speaker_error(),
                                  "quick_phrases": phrases,
                                  "models_reload_needed": reload_needed,
                                  "models_reload_detail": reload_detail,
                                  "vocabulary": self.vocabulary})

    def _speaker_speaking(self):
        # Strict: test doubles (e.g. Mock) answer any attribute with a truthy object.
        try:
            return getattr(self.speaker, "speaking", False) is True
        except Exception:
            return False

    def _speech_busy(self):
        """A voice is playing, or a validated speak request is starting one."""
        return self._speech_dispatching or self._speaker_speaking()

    SPEECH_SAMPLE = "This is how your messages will sound."

    def speak_sample(self):
        """Speak one fixed sentence with the current voice settings, on request.

        For trying a voice or rate in Settings. It never touches the message,
        its revision, last_spoken or any suggestion, and it is refused while
        speech is busy so it cannot cut off or replay a message.
        """
        with self.lock:
            if self._switching_profile:
                raise ProfileSwitchConflict("A profile switch is in progress")
            if self._speech_busy():
                raise ValueError("Speech is already playing. Stop it before testing the voice")
            self._speech_dispatching = True
        try:
            self.speaker.speak(self.SPEECH_SAMPLE)
        finally:
            with self.lock:
                self._speech_dispatching = False

    def _speaker_error(self):
        try:
            error = getattr(self.speaker, "last_error", "")
        except Exception:
            return ""
        return error if isinstance(error, str) else ""

    def _speaker_backend(self):
        try:
            backend = getattr(self.speaker, "backend", None)
        except Exception:
            return None
        return backend if isinstance(backend, str) else None

    def stop_speaking(self):
        """Stop local speech. Never touches the message, revision or phase."""
        stop = getattr(self.speaker, "stop", None)
        if callable(stop):
            stop()

    @property
    def jpeg(self):
        """Mirrored JPEG of the latest camera frame, or None (encoded on demand)."""
        return self.preview_jpeg()

    @jpeg.setter
    def jpeg(self, value):
        with self.lock:
            self._preview_seq += 1
            self._preview_frame = None
            self._preview_cache = None if value is None else (self._preview_seq, bytes(value))

    def _set_preview_frame(self, frame_bgr):
        # Caller holds self.lock. Frames are read-only for consumers.
        self._preview_seq += 1
        self._preview_frame = frame_bgr

    def preview_jpeg(self):
        """Encode the latest frame at most once, outside the runtime lock."""
        with self.lock:
            frame, seq, cache = self._preview_frame, self._preview_seq, self._preview_cache
        if cache is not None and cache[0] == seq:
            return cache[1]
        if frame is None:
            return None
        with self._preview_encode_lock:
            with self.lock:
                cache = self._preview_cache
            if cache is not None and cache[0] == seq:
                return cache[1]
            import cv2
            ok, encoded = cv2.imencode(".jpg", cv2.flip(frame, 1), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return None
            data = encoded.tobytes()
            with self.lock:
                if self._preview_seq == seq:
                    self._preview_cache = (seq, data)
                elif self._preview_frame is None:
                    return None  # Cleared (camera stop / profile switch) while encoding.
            return data

    def tracking_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.tracking)

    def cleanup_config(self):
        return self.cleanup.public_config()

    def configure_cleanup(self, data):
        with self.lock:
            config = self.cleanup.configure(data)
            previous = self.state["cleanup"].get("suggestion", "")
            if (previous and self.confirmed_revision != self.revision
                    and previous != self.state["decision"]["text"]
                    and previous not in self.state["candidates"]
                    and previous not in self.state["phrase_matches"]):
                self.state["alternatives"] = [text for text in self.state["alternatives"] if text != previous]
            self._cleanup_generation += 1
            self.state["cleanup"] = {"status": "idle" if config["enabled"] else "disabled",
                                     "suggestion": "", "latency_ms": 0, "error": "",
                                     "source_revision": self.revision}
            return config

    def test_cleanup(self):
        # Deliberately outside the runtime lock: camera and state remain responsive.
        return self.cleanup.test()

    def _publish(self, lips, events, now, since=None, cleanup_labels=None, apply_phrases=True):
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
        for metric in _PER_RESULT_METRICS:
            self.state["metrics"].pop(metric, None)
        if lips is not None:
            self.state["metrics"].update(
                lip_capture_fps=lips.capture_fps,
                lip_raw_frames=lips.raw_frames,
                lip_max_frame_gap_ms=lips.max_frame_gap_ms,
                lip_start_ms=lips.start_ms,
                lip_end_ms=lips.end_ms,
                lip_candidate_count=len(self.state["candidates"]),
            )
        if decision.status == "accepted" and decision.text:
            # Only verified gesture compositions are accepted without a tap.
            self.state["metrics"]["accepted_source"] = "gesture"
        if self._session_evidence:
            self.state["metrics"]["session_chunk_evidence"] = copy.deepcopy(self._session_evidence)
        if apply_phrases:
            self._apply_phrase_suggestions(
                lips, session=bool(self.state["session_active"] and self._session_windows))
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

    def _apply_phrase_suggestions(self, lips=None, session=False):
        """Offer the person's own phrases behind every recognised alternative.

        Never changes the decision text or the order or content of existing
        alternatives; adds at most ``phrase_match.MAX_SUGGESTIONS`` texts at
        the end and lists them in ``phrase_matches``. Nothing is offered
        without a non-empty lip candidate. A suggestion whose negation, number
        or safety words differ from the displayed text marks the decision as a
        conflict, so the person must choose explicitly.

        Single window: phrases are compared with the lip n-best. Session: with
        the whole accumulated text and its chunk-swapped variants (a phrase
        split across endpointed chunks), and, from two chunks on, with the last
        chunk's n-best, offered as earlier chunks + phrase (like the existing
        chunk swaps, earlier recognised words are kept).
        """
        self.state["phrase_matches"] = []
        self.state["metrics"].pop("phrase_match", None)
        decision = self.state["decision"]
        if session:
            candidates = [text for text in self.state["alternatives"] if text]
        else:
            candidates = list(dict.fromkeys(c.strip() for c in (lips.candidates if lips else [])
                                            if isinstance(c, str) and c.strip()))
        if not candidates:
            return
        phrases = self._model_watch.suggestion_phrases()
        evidence = {"enabled": self.phrase_suggestions, "phrases": len(phrases),
                    "candidates": len(candidates), "already_listed": 0, "offered": []}
        self.state["metrics"]["phrase_match"] = evidence
        if not self.phrase_suggestions or not phrases:
            return
        top = decision["text"] or candidates[0]
        existing = list(self.state["alternatives"])
        matches = phrase_match.match_phrases(candidates, phrases, top=top, existing=existing)
        evidence["already_listed"] = sum(m.already_listed for m in matches)
        offers = [(m, m.phrase, "session" if session else "window") for m in matches if not m.already_listed]
        if session and len(self._session_windows) > 1:
            earlier = [selected for selected, _ in self._session_windows[:-1] if selected]
            last_selected, last_candidates = self._session_windows[-1]
            listed = {phrase_match.normalized(text) for text in existing}
            for m in phrase_match.match_phrases(last_candidates, phrases, top=last_selected):
                text = " ".join([*earlier, m.phrase])
                if m.already_listed or phrase_match.normalized(text) in listed:
                    continue
                differs = phrase_match.sensitive_words(text) != phrase_match.sensitive_words(top)
                offers.append((replace(m, sensitive_difference=differs), text, "last_chunk"))
        offered = []
        for m, text, scope in offers:
            if text in existing or text in offered:
                continue
            offered.append(text)
            evidence["offered"].append({**m.evidence(), "scope": scope})
            if len(offered) == phrase_match.MAX_SUGGESTIONS:
                break
        if not offered:
            return
        self.state["alternatives"].extend(offered)
        self.state["phrase_matches"] = offered
        # A gained suggestion is never auto-accepted (lip decisions never are).
        decision["status"] = "confirmation_needed"
        if any(item["sensitive_difference"] for item in evidence["offered"]):
            decision.update(reason="conflict", matched_signs=[])

    def _accepted_source(self, text):
        """Compact, text-free provenance of a confirmed alternative."""
        candidates = self.state["candidates"]
        if text in self.state["phrase_matches"]:
            return "phrase_suggestion"
        if self._session_windows and text == self.session.text:
            return "session_text"
        if candidates and text == candidates[0]:
            return "lip_top"
        if text in candidates:
            return "lip_candidate"
        if text and text == self.state["cleanup"].get("suggestion"):
            return "cleanup_suggestion"
        if self._session_windows:
            return "session_alternative"
        if text == self.state["decision"]["text"] and self.state["decision"]["reason"] in {
                "agreement", "partial_agreement", "sign_only"}:
            return "fusion" if candidates else "gesture"
        if any(text == entry.get("phrase") for entry in self._fusion_vocabulary.values()):
            return "gesture"
        return "other"

    def set_phrase_suggestions(self, enabled):
        """Switch personal phrase suggestions; takes effect from the next result."""
        with self.lock:
            self.phrase_suggestions = bool(enabled)

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
                    raise TypeError("Recognizer must return quietbridge.contracts.SignEvent instances")
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
        if name == "stop_speaking":
            # Allowed in any phase and idempotent. Stopping can wait for the
            # speech process to exit, so it deliberately runs outside the lock.
            self.stop_speaking()
            return
        if name == "speak":
            self._action_speak(data)
            return
        handler = getattr(self, f"_action_{name}") if name in self._LOCKED_ACTIONS else None
        with self.lock:
            if self._switching_profile:
                raise ProfileSwitchConflict("A profile switch is in progress")
            self.state["error"] = ""
            if handler is None:
                raise ValueError("Unknown action")
            handler(data)

    # Each _action_* below runs with the runtime lock held, after the switch
    # check, and either raises before changing state or completes.

    def _action_camera_session(self, data):
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
            self._session_evidence.clear()
            self._session_conflict = False
            self.session_signs.clear()
            self.state["session_gestures"] = []
            self._action_cancel({})
            self.state.update(session_active=True, session_finalizing=False,
                              session_text="", session_chunks=[])
            return
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

    def _action_mode(self, data):
        if self.state["phase"] != "idle":
            raise ValueError("Finish the current recording before changing mode")
        if data.get("mode") not in {"signs", "lips", "combined"}:
            raise ValueError("Unknown input mode")
        self.state["mode"] = data["mode"]
        self._camera_generation += 1  # reset experimental motion windows
        self.events.clear()
        self.state["last_sign"] = None
        self._publish(None, [], clock_ms())

    def _action_start(self, data):
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
        self._window_dropped_frames = 0
        self.release_ms = self.finish_at = None
        self._window_motion = motion and mode != "lips"
        self._window_motion_done = not self._window_motion
        self.events.clear()
        if not self.state["session_active"]:
            self._publish(None, [], self.record_start)
        self.state.update(phase="recording", recording=True, lip_error="", sign_error="")
        if self._window_motion:
            self.hand_control.put(("start", self.record_start, self.record_generation))

    def _action_stop(self, data):
        if self.state["phase"] != "recording":
            return  # A late key/pointer release after timeout or cancel is harmless.
        self.release_ms = clock_ms()
        self.finish_at = self.release_ms + 400
        self.state.update(phase="tail", recording=False)
        if self._window_motion:
            self.hand_control.put(("finish", self.finish_at, self.record_generation))

    def _action_cancel(self, data):
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

    def _action_new_message(self, data):
        self._check_revision(data)
        if (self.state["session_active"] or self.state["session_finalizing"]
                or self.state["phase"] != "idle"
                or self._speech_busy()):
            raise ValueError("Finish the current input or speech before starting a new message")
        # Cancel publishes a decision, so discard the accumulated session first.
        self.session.reset()
        self._session_windows.clear()
        self._session_evidence.clear()
        self._session_conflict = False
        self.session_signs.clear()
        self.events.clear()
        self.seen.clear()
        self.state.update(session_text="", session_chunks=[], session_gestures=[],
                          last_sign=None, last_correction="")
        clear_error = getattr(self.speaker, "clear_error", None)
        if callable(clear_error):
            clear_error()  # a failure report belongs to the previous message
        self._action_cancel({})

    def _action_correct(self, data):
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
            self._model_watch.note_learned(corrected)
        # The edit is a new revision: phrase rows belonged to the old result.
        if self.state["phrase_matches"]:
            self.state["alternatives"] = [text for text in self.state["alternatives"]
                                          if text not in self.state["phrase_matches"]]
            self.state["phrase_matches"] = []
        if corrected not in self.state["alternatives"]:
            self.state["alternatives"].append(corrected)
        self.state["decision"].update(text=corrected, status="accepted", reason="corrected",
                                      matched_signs=[])
        self.revision += 1
        self.confirmed_revision = self.revision
        self.state["last_correction"] = "Correction saved for future training" if learn else "Correction applied locally"
        self.state["metrics"].update(accepted_source="manual_edit", accepted_candidate_index=None)
        self._reset_cleanup_suggestion()

    def _action_confirm(self, data):
        self._check_revision(data)
        text = data.get("text", "")
        if not text or text not in self.state["alternatives"]:
            raise ValueError("Select one of the displayed alternatives")
        candidates = self.state["candidates"]
        self.state["metrics"].update(
            accepted_source=self._accepted_source(text),
            accepted_candidate_index=candidates.index(text) if text in candidates else None)
        self.state["decision"]["text"] = text
        self.state["decision"]["status"] = "accepted"
        self.confirmed_revision = self.revision
        self._reset_cleanup_suggestion()

    def _action_rehearse(self, data):
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

    # Actions validated and applied entirely under the runtime lock.
    _LOCKED_ACTIONS = frozenset({"camera_session", "mode", "start", "stop", "cancel",
                                 "new_message", "correct", "confirm", "rehearse"})

    def _reset_cleanup_suggestion(self):
        self._cleanup_generation += 1
        self.state["cleanup"].update(status="idle" if self.cleanup.public_config()["enabled"] else "disabled",
                                     suggestion="", error="")

    def _action_speak(self, data):
        """Speak the confirmed text of the requested revision; never while speech is busy.

        Validation is revision-bound and runs under the runtime lock. Starting
        the voice (process spawn plus the stdin write, which can wait for a
        slow-starting voice) runs outside it, so polling and the recognition
        workers are never stalled by speech. A second request while a voice is
        starting or playing is rejected rather than restarting it: a repeated
        click or request must not cut off and replay the message.
        """
        with self.lock:
            if self._switching_profile:
                raise ProfileSwitchConflict("A profile switch is in progress")
            self.state["error"] = ""
            self._check_revision(data)
            decision = self.state["decision"]
            if decision["status"] != "accepted" or not decision["text"]:
                raise ValueError("Confirm a displayed alternative before speaking")
            if data.get("text") != decision["text"]:
                raise ValueError("The message changed. Check it again before speaking")
            if self._speech_busy():
                raise ValueError("Speech is already playing. Stop it before speaking again")
            text, revision, epoch = decision["text"], self.revision, self._message_epoch
            self._speech_dispatching = True
        try:
            started = time.perf_counter()
            self.speaker.speak(text)
            dispatch_ms = round((time.perf_counter() - started) * 1000, 2)
        finally:
            with self.lock:
                self._speech_dispatching = False
        with self.lock:
            if epoch != self._message_epoch:
                return  # profile switched meanwhile; its fresh state stays untouched
            self.state["last_spoken"] = text
            self.state["metrics"]["tts_dispatch_ms"] = dispatch_ms
            if self.revision == revision:
                self._reset_cleanup_suggestion()

    def _auto_reset(self):
        self._auto_open_since = self._auto_quiet_since = None
        self._auto_open_frames = 0
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

    def _chunk_evidence(self, result, reason):
        """Text-free decoder and capture evidence for one session chunk."""
        return {"candidate_count": len(result.candidates) if result else 0,
                "candidate_scores": list(result.candidate_scores) if result else [],
                "start_ms": result.start_ms if result else None,
                "end_ms": result.end_ms if result else None,
                "raw_frames": result.raw_frames if result else 0,
                "capture_fps": result.capture_fps if result else None,
                "max_frame_gap_ms": result.max_frame_gap_ms if result else None,
                "decode_ms": result.decode_ms if result else None,
                "dropped_frames": self._window_dropped_frames,
                "selected": reason}

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
                self._session_evidence.append(self._chunk_evidence(result, decision.reason))
            self.session.append(selected)
            self.events.clear()  # Window evidence must not rerank a later session chunk.
        elif result is not None and result.candidates:
            selected = result.candidates[0].strip()
            self._session_windows.append((selected, result.candidates))
            self._session_evidence.append(self._chunk_evidence(result, "lip_only"))
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
                      self.record_start if self._window_motion else None, cleanup_labels=context,
                      apply_phrases=False)
        # Standalone gesture choices are not complete accumulated-session text.
        if self._session_windows:
            self._set_session_alternatives()
        if self._session_conflict and text:
            self.state["decision"].update(status="confirmation_needed", reason="conflict", matched_signs=[])
        self._apply_phrase_suggestions(session_lips, session=bool(self._session_windows))
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
                self._lip_activity.sync_recording(self.record_start if self.record_start is not None else timestamp_ms)

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
        if preserve and dropped[1]:
            self._window_dropped_frames += 1
        if self.capture_dropped_frames:
            self.state["metrics"]["capture_dropped_frames"] = self.capture_dropped_frames

    def receive_frame(self, frame_bgr, timestamp_ms=None):
        """Ingest one browser frame; return False for a repeated timestamp."""
        with self.lock:
            if self._switching_profile:
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
            # No encode here: the preview is produced only if GET /api/frame asks.
            self._set_preview_frame(frame_bgr)
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
                        self._set_preview_frame(frame)
                        self.state["camera"] = "ready"
                        self._note_frame_time(now)
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
                module = importlib.import_module("quietbridge.signs.recognizer")
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

    @staticmethod
    def _capture_lower_bound(adapter, start):
        pre_roll = getattr(adapter, "pre_roll_ms", None)
        if start is None or isinstance(pre_roll, bool) or not isinstance(pre_roll, (int, float)):
            return start
        return start - max(0, pre_roll)

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
                    # A frame queued before the start but dequeued after it is
                    # still pre-roll: an adapter that declares a pre-roll takes
                    # it (and enforces its own window boundaries).
                    lower = self._capture_lower_bound(adapter, active_start)
                    in_capture = (active_start is not None and packet[1] >= lower
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
                    lower = self._capture_lower_bound(adapter, active_start)
                    with self.lip_frames.mutex:
                        pending_capture = any(lower <= ts <= self.finish_at
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
                            lip_dropped_frames=self._window_dropped_frames,
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
            if self._switching_profile:
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

            # Motion model resolution
            if self._explicit_motion_model is not None:
                target_motion = self._explicit_motion_model
            else:
                target_motion = new_motion

            if new_vocab is None:
                if (target_motion is not None and not self.rehearsal
                        and self.isolated_sign_model_path is None):
                    # Same rule as startup: no stock meanings for a custom model.
                    raise _motion_vocabulary_error(target_motion)
                new_vocab = load_vocabulary()

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
                self._session_evidence.clear()
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
                self._message_epoch += 1
                self.confirmed_revision = None
                self.record_start = self.release_ms = self.finish_at = None
                self.record_generation = None
                self._window_motion = False
                self._window_motion_done = False
                self._auto_reset()
                self._cleanup_generation += 1
                self._reset_model_watch()

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
                    "candidates": [], "candidate_scores": [], "alternatives": [], "phrase_matches": [],
                    "last_sign": None, "metrics": {}, "last_spoken": "", "last_correction": "", "recording": False,
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

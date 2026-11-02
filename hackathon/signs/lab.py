"""Standalone loopback lab server for recording raw DDE gesture sequences.

Independent from Sam's runtime: serves ``lab.html`` and a small JSON/JPEG API
on 127.0.0.1 only, records raw MediaPipe hand landmarks (both hands, no
normalization) into local ``.npz`` sequence files. Nothing is uploaded and no
images or video are stored.

CLI::

    .venv/bin/python -m hackathon.signs.lab --port 8766
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

#: Mirrors recognizer.SUPPORTED_LABELS without importing mediapipe at module load.
GESTURE_SUPPORT = ["THUMB_UP", "THUMB_DOWN", "OPEN_PALM", "POINTING_UP"]

MIN_SCORE = 0.75
NUM_HANDS = 2
MAX_FRAME_BYTES = 512 * 1024
MAX_JSON_BYTES = 4096
MAX_RECORD_FRAMES = 600
MAX_RECORD_SECONDS = 60.0
DEFAULT_PORT = 8766
DEFAULT_DATA_ROOT = Path.home() / ".local" / "share" / "quietbridge" / "sequences"
LAB_HTML = Path(__file__).with_name("lab.html")
STATE_NOTE = "Loopback lab recorder; sequences stay on this Mac, no uploads."

_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
_DOWNLOAD_HINT = "python -m hackathon.signs.assets --download"

#: Left hand goes to slot 0, right hand to slot 1.
_HAND_SLOTS = {"Left": 0, "Right": 1}


@dataclass
class HandResult:
    """One detected hand from a single inference."""

    handedness: str  # "Left" or "Right", exactly as MediaPipe reports it
    label: str  # raw canned-classifier top-1 category name, "" when none
    score: float
    landmarks: list  # 21x3 normalized coordinates, as floats


@dataclass
class FrameResult:
    """Engine output for one frame."""

    hands: list  # list[HandResult]
    events: list  # list[hackathon.contracts.SignEvent]


class Engine:
    """Real MediaPipe GestureRecognizer engine (VIDEO mode, built once).

    Imports mediapipe lazily so tests can run the server with a fake engine
    without the native dependency. Raw top-1 labels come straight from the
    classifier; confirmed events reuse the existing EventStabilizer gate and
    the recognizer's conservative ``_pick_label`` reducer, so an unsupported
    pose like DDE is never reported as a recognized sign.
    """

    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        from .assets import default_model_path
        from .recognizer import SignRecognizer
        from .stabilizer import EventStabilizer

        if model_path is None:
            model_path = default_model_path()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Gesture recognizer model not found at {model_path}. "
                f"Download it first with: {_DOWNLOAD_HINT}"
            )
        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            # No allowlist: raw top-1 stays visible; unsupported labels are
            # filtered later by _pick_label + the stabilizer, not hidden.
        )
        self._backend = vision.GestureRecognizer.create_from_options(options)
        self._stabilizer = EventStabilizer(min_score=MIN_SCORE)
        self._pick_label = SignRecognizer._pick_label

    def infer(self, frame_rgb, timestamp_ms: int) -> FrameResult:
        """Run one inference on an HxWx3 uint8 RGB frame."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._backend.recognize_for_video(image, timestamp_ms)
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            gestures = result.gestures[index] if index < len(result.gestures) else []
            top = gestures[0] if gestures else None
            handedness = ""
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name
            hands.append(
                HandResult(
                    handedness=handedness,
                    label=top.category_name if top and top.category_name else "",
                    score=float(top.score) if top else 0.0,
                    landmarks=[[float(p.x), float(p.y), float(p.z)] for p in landmarks],
                )
            )
        label, score = self._pick_label(result)
        events = list(self._stabilizer.update(label, score, timestamp_ms))
        return FrameResult(hands=hands, events=events)

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()


@dataclass
class _Recording:
    """In-memory sequence buffer; written to disk only by an explicit stop."""

    label: str
    split: str
    session: str
    start_monotonic: float
    data_root: Path | None = None
    limit_reached: bool = False
    timestamps_ms: list = field(default_factory=list)
    landmarks: list = field(default_factory=list)  # per frame (2, 21, 3) float32
    hand_present: list = field(default_factory=list)  # per frame (2,) bool
    handedness: list = field(default_factory=list)  # per frame (2,) strings


class _LabServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, data_root, max_frames, max_seconds, profile_manager=None):
        super().__init__(address, _Handler)
        self.engine = engine
        from .lip_lab import LipLab
        from .mouth_capture import MouthCapture
        from hackathon.profiles import ProfileManager, get_default_root

        self.lips = LipLab()
        self.mouth = MouthCapture()
        self.token = secrets.token_urlsafe(24)
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.lock = threading.Lock()
        self.recording: _Recording | None = None
        self.saved_count = 0
        self.last_saved = ""
        self._last_timestamp_ms: int | None = None

        if profile_manager is not None:
            self.profile_manager = profile_manager
        else:
            if data_root is not None and Path(data_root).name in ("seq", "sequences"):
                pm_root = Path(data_root).parent
            elif data_root is not None:
                pm_root = Path(data_root)
            else:
                pm_root = get_default_root()
            self.profile_manager = ProfileManager(root_dir=pm_root)

        if data_root is not None:
            self.data_root = Path(data_root).expanduser()
        else:
            active_p = self.profile_manager.get_active_profile()
            self.data_root = active_p.sequences_dir
            self.mouth.set_root(active_p.mouths_dir)

    def server_close(self):
        self.lips.close()
        self.mouth.close()
        super().server_close()

    def next_timestamp_ms(self) -> int:
        """Strictly increasing monotonic millisecond clock for VIDEO mode."""
        now = int(time.perf_counter() * 1000)
        if self._last_timestamp_ms is not None and now <= self._last_timestamp_ms:
            now = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now
        return now


def _valid_session(session) -> bool:
    return (
        isinstance(session, str)
        and bool(session)
        and len(session) <= 64
        and Path(session).name == session
        and session not in (".", "..")
    )


def _check_session_split(data_root: Path, split: str, session: str) -> None:
    for other in ("train", "validation", "test"):
        if other == split:
            continue
        other_dir = data_root / other
        if other_dir.is_dir() and any(other_dir.glob(f"{session}_*.npz")):
            raise ValueError(
                f"session {session!r} already has sequences in {other}; refusing split leakage"
            )


def _save_sequence(server: _LabServer, rec: _Recording) -> Path:
    """Write the buffered sequence as one .npz and return its path."""
    target_root = rec.data_root if rec.data_root is not None else server.data_root
    _check_session_split(target_root, rec.split, rec.session)
    trial_id = str(uuid.uuid4())
    out_dir = target_root / rec.split
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{rec.session}_{trial_id}.npz"
    # Exclusive creation: never overwrite an existing recording.
    with destination.open("xb") as archive:
        np.savez_compressed(
            archive,
            timestamps_ms=np.asarray(rec.timestamps_ms, dtype=np.int64),
            landmarks=np.stack(rec.landmarks).astype(np.float32),
            hand_present=np.stack(rec.hand_present).astype(bool),
            handedness=np.stack(rec.handedness),
            label=np.asarray(rec.label),
            split=np.asarray(rec.split),
            session_id=np.asarray(rec.session),
            trial_id=np.asarray(trial_id),
        )
    return destination
    # Exclusive creation: never overwrite an existing recording.
    with destination.open("xb") as archive:
        np.savez_compressed(
            archive,
            timestamps_ms=np.asarray(rec.timestamps_ms, dtype=np.int64),
            landmarks=np.stack(rec.landmarks).astype(np.float32),
            hand_present=np.stack(rec.hand_present).astype(bool),
            handedness=np.stack(rec.handedness),
            label=np.asarray(rec.label),
            split=np.asarray(rec.split),
            session_id=np.asarray(rec.session),
            trial_id=np.asarray(trial_id),
        )
    return destination


def _append_frame(server: _LabServer, rec: _Recording, result: FrameResult, timestamp_ms: int) -> None:
    """Append one frame to the recording buffer unless a limit was reached."""
    if len(rec.timestamps_ms) >= server.max_frames:
        rec.limit_reached = True
        return
    if time.monotonic() - rec.start_monotonic >= server.max_seconds:
        rec.limit_reached = True
        return
    landmarks = np.full((2, 21, 3), np.nan, dtype=np.float32)
    present = np.zeros((2,), dtype=bool)
    handedness = np.asarray(["", ""], dtype="<U5")
    for hand in result.hands:
        slot = _HAND_SLOTS.get(hand.handedness)
        if slot is None:
            continue
        landmarks[slot] = np.asarray(hand.landmarks, dtype=np.float32)
        present[slot] = True
        handedness[slot] = hand.handedness
    rec.timestamps_ms.append(timestamp_ms)
    rec.landmarks.append(landmarks)
    rec.hand_present.append(present)
    rec.handedness.append(handedness)


class _Handler(BaseHTTPRequestHandler):
    server: _LabServer
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------

    def log_message(self, fmt, *args):  # keep the console quiet during demos
        pass

    def _security_headers(self, content_type: str, length: int):
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # Loopback page with an inline script; no external CDN or plugins.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' blob:; "
            "connect-src 'self'; media-src 'self' blob:",
        )

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str):
        self._send_json(status, {"error": message})

    def _check_host(self) -> bool:
        """Only accept requests addressed to this loopback port."""
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        if host in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return True
        self._error(403, "Host header must be 127.0.0.1 or localhost on this port")
        return False

    def _check_post(self) -> bool:
        """Host + optional same-origin Origin + token for every POST."""
        if not self._check_host():
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            port = self.server.server_address[1]
            if origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self._drain_body()
                self._error(403, "Origin must match this loopback server")
                return False
        if self.headers.get("X-QuietBridge-Token") != self.server.token:
            self._drain_body()
            self._error(403, "missing or invalid X-QuietBridge-Token")
            return False
        return True

    def _drain_body(self, limit: int = MAX_FRAME_BYTES) -> None:
        """Consume an unread request body so keep-alive stays usable.

        Oversized or malformed lengths close the connection instead.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.close_connection = True
            return
        if length < 0 or length > limit:
            self.close_connection = True
            return
        while length:
            chunk = self.rfile.read(min(length, 65536))
            if not chunk:
                break
            length -= len(chunk)

    def _read_body(self, limit: int):
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._error(411, "Content-Length required")
            return None
        if length < 0 or length > limit:
            self.close_connection = True
            self._error(413, f"body too large (limit {limit} bytes)")
            return None
        return self.rfile.read(length)

    # -- GET -------------------------------------------------------------

    def do_GET(self):
        if not self._check_host():
            return
        if self.path == "/":
            try:
                body = LAB_HTML.read_bytes()
            except FileNotFoundError:
                self._error(404, "lab.html not found next to lab.py")
                return
            self.send_response(200)
            self._security_headers("text/html; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/profiles":
            with self.server.lock:
                payload = {
                    "active_profile": self.server.profile_manager.active_profile_id,
                    "profiles": self.server.profile_manager.list_profiles(include_stats=True),
                }
            self._send_json(200, payload)
            return
        if self.path == "/api/profile":
            with self.server.lock:
                active_p = self.server.profile_manager.get_active_profile()
                payload = active_p.to_dict(include_stats=True)
            self._send_json(200, payload)
            return
        if self.path == "/api/state":
            with self.server.lock:
                rec = self.server.recording
                active_p = self.server.profile_manager.get_active_profile()
                prof_labels = [g["label"] for g in active_p.gestures]
                combined_support = list(dict.fromkeys(GESTURE_SUPPORT + prof_labels))
                payload = {
                    "token": self.server.token,
                    "recording": rec is not None,
                    "saved_count": self.server.saved_count,
                    "last_saved": self.server.last_saved,
                    "model_ready": self.server.engine is not None,
                    "gesture_support": combined_support,
                    "note": STATE_NOTE,
                    "recorded_frames": len(rec.timestamps_ms) if rec else 0,
                    "limit_reached": bool(rec and rec.limit_reached),
                    "lips": self.server.lips.snapshot(),
                    "mouth": self.server.mouth.snapshot(),
                    "active_profile": active_p.id,
                    "profile": active_p.to_dict(include_stats=True),
                }
            self._send_json(200, payload)
            return
        self._error(404, "not found")

    # -- POST ------------------------------------------------------------

    def do_POST(self):
        if not self._check_post():
            return
        if self.path == "/api/frame":
            self._handle_frame()
        elif self.path == "/api/start":
            self._handle_start()
        elif self.path == "/api/stop":
            self._handle_stop()
        elif self.path == "/api/discard":
            self._handle_discard()
        elif self.path.startswith("/api/lips/"):
            self._handle_lips()
        elif self.path.startswith("/api/mouth/"):
            self._handle_mouth()
        elif self.path == "/api/profile/select":
            self._handle_profile_select()
        elif self.path == "/api/profile/create":
            self._handle_profile_create()
        elif self.path == "/api/profile/update":
            self._handle_profile_update()
        else:
            self._error(404, "not found")

    def _handle_frame(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "image/jpeg":
            self._drain_body()
            self._error(415, "Content-Type must be image/jpeg")
            return
        body = self._read_body(MAX_FRAME_BYTES)
        if body is None:
            return
        with self.server.lock:
            timestamp_ms = self.server.next_timestamp_ms()
            decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                self._error(400, "body is not a decodable JPEG")
                return
            self.server.mouth.feed(decoded, timestamp_ms)
            mouth_state = self.server.mouth.snapshot()
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            start = time.perf_counter()
            result = self.server.engine.infer(rgb, timestamp_ms)
            process_ms = (time.perf_counter() - start) * 1000.0
            rec = self.server.recording
            if rec is not None and not rec.limit_reached:
                _append_frame(self.server, rec, result, timestamp_ms)
            payload = {
                "raw": [{"label": h.label, "score": h.score} for h in result.hands],
                "mouth": mouth_state,
                "tracking": [{"handedness": h.handedness, "landmarks": h.landmarks,
                              "label": h.label, "score": h.score} for h in result.hands],
                "events": [asdict(event) for event in result.events],
                "hand_count": len(result.hands),
                "process_ms": process_ms,
                "recording": rec is not None,
                "recorded_frames": len(rec.timestamps_ms) if rec else 0,
                "limit_reached": bool(rec and rec.limit_reached),
            }
        # Same decoded camera frame, independent latest-frame lip worker. Hand
        # inference and browser controls never wait for a VSR decode.
        self.server.lips.feed(decoded, timestamp_ms)
        payload["lips"] = self.server.lips.snapshot()
        self._send_json(200, payload)

    def _handle_mouth(self):
        body = self._read_body(MAX_JSON_BYTES)
        if body is None:
            return
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            action = self.path.removeprefix("/api/mouth/")
            with self.server.lock:
                if action == "preview":
                    self.server.mouth.preview()
                elif action == "start":
                    if self.server.recording is not None:
                        raise ValueError("Finish the gesture trial before recording lips")
                    if data.get("profile"):
                        prof = self.server.profile_manager.get_profile(data["profile"])
                        self.server.mouth.set_root(prof.mouths_dir)
                    self.server.mouth.start(data.get("text", ""), data.get("split", ""), data.get("session", ""))
                elif action == "stop":
                    self.server.mouth.stop()
                elif action == "discard":
                    self.server.mouth.discard()
                else:
                    raise ValueError("Unknown mouth capture action")
                state = self.server.mouth.snapshot()
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            self._error(400, str(error))
            return
        self._send_json(200, {**state, "mouth": state})

    def _handle_lips(self):
        body = self._read_body(MAX_JSON_BYTES)
        if body is None:
            return
        action = self.path.removeprefix("/api/lips/")
        if action not in {"enable", "start", "stop", "cancel"}:
            self._error(404, "unknown lip action")
            return
        try:
            getattr(self.server.lips, action)()
        except (ValueError, RuntimeError) as error:
            self._error(400, str(error))
            return
        self._send_json(200, {"lips": self.server.lips.snapshot()})

    def _handle_start(self):
        body = self._read_body(MAX_JSON_BYTES)
        if body is None:
            return
        try:
            data = json.loads(body)
        except ValueError:
            self._error(400, "invalid JSON body")
            return
        label = data.get("label")
        split = data.get("split")
        session = data.get("session")
        profile_id = data.get("profile")
        if not isinstance(label, str) or not _LABEL_RE.match(label):
            self._error(400, "label must be a safe UPPER_SNAKE_CASE identifier")
            return
        if split not in ("train", "validation", "test"):
            self._error(400, "split must be 'train', 'validation', or 'test'")
            return
        if not _valid_session(session):
            self._error(400, "session must be a non-empty filename-safe identifier")
            return
        with self.server.lock:
            if self.server.mouth.snapshot()["recording"]:
                self._error(409, "Finish the mouth trial before recording gestures")
                return
            if self.server.recording is not None:
                self._error(409, "already recording; stop or discard first")
                return
            target_root = self.server.data_root
            if profile_id:
                try:
                    prof = self.server.profile_manager.get_profile(profile_id)
                    target_root = prof.sequences_dir
                except Exception as error:
                    self._error(400, str(error))
                    return
            try:
                _check_session_split(target_root, split, session)
            except ValueError as error:
                self._error(400, str(error))
                return
            # In-memory buffer only; nothing touches disk until stop.
            self.server.recording = _Recording(
                label=label,
                split=split,
                session=session,
                start_monotonic=time.monotonic(),
                data_root=target_root,
            )
        self._send_json(200, {"recording": True, "label": label, "split": split, "session": session})

    def _handle_profile_select(self):
        body = self._read_body(MAX_JSON_BYTES)
        if body is None:
            return
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            profile_id = data.get("id")
            with self.server.lock:
                if self.server.recording is not None:
                    raise ValueError("Cannot switch profile while recording is in progress")
                if self.server.mouth.snapshot()["recording"]:
                    raise ValueError("Cannot switch profile while mouth capture is in progress")
                profile = self.server.profile_manager.set_active_profile(profile_id)
                self.server.data_root = profile.sequences_dir
                self.server.mouth.set_root(profile.mouths_dir)
                payload = profile.to_dict(include_stats=True)
        except (ValueError, KeyError, OSError) as error:
            self._error(400, str(error))
            return
        self._send_json(200, payload)

    def _handle_profile_create(self):
        body = self._read_body(MAX_JSON_BYTES * 8)
        if body is None:
            return
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            profile_id = data.get("id")
            name = data.get("name")
            gestures = data.get("gestures")
            lip_prompts = data.get("lip_prompts")
            with self.server.lock:
                if self.server.recording is not None:
                    raise ValueError("Cannot create/switch profile while recording is in progress")
                if self.server.mouth.snapshot()["recording"]:
                    raise ValueError("Cannot create/switch profile while mouth capture is in progress")
                profile = self.server.profile_manager.create_profile(
                    profile_id=profile_id,
                    name=name,
                    gestures=gestures,
                    lip_prompts=lip_prompts,
                )
                self.server.profile_manager.set_active_profile(profile.id)
                self.server.data_root = profile.sequences_dir
                self.server.mouth.set_root(profile.mouths_dir)
                payload = profile.to_dict(include_stats=True)
        except (ValueError, OSError) as error:
            self._error(400, str(error))
            return
        self._send_json(200, payload)

    def _handle_profile_update(self):
        body = self._read_body(MAX_JSON_BYTES * 8)
        if body is None:
            return
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            profile_id = data.get("id") or self.server.profile_manager.active_profile_id
            name = data.get("name")
            gestures = data.get("gestures")
            lip_prompts = data.get("lip_prompts")
            with self.server.lock:
                if self.server.recording is not None:
                    raise ValueError("Cannot update profile while recording is in progress")
                if self.server.mouth.snapshot()["recording"]:
                    raise ValueError("Cannot update profile while mouth capture is in progress")
                profile = self.server.profile_manager.update_profile(
                    profile_id=profile_id,
                    name=name,
                    gestures=gestures,
                    lip_prompts=lip_prompts,
                )
                payload = profile.to_dict(include_stats=True)
        except (ValueError, KeyError, OSError) as error:
            self._error(400, str(error))
            return
        self._send_json(200, payload)

    def _handle_stop(self):
        self._read_body(MAX_JSON_BYTES)
        with self.server.lock:
            rec = self.server.recording
            if rec is None:
                self._error(400, "not recording")
                return
            if not rec.timestamps_ms:
                # Zero captured frames: report an error and keep recording so
                # the user can retry or discard explicitly.
                self._error(400, "no frames captured; nothing to save")
                return
            try:
                path = _save_sequence(self.server, rec)
            except ValueError as error:
                self._error(400, str(error))
                return
            self.server.recording = None
            self.server.saved_count += 1
            self.server.last_saved = str(path)
            payload = {
                "saved": str(path),
                "recorded_frames": len(rec.timestamps_ms),
                "saved_count": self.server.saved_count,
            }
        self._send_json(200, payload)

    def _handle_discard(self):
        self._read_body(MAX_JSON_BYTES)
        with self.server.lock:
            had = self.server.recording is not None
            self.server.recording = None
        self._send_json(200, {"discarded": had, "recording": False})


def create_server(engine=None, port: int = DEFAULT_PORT, data_root=None,
                  max_frames: int = MAX_RECORD_FRAMES,
                  max_seconds: float = MAX_RECORD_SECONDS,
                  profile_manager=None) -> _LabServer:
    """Build the loopback lab server.

    ``engine`` defaults to a real MediaPipe :class:`Engine`; tests pass a
    fake engine with compatible ``infer``/``close`` methods. The server is
    bound to 127.0.0.1 only; call ``serve_forever`` to start it.
    """
    if engine is None:
        engine = Engine()
    return _LabServer(
        ("127.0.0.1", port),
        engine=engine,
        data_root=data_root,
        max_frames=max_frames,
        max_seconds=max_seconds,
        profile_manager=profile_manager,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None, help="active profile id")
    parser.add_argument("--model", type=Path, default=None, help="gesture .task model path")
    args = parser.parse_args(argv)

    try:
        engine = Engine(model_path=args.model)
    except FileNotFoundError as error:
        parser.exit(2, f"{error}\n")
    try:
        # One tiny warmup inference so the first real frame is not slowed by
        # lazy backend initialization. Timestamp goes through the server's
        # monotonic helper, so warmup cannot break strict ordering.
        server = create_server(engine=engine, port=args.port, data_root=args.data_root)
        if args.profile:
            server.profile_manager.set_active_profile(args.profile)
            p = server.profile_manager.get_active_profile()
            server.data_root = p.sequences_dir
            server.mouth.set_root(p.mouths_dir)
        warmup = np.zeros((64, 64, 3), dtype=np.uint8)
        engine.infer(warmup, server.next_timestamp_ms())
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"QuietBridge sign lab ready at {url} (loopback only, no uploads)")
        print(f"Sequences save to {server.data_root}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()
            if server.recording is not None:
                print("Recording discarded without saving (stop was not called).")
    finally:
        # On shutdown the model is closed; an unsaved buffer is never written.
        engine.close()


if __name__ == "__main__":
    main()

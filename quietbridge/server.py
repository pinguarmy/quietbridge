"""QuietBridge loopback server.

Run ``python -m quietbridge.server`` (or ``quietbridge``): the page opens in the
default browser once the server listens, and asks the browser for camera access
(the default). ``--rehearsal`` uses synthetic inputs only; ``--server-camera``
makes this process open a local camera itself.

Opening the page is best-effort and skipped with ``--no-open``, when
``QUIETBRIDGE_NO_OPEN`` is set to anything non-empty, or on Linux/BSD when
neither ``DISPLAY`` nor ``WAYLAND_DISPLAY`` is set (an SSH session or a
headless box, where Python's webbrowser would start a text browser inside the
terminal). Whether stdout is a terminal does not matter: a launcher without a
console (Start menu, .desktop file) is exactly when the page should open.

If the port is taken by a running QuietBridge, that one's page is opened and
the command exits 0; any other program on the port is named in a one-line
error (exit 2). ``POST /api/shutdown`` stops the app from the page.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
import os
import secrets
import socket
import sys
import threading
import traceback
import urllib.request
import webbrowser

from .runtime import Runtime, ProfileSwitchConflict
from .profiles import (ProfileManager, load_speech_settings, profile_configuration,
                       save_speech_settings)
from . import assets as setup_assets
from .speech import RATES, Speaker

UI = Path(__file__).with_name("ui")
# Idle keep-alive connections close after this many seconds; server_close()
# also shuts down every open connection so restarts never reuse old handlers.
IDLE_TIMEOUT_S = 15
# Largest ignored request body drained to keep a keep-alive connection usable.
MAX_DRAIN_BYTES = 1024 * 1024


class _LazyLabEngine:
    """Create the real tracking engine only for an explicit frame."""

    def __init__(self, rehearsal=False):
        self.rehearsal = rehearsal
        self.engine = None

    @property
    def ready(self):
        return self.engine is not None

    def infer(self, frame, timestamp_ms):
        if self.rehearsal:
            raise RuntimeError("Camera tracking is unavailable in rehearsal mode")
        if self.engine is None:
            from .signs.lab import Engine
            self.engine = Engine()
        return self.engine.infer(frame, timestamp_ms)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def app_version():
    try:
        from importlib.metadata import version
        return version("lipflow")
    except Exception:
        return "unknown"


def _started_with(runtime):
    # Strict checks: test doubles answer any attribute with a truthy object.
    if getattr(runtime, "rehearsal", False) is True:
        return "rehearsal"
    if getattr(runtime, "browser_camera", False) is True:
        return "browser_camera"
    return "server_camera"


def _runtime_state(runtime, key):
    state = getattr(runtime, "state", None)
    value = state.get(key) if isinstance(state, dict) else None
    return value if isinstance(value, str) else None


def _clean_voices(voices):
    if not isinstance(voices, list):
        return []
    return [{"id": v["id"], "label": v["label"]} for v in voices
            if isinstance(v, dict) and isinstance(v.get("id"), str) and isinstance(v.get("label"), str)]


class _SpeechSettings:
    """Rate and voice for the next utterance, saved per profile in profiles/<id>/speech.json.

    Only a real speech.Speaker receives them; with a test double the choice is
    still validated, saved and reported (with no voices to choose from).
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = threading.Lock()
        self.rate, self.voice = "normal", ""
        self.profile_id = object()  # forces the first sync
        speaker = self.speaker()
        if speaker is not None:
            speaker.voices()  # start the bounded background listing now

    def speaker(self):
        speaker = getattr(self.runtime, "speaker", None)
        return speaker if isinstance(speaker, Speaker) else None

    def _profile(self):
        profile = getattr(self.runtime, "profile", None)
        return profile if isinstance(getattr(profile, "id", None), str) and \
            isinstance(getattr(profile, "profile_dir", None), Path) else None

    def sync(self):
        """Apply the active profile's saved settings once per profile (also after a switch)."""
        profile = self._profile()
        profile_id = profile.id if profile is not None else None
        with self.lock:
            if profile_id == self.profile_id:
                return
            self.profile_id = profile_id
            saved = load_speech_settings(profile) if profile is not None else {"rate": "normal", "voice": ""}
            self.rate, self.voice = saved["rate"], saved["voice"]
            speaker = self.speaker()
            if speaker is not None:
                speaker.set_rate(self.rate)
                speaker.set_voice(self.voice, strict=False)

    def payload(self):
        self.sync()
        speaker = self.speaker()
        voices, voice = [], ""
        if speaker is not None:
            try:
                voices = _clean_voices(speaker.voices())
                current = speaker.settings().get("voice", "")
                voice = current if isinstance(current, str) and any(v["id"] == current for v in voices) else ""
            except Exception:
                voices, voice = [], ""
        return {"rate": self.rate, "voice": voice, "voices": voices}

    def update(self, data):
        unknown = set(data) - {"rate", "voice"}
        if unknown:
            raise ValueError(f"Unknown speech setting: {sorted(unknown)[0]}")
        self.sync()
        rate = data.get("rate", self.rate)
        voice = data.get("voice", self.voice)
        if rate not in RATES:
            raise ValueError("Speaking rate must be slow, normal or fast")
        if not isinstance(voice, str):
            raise ValueError("Voice must be text")
        speaker = self.speaker()
        if voice:
            installed = _clean_voices(speaker.voices(wait=True)) if speaker is not None else []
            if not any(v["id"] == voice for v in installed):
                raise ValueError("That voice is not installed on this computer; choose one from the list")
        with self.lock:
            profile = self._profile()
            if profile is not None:
                save_speech_settings(profile, rate, voice)
            self.rate, self.voice = rate, voice
            if speaker is not None:
                speaker.set_rate(rate)
                speaker.set_voice(voice, strict=False)  # validated above


def make_server(runtime, port=8765, *, downloads=None):
    token = secrets.token_urlsafe(32)
    server_instance = secrets.token_urlsafe(16)
    app_info = {"version": app_version(), "started_with": _started_with(runtime)}
    speech_settings = _SpeechSettings(runtime)
    reload_lock = threading.Lock()
    pending_reload = set()  # groups installed after this process tried to load them

    def models_unloaded(group_id):
        if getattr(runtime, "rehearsal", False) is True:
            return False
        if group_id == "lips":
            return getattr(runtime, "enable_lips", False) is True and _runtime_state(runtime, "lips") != "ready"
        return getattr(runtime, "isolated_sign_model_path", None) is None and \
            _runtime_state(runtime, "signs") != "ready"

    def group_ready(group_id):
        # Never loads anything: the page offers the existing explicit reload.
        if models_unloaded(group_id):
            with reload_lock:
                pending_reload.add(group_id)

    setup_downloads = downloads if downloads is not None else setup_assets.SetupDownloads(on_ready=group_ready)
    if downloads is not None and getattr(downloads, "_on_ready", None) is None:
        downloads._on_ready = group_ready

    def setup_payload():
        groups = []
        for group_id in ("lips", "gestures"):
            try:
                groups.append(setup_downloads.group_state(group_id))
            except Exception as exc:  # a broken path must not break polling
                info = setup_assets.GROUPS[group_id]
                groups.append({"id": group_id, "label": info["label"], "status": "error",
                               "bytes_done": 0, "bytes_total": 0,
                               "error": f"Could not check the model files ({exc.__class__.__name__}).",
                               "size_note": "", "license_note": info["license_note"]})
        with reload_lock:
            reload_needed = bool(pending_reload)
        return {"busy": setup_downloads.busy, "reload_needed": reload_needed, "assets": groups}

    def state_payload():
        return {**runtime.snapshot(), "token": token, "server_instance": server_instance,
                "setup": setup_payload(), "speech_settings": speech_settings.payload(),
                "app": dict(app_info)}

    def setup_action(path, data):
        group_id = data.get("id")
        if not isinstance(group_id, str) or group_id not in setup_assets.GROUPS:
            raise ValueError("Unknown download; choose lips or gestures")
        if path == "/api/setup/cancel":
            setup_downloads.cancel(group_id)
            return
        if data.get("accept_license") is not True:
            raise ValueError("Read and accept the model's license note before downloading")
        if getattr(runtime, "rehearsal", False) is True:
            raise ValueError("Downloads are off in rehearsal mode, which never loads models. "
                             "Start QuietBridge without --rehearsal to download them.")
        setup_downloads.start(group_id)

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 keep-alive: the page polls state, tracking and frames many
        # times per second. Every response carries Content-Length, and an
        # unread request body is drained or the connection is closed.
        protocol_version = "HTTP/1.1"
        timeout = IDLE_TIMEOUT_S  # idle keep-alive connections do not linger
        # Headers and body are separate writes; with Nagle plus the client's
        # delayed ACK each keep-alive response would stall for ~40 ms.
        disable_nagle_algorithm = True

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.server.track_connection(self.connection, True)

        def finish(self):
            try:
                super().finish()
            finally:
                self.server.track_connection(self.connection, False)

        def _begin(self):
            self._body_consumed = False
            self._responded = False

        def _read_body(self, length):
            self._body_consumed = True
            return self.rfile.read(length)

        def _drain_unread_body(self):
            """Keep the connection usable after a response that ignored the body."""
            if getattr(self, "_body_consumed", True):
                return
            self._body_consumed = True
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                return
            try:
                remaining = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                self.close_connection = True
                return
            if remaining <= 0:
                return
            if remaining > MAX_DRAIN_BYTES:
                self.close_connection = True
                remaining = MAX_DRAIN_BYTES  # read what we can so the reply is not reset
            previous = self.connection.gettimeout()
            self.connection.settimeout(0.25)
            try:
                while remaining:
                    chunk = self.rfile.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except (TimeoutError, OSError):
                pass
            finally:
                try:
                    self.connection.settimeout(previous)
                except OSError:
                    pass
            if remaining:
                self.close_connection = True

        def _dispatch(self, handler):
            self._begin()
            try:
                handler()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:
                # Unexpected failures answer with a generic 500; details stay in the log.
                traceback.print_exc(file=sys.stderr)
                if self._responded:
                    self.close_connection = True
                    return
                try:
                    self._json(500, {"error": "Internal server error; see the server log"})
                except OSError:
                    self.close_connection = True

        def _local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self._json(403, {"error": "Use the localhost URL printed by the server"})
                return False
            return True

        def _send(self, status, body, content_type, headers=()):
            self._drain_unread_body()
            if self.server.closing:
                self.close_connection = True
            self._responded = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            for name, value in headers:
                self.send_header(name, value)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, data):
            self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            self._dispatch(self._get)

        def do_POST(self):
            self._dispatch(self._post)

        def _get(self):
            if not self._local_request():
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._json(200, state_payload())
            elif path == "/api/cleanup/config":
                self._json(200, runtime.cleanup_config())
            elif path == "/api/profiles":
                manager = getattr(runtime, "profile_manager", None) or ProfileManager()
                with runtime.lock:
                    active_id = runtime.profile.id if runtime.profile else manager.active_profile_id
                self._json(200, {
                    "active_profile": active_id,
                    "profiles": manager.list_profiles(include_stats=True),
                })
            elif path == "/api/tracking":
                self._json(200, runtime.tracking_snapshot())
            elif path == "/api/frame":
                # Encoded on demand, outside the runtime lock.
                jpeg = runtime.preview_jpeg()
                if jpeg:
                    self._send(200, jpeg, "image/jpeg")
                else:
                    self._json(503, {"error": "No camera frame yet"})
            elif path == "/record":
                try:
                    lab = self.server.personalize_server()
                except (OSError, ValueError, RuntimeError) as exc:
                    self._json(503, {"error": f"Personalize could not start: {exc}"})
                    return
                return_to = f"http://127.0.0.1:{self.server.server_port}/"
                location = f"http://127.0.0.1:{lab.server_port}/?" + urlencode({"return_to": return_to})
                self._send(302, b"", "text/plain; charset=utf-8", [("Location", location)])
            elif path in {"/", "/index.html", "/communicate", "/about", "/settings", "/settings.html",
                          "/app.js", "/cleanup-settings.js", "/style.css", "/design.css", "/favicon.svg"}:
                filename = {"/": "index.html", "/communicate": "index.html", "/about": "about.html",
                            "/settings": "settings.html"}.get(path, path[1:])
                mime = {"index.html": "text/html", "about.html": "text/html", "settings.html": "text/html",
                        "app.js": "text/javascript", "cleanup-settings.js": "text/javascript", "style.css": "text/css",
                        "design.css": "text/css", "favicon.svg": "image/svg+xml"}[filename]
                self._send(200, (UI / filename).read_bytes(), mime + "; charset=utf-8")
            else:
                self._json(404, {"error": "Not found"})

        def _post(self):
            if not self._local_request():
                return
            origin = self.headers.get("Origin")
            if (origin and origin not in {f"http://127.0.0.1:{self.server.server_port}",
                                         f"http://localhost:{self.server.server_port}"}) or self.headers.get("X-Lipflow-Token") != token:
                self._json(403, {"error": "Reload the local demo page"})
                return
            if self.path == "/api/frame":
                self._handle_frame()
                return
            if self.path in {"/api/profile/select", "/api/action/profile_select"}:
                self._handle_profile_select()
                return
            if self.path == "/api/shutdown":
                self._handle_shutdown()
                return
            cleanup_path = self.path in {"/api/cleanup/config", "/api/cleanup/test"}
            setup_path = self.path in {"/api/setup/download", "/api/setup/cancel", "/api/speech/settings",
                                       "/api/speech/test"}
            if not cleanup_path and not setup_path and not self.path.startswith("/api/action/"):
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                data = json.loads(self._read_body(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/cleanup/config":
                    payload = runtime.configure_cleanup(data)
                elif self.path == "/api/cleanup/test":
                    payload = runtime.test_cleanup()
                elif self.path == "/api/speech/settings":
                    speech_settings.update(data)
                    payload = state_payload()
                elif self.path == "/api/speech/test":
                    # An explicit request to hear the current voice; the message is untouched.
                    runtime.speak_sample()
                    payload = state_payload()
                elif setup_path:
                    setup_action(self.path, data)
                    payload = state_payload()
                else:
                    runtime.action(self.path.removeprefix("/api/action/"), data)
                    payload = state_payload()
            except (BrokenPipeError, ConnectionResetError):
                raise
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, payload)

        def _handle_frame(self):
            if not runtime.browser_camera or runtime.rehearsal:
                self.close_connection = True
                self._json(400, {"error": "Browser camera input is not enabled"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
                self.close_connection = True
                self._json(415, {"error": "Content-Type must be image/jpeg"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length > 512 * 1024:
                # Drain the declared request without retaining it. Closing before
                # the body is read races the client's send and can surface as a
                # BrokenPipe instead of the intended 413 response.
                self._body_consumed = True
                remaining = length
                self.connection.settimeout(0.25)
                try:
                    while remaining:
                        chunk = self.rfile.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                except (TimeoutError, OSError):
                    # Header-only rejection and full-body clients use the same 413.
                    pass
                self.close_connection = True
                self._json(413, {"error": "Frame exceeds 512 KiB"})
                return
            if length <= 0:
                self.close_connection = True
                self._json(400, {"error": "A JPEG frame is required"})
                return
            body = self._read_body(length)
            import cv2
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self._json(400, {"error": "Body is not a decodable JPEG"})
                return
            runtime.receive_frame(frame)
            self._json(200, {"ok": True})

        def _handle_profile_select(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                body = self._read_body(length)
                try:
                    data = json.loads(body)
                except ValueError:
                    raise ValueError("Invalid JSON body")
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                profile_id = data.get("id") or data.get("profile")
                if not profile_id or not isinstance(profile_id, str):
                    raise ValueError("Profile id must be a non-empty string")
                profile = runtime.switch_profile(profile_id.strip())
                with reload_lock:
                    pending_reload.clear()  # the workers were restarted: new files are loaded
                payload = profile.to_dict(include_stats=True)
                payload["active_profile"] = profile.id
                payload["profile"] = profile.to_dict(include_stats=True)
                payload["ok"] = True
                payload.update(state_payload())
            except ProfileSwitchConflict as exc:
                self._json(409, {"error": str(exc)})
                return
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._json(400, {"error": str(exc)})
                return
            # Anything else reaches _dispatch: a generic 500, details in the log.
            self._json(200, payload)

        def _handle_shutdown(self):
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= 8192:
                self._json(400, {"error": "Invalid request size"})
                return
            body = self._read_body(length) if length else b"{}"
            try:
                data = json.loads(body or b"{}")
            except ValueError:
                data = None
            if not isinstance(data, dict):
                self._json(400, {"error": "Expected a JSON object"})
                return
            self.close_connection = True
            self._json(200, {"ok": True})
            self.server.request_shutdown()

    class MainServer(ThreadingHTTPServer):
        daemon_threads = True
        # On Windows SO_REUSEADDR lets a second process bind the same port and
        # take over requests without any error; ask for exclusive use there.
        allow_reuse_address = sys.platform != "win32"

        def server_bind(self):
            if sys.platform == "win32":
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

        def __init__(self, address):
            # Set before binding: a failed bind calls server_close() from
            # TCPServer.__init__, which must report the real OSError.
            self._lab_lock = threading.Lock()
            self._lab = None
            self._lab_thread = None
            self._closed = False
            self._shutdown_requested = threading.Event()
            self.setup_downloads = setup_downloads
            self._connections_lock = threading.Lock()
            self._connections = set()
            super().__init__(address, Handler)

        @property
        def closing(self):
            return self._closed

        def track_connection(self, connection, open_):
            with self._connections_lock:
                if open_:
                    self._connections.add(connection)
                else:
                    self._connections.discard(connection)

        def _shutdown_connections(self):
            with self._connections_lock:
                connections = list(self._connections)
            for connection in connections:
                try:
                    # Wake handler threads blocked on an idle keep-alive read.
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        def personalize_server(self):
            with self._lab_lock:
                if self._closed:
                    raise RuntimeError("Server is closing")
                if self._lab is None:
                    from .signs import lab as lab_module
                    engine = _LazyLabEngine(rehearsal=runtime.rehearsal)
                    manager = getattr(runtime, "profile_manager", None) or ProfileManager()
                    lab = lab_module.create_server(
                        engine=engine, port=0, profile_manager=manager,
                        sign_model=getattr(runtime, "isolated_sign_model_path", None))
                    if runtime.rehearsal:
                        class RehearsalHandler(lab_module._Handler):
                            def do_POST(self):
                                if self.path in {"/api/frame", "/api/start", "/api/paired/start",
                                                 "/api/mouth/preview", "/api/mouth/start",
                                                 "/api/lips/enable", "/api/lips/start"}:
                                    if self._check_post():
                                        self._drain_body()
                                        self._error(400, "Camera tracking is unavailable in rehearsal mode")
                                    return
                                super().do_POST()
                        lab.RequestHandlerClass = RehearsalHandler
                    thread = threading.Thread(target=lab.serve_forever,
                                              kwargs={"poll_interval": 0.1}, daemon=True,
                                              name="quietbridge-personalize")
                    thread.start()
                    self._lab, self._lab_thread = lab, thread
                return self._lab

        @property
        def shutdown_requested(self):
            return self._shutdown_requested.is_set()

        def request_shutdown(self):
            """Stop serve_forever() from another thread (main() then closes the runtime)."""
            if self._shutdown_requested.is_set():
                return
            self._shutdown_requested.set()
            threading.Thread(target=self.shutdown, name="quietbridge-shutdown", daemon=True).start()

        def server_close(self):
            setup_downloads.close()
            with self._lab_lock:
                self._closed = True
                lab, thread = self._lab, self._lab_thread
                self._lab = self._lab_thread = None
            if lab is not None:
                lab.shutdown()
                thread.join()
                lab.server_close()
                lab.engine.close()
            super().server_close()
            self._shutdown_connections()

    return MainServer(("127.0.0.1", port))


def open_in_browser(url, opener=None):
    """Best-effort: open the printed URL without delaying or failing the server."""
    def run():
        try:
            (opener or webbrowser.open)(url)
        except Exception as exc:  # never fatal: the URL is printed as well
            print(f"Could not open a browser ({exc}); open {url} manually.",
                  file=sys.stderr, flush=True)
    thread = threading.Thread(target=run, name="quietbridge-open-browser", daemon=True)
    thread.start()
    return thread


def should_open_browser(no_open=False, environ=None, platform=None):
    """The documented rule for opening the page (see the module docstring)."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if no_open or (environ.get("QUIETBRIDGE_NO_OPEN") or "").strip():
        return False
    if platform == "win32" or platform == "darwin":
        return True
    return bool((environ.get("DISPLAY") or "").strip() or (environ.get("WAYLAND_DISPLAY") or "").strip())


def probe_running(port, timeout=2.0):
    """True when http://127.0.0.1:<port>/api/state answers like QuietBridge."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never a proxy
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/state", timeout=timeout) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read(4 * 1024 * 1024))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("server_instance"), str) and "revision" in data


def _bind_failed(port, exc, open_page):
    """Port taken: hand over to a running QuietBridge (exit 0) or explain (exit 2). No traceback."""
    url = f"http://127.0.0.1:{port}"
    if probe_running(port):
        print(f"QuietBridge is already running at {url}", flush=True)
        if open_page:
            open_in_browser(url).join(10)
        return 0
    reason = getattr(exc, "strerror", None) or str(exc)
    print(f"QuietBridge could not use port {port} ({reason}): another program is using it. "
          f"Start QuietBridge with --port and another number, for example --port {port + 1 if port < 65535 else 8766}.",
          file=sys.stderr, flush=True)
    return 2


def _missing_downloads_line(runtime):
    if getattr(runtime, "rehearsal", False) is True:
        return ""
    names = []
    try:
        if not setup_assets.group_ready("lips") and getattr(runtime, "enable_lips", False) is True:
            names.append("lip reading")
        if not setup_assets.group_ready("gestures") and \
                getattr(runtime, "isolated_sign_model_path", None) is None:
            names.append("hand gestures")
    except Exception:
        return ""
    if not names:
        return ""
    return (f"Not downloaded yet: {' and '.join(names)}. Typing works now; download them from Setup "
            "in the page (one time).")


def main():
    parser = argparse.ArgumentParser(
        description="QuietBridge loopback server. By default the page uses the browser's camera "
                    "(the browser asks for permission); recognition runs on this computer.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rehearsal", action="store_true", help="Synthetic inputs only; no camera or model loading")
    source.add_argument("--browser-camera", action="store_true",
                        help="Use the browser's camera (the default; accepted for compatibility)")
    source.add_argument("--server-camera", action="store_true",
                        help="Open a camera from this server process through OpenCV instead of the browser")
    parser.add_argument("--sign-only", action="store_true", help="Do not load the lip model")
    parser.add_argument("--camera", type=int, metavar="N",
                        help="Camera index for --server-camera (default 0)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vocabulary", type=Path, help="JSON label-to-phrase mapping, default quietbridge/vocabulary.json")
    parser.add_argument("--profile", help="Profile id; also selects its trained custom motion model when present")
    parser.add_argument("--lip-model", type=Path, help="Explicit personal VSR delta; stock remains the default")
    parser.add_argument("--lip-language-model", type=Path, help="Explicit personal language-model weights")
    parser.add_argument("--sign-model", type=Path, help="Explicit local BSL/ASL isolated-sign manifest; replaces stock/personal gesture backend")
    parser.add_argument("--max-utterance-seconds", type=float, default=30.0)
    parser.add_argument("--open", action="store_true",
                        help="Accepted for compatibility: the page opens by default")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open the page in the default browser (also: QUIETBRIDGE_NO_OPEN=1)")
    args = parser.parse_args()
    open_page = should_open_browser(args.no_open)
    if args.camera is not None and not args.server_camera:
        parser.error("--camera selects a camera opened by this server; add --server-camera "
                     "(by default the page uses the browser's camera)")
    browser_camera = not (args.rehearsal or args.server_camera)
    from .profiles import (ProfileManager, profile_language_model,
                           profile_lip_model)
    profiles = ProfileManager()
    profile = profiles.get_profile(args.profile) if args.profile else profiles.get_active_profile()
    vocabulary, motion_model, profile_warning = profile_configuration(profile, args.vocabulary)
    promoted_lip = profile_lip_model(profile)
    promoted_language = profile_language_model(profile)
    lip_model = args.lip_model or (promoted_lip if promoted_lip.is_file() else None)
    language_model = args.lip_language_model or (promoted_language if promoted_language.is_file() else None)
    # The active profile is always the runtime's profile, also when it has no
    # promoted model yet (stock models are then used and the warning says so).
    # Its phrases, correction store and model-change notice then work from the
    # start, and the shown profile matches what "Reload current models" selects.
    try:
        runtime = Runtime(rehearsal=args.rehearsal, browser_camera=browser_camera,
                          camera=args.camera if args.camera is not None else 0,
                          enable_lips=not args.sign_only,
                          vocabulary=vocabulary, lip_model=lip_model,
                          lip_language_model=language_model,
                          motion_model=motion_model, sign_model=args.sign_model,
                          profile_name=profile.name,
                          profile=profile,
                          configuration_warning=profile_warning,
                          max_utterance_seconds=args.max_utterance_seconds,
                          profile_manager=profiles,
                          explicit_lip_model=args.lip_model,
                          explicit_lip_language_model=args.lip_language_model,
                          explicit_vocabulary_path=args.vocabulary)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        server = make_server(runtime, args.port)
    except OSError as exc:
        runtime.close()
        code = _bind_failed(args.port, exc, open_page)
        if code:
            raise SystemExit(code)
        return
    runtime.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"QuietBridge: {url}", flush=True)
    print("REHEARSAL — synthetic inputs, no recognition" if args.rehearsal else
          "BROWSER CAMERA — open the page and allow camera access there. Recognition runs on "
          "this computer; wait for warmup before recording." if browser_camera else
          f"SERVER CAMERA {args.camera or 0} — this process opens the camera itself. Recognition "
          "runs on this computer; wait for warmup before recording.", flush=True)
    missing = _missing_downloads_line(runtime)
    if missing:
        print(missing, flush=True)
    print("Stop with Ctrl+C here, or Quit in the page's Settings.", flush=True)
    if open_page:
        # The socket is already listening; requests queue until serve_forever runs.
        open_in_browser(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()
    if server.shutdown_requested:
        print("QuietBridge stopped from the page.", flush=True)


if __name__ == "__main__":
    main()

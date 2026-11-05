"""Loopback-only demo server. Run: python -m hackathon.server --rehearsal."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
import secrets
import threading

from .runtime import Runtime, ProfileSwitchConflict
from .profiles import ProfileManager, profile_configuration

UI = Path(__file__).with_name("ui")


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


def make_server(runtime, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self._json(403, {"error": "Use the localhost URL printed by the server"})
                return False
            return True

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, data):
            self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if not self._local_request():
                return
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/state":
                    self._json(200, {**runtime.snapshot(), "token": token})
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
                    with runtime.lock:
                        jpeg = runtime.jpeg
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
                    self.send_response(302)
                    return_to = f"http://127.0.0.1:{self.server.server_port}/"
                    self.send_header("Location", f"http://127.0.0.1:{lab.server_port}/?" + urlencode({"return_to": return_to}))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif path in {"/", "/index.html", "/settings", "/settings.html", "/app.js", "/cleanup-settings.js", "/cleanup-display.js", "/style.css"}:
                    filename = {"/": "index.html", "/settings": "settings.html"}.get(path, path[1:])
                    mime = {"index.html": "text/html", "settings.html": "text/html", "app.js": "text/javascript", "cleanup-settings.js": "text/javascript", "cleanup-display.js": "text/javascript", "style.css": "text/css"}[filename]
                    self._send(200, (UI / filename).read_bytes(), mime + "; charset=utf-8")
                else:
                    self._json(404, {"error": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
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
            cleanup_path = self.path in {"/api/cleanup/config", "/api/cleanup/test"}
            if not cleanup_path and not self.path.startswith("/api/action/"):
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/cleanup/config":
                    self._json(200, runtime.configure_cleanup(data))
                elif self.path == "/api/cleanup/test":
                    self._json(200, runtime.test_cleanup())
                else:
                    runtime.action(self.path.removeprefix("/api/action/"), data)
                    self._json(200, runtime.snapshot())
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                self._json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

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
            body = self.rfile.read(length)
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
                body = self.rfile.read(length)
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
                payload = profile.to_dict(include_stats=True)
                payload["active_profile"] = profile.id
                payload["profile"] = profile.to_dict(include_stats=True)
                payload["ok"] = True
                payload.update(runtime.snapshot())
                payload["token"] = token
                self._json(200, payload)
            except ProfileSwitchConflict as exc:
                self._json(409, {"error": str(exc)})
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._json(500, {"error": str(exc)})

    class MainServer(ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self, address):
            super().__init__(address, Handler)
            self._lab_lock = threading.Lock()
            self._lab = None
            self._lab_thread = None
            self._closed = False

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

        def server_close(self):
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

    return MainServer(("127.0.0.1", port))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rehearsal", action="store_true", help="Synthetic inputs only; no camera or model loading")
    source.add_argument("--browser-camera", action="store_true", help="Receive camera frames from the browser")
    parser.add_argument("--sign-only", action="store_true", help="Do not load the lip model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vocabulary", type=Path, help="JSON label-to-phrase mapping, default hackathon/vocabulary.json")
    parser.add_argument("--profile", help="Profile id; also selects its trained custom motion model when present")
    parser.add_argument("--lip-model", type=Path, help="Explicit personal VSR delta; stock remains the default")
    parser.add_argument("--lip-language-model", type=Path, help="Explicit personal language-model weights")
    parser.add_argument("--sign-model", type=Path, help="Explicit local BSL/ASL isolated-sign manifest; replaces stock/personal gesture backend")
    parser.add_argument("--max-utterance-seconds", type=float, default=30.0)
    args = parser.parse_args()
    from .profiles import (ProfileManager, profile_language_model,
                           profile_lip_model)
    profiles = ProfileManager()
    profile = profiles.get_profile(args.profile) if args.profile else profiles.get_active_profile()
    vocabulary, motion_model, profile_warning = profile_configuration(profile, args.vocabulary)
    promoted_lip = profile_lip_model(profile)
    promoted_language = profile_language_model(profile)
    lip_model = args.lip_model or (promoted_lip if promoted_lip.is_file() else None)
    language_model = args.lip_language_model or (promoted_language if promoted_language.is_file() else None)
    use_profile = args.profile is not None or motion_model is not None or lip_model is not None or language_model is not None
    runtime = Runtime(rehearsal=args.rehearsal, browser_camera=args.browser_camera,
                      camera=args.camera, enable_lips=not args.sign_only,
                      vocabulary=vocabulary, lip_model=lip_model,
                      lip_language_model=language_model,
                      motion_model=motion_model, sign_model=args.sign_model,
                      profile_name=profile.name if use_profile else None,
                      profile=profile if use_profile else None,
                      configuration_warning=profile_warning,
                      max_utterance_seconds=args.max_utterance_seconds,
                      profile_manager=profiles,
                      explicit_lip_model=args.lip_model,
                      explicit_lip_language_model=args.lip_language_model,
                      explicit_vocabulary_path=args.vocabulary)
    server = make_server(runtime, args.port)
    runtime.start()
    print(f"QuietBridge: http://127.0.0.1:{server.server_port}", flush=True)
    print("REHEARSAL — synthetic inputs, no recognition" if args.rehearsal else
          "BROWSER CAMERA — allow camera access in the browser. Wait for warmup before recording."
          if args.browser_camera else
          "LOCAL — no cloud inference. Wait for warmup before recording.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()

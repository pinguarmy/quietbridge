"""Loopback-only demo server. Run: python -m hackathon.server --rehearsal."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets

from .runtime import Runtime

UI = Path(__file__).with_name("ui")


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
                elif path == "/api/frame":
                    with runtime.lock:
                        jpeg = runtime.jpeg
                    if jpeg:
                        self._send(200, jpeg, "image/jpeg")
                    else:
                        self._json(503, {"error": "No camera frame yet"})
                elif path in {"/", "/index.html", "/settings", "/settings.html", "/app.js", "/style.css"}:
                    filename = {"/": "index.html", "/settings": "settings.html"}.get(path, path[1:])
                    mime = {"index.html": "text/html", "settings.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[filename]
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
            if not self.path.startswith("/api/action/"):
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                runtime.action(self.path.removeprefix("/api/action/"), data)
                self._json(200, runtime.snapshot())
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                self._json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", action="store_true", help="Synthetic inputs only; no camera or model loading")
    parser.add_argument("--sign-only", action="store_true", help="Do not load the lip model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vocabulary", type=Path, help="JSON label-to-phrase mapping, default hackathon/vocabulary.json")
    args = parser.parse_args()
    from .fusion import load_vocabulary
    runtime = Runtime(rehearsal=args.rehearsal, camera=args.camera, enable_lips=not args.sign_only,
                      vocabulary=load_vocabulary(args.vocabulary) if args.vocabulary else None)
    server = make_server(runtime, args.port)
    runtime.start()
    print(f"QuietBridge: http://127.0.0.1:{server.server_port}", flush=True)
    print("REHEARSAL — synthetic inputs, no recognition" if args.rehearsal else
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

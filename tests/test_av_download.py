"""Whisper-mode weights download: resumes an interrupted transfer, publishes only complete files."""
import http.client
import http.server
import threading

import pytest

from lipflow import av

BODY = bytes(range(256)) * 4096  # 1 MiB


class Handler(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.hits.append((self.path, self.headers.get("Range")))
        if self.path.endswith("config.json"):
            body = b'{"odim": 5049}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        rng = self.headers.get("Range")
        if rng is None:  # first try: promise everything, deliver half, hang up
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY[:len(BODY) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        start = int(rng.split("=")[1].split("-")[0])
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        self.send_header("Content-Length", str(len(BODY) - start))
        self.end_headers()
        self.wfile.write(BODY[start:])


@pytest.fixture()
def server(monkeypatch, tmp_path):
    Handler.hits = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(av, "URL", f"http://127.0.0.1:{httpd.server_address[1]}/")
    monkeypatch.setattr(av, "AV_DIR", str(tmp_path / "av"))
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    yield
    httpd.shutdown()


@pytest.mark.usefixtures("server")
def test_interrupted_download_resumes_and_only_then_appears(tmp_path):
    with pytest.raises((OSError, http.client.HTTPException)):
        av.download(timeout=10)
    assert not av.available()  # never a half file under the final name
    assert (tmp_path / "av" / "model.safetensors.part").stat().st_size == len(BODY) // 2
    seen = []
    av.download(progress=seen.append, timeout=10)
    assert av.available()
    assert (tmp_path / "av" / "model.safetensors").read_bytes() == BODY
    assert Handler.hits[-1] == ("/model.safetensors", f"bytes={len(BODY) // 2}-")
    assert seen and seen[-1] == pytest.approx(100.0)
    assert sum(path.endswith("config.json") for path, _ in Handler.hits) == 1  # kept from the first call


def test_missing_safetensors_is_a_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    with pytest.raises(RuntimeError, match="safetensors"):
        av.AVReader()

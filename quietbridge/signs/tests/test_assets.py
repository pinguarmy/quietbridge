import http.server
import threading

import pytest

from quietbridge.signs import assets


def test_model_path_override_is_expanded(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", "~/demo/gesture.task")
    assert assets.default_model_path().is_absolute()
    assert assets.default_model_path().name == "gesture.task"


class _BadModel(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b"x" * assets.MODEL_SIZE  # the right size, the wrong content
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_download_verifies_before_writing(monkeypatch, tmp_path):
    path = tmp_path / "gesture.task"
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BadModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(assets, "MODEL_URL", f"http://127.0.0.1:{server.server_port}/gesture.task")
    try:
        with pytest.raises(ValueError, match="did not match the expected file"):
            assets.download_model(path)
    finally:
        server.shutdown()
        server.server_close()
    assert not path.exists()
    assert not path.with_suffix(".task.part").exists()


def test_existing_file_is_not_overwritten(monkeypatch, tmp_path):
    path = tmp_path / "gesture.task"
    path.write_bytes(b"my local model")
    with pytest.raises(ValueError, match="not overwriting"):
        assets.download_model(path)
    assert path.read_bytes() == b"my local model"

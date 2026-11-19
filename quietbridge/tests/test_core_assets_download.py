"""Model downloads install only complete files (local HTTP server, no network)."""
import socket
import threading

import pytest

from quietbridge import assets

BODY = bytes(range(256)) * 40  # 10240 bytes


def _server(respond):
    """Raw socket server: respond(request_headers) -> bytes to send, then close."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    requests = []

    def serve():
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return
            with conn:
                head = conn.recv(65536).decode("latin-1")
                headers = {line.split(":", 1)[0].lower(): line.split(":", 1)[1].strip()
                           for line in head.split("\r\n")[1:] if ":" in line}
                requests.append(headers)
                conn.sendall(respond(headers))

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{sock.getsockname()[1]}/model.pth", requests, sock


def _response(status, body, **headers):
    lines = [f"HTTP/1.1 {status}"] + [f"{k.replace('_', '-')}: {v}" for k, v in headers.items()]
    return ("\r\n".join(lines) + "\r\nConnection: close\r\n\r\n").encode() + body


def test_truncated_body_is_not_installed_and_resumes(tmp_path):
    state = {"calls": 0}

    def respond(headers):
        state["calls"] += 1
        if state["calls"] == 1:  # connection drops after 3000 of 10240 bytes
            return _response("200 OK", BODY[:3000], Content_Length=len(BODY))
        start = int(headers["range"].split("=")[1].rstrip("-"))
        return _response("206 Partial Content", BODY[start:], Content_Length=len(BODY) - start,
                         Content_Range=f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")

    url, requests, sock = _server(respond)
    target = tmp_path / "vsr" / "model.pth"
    try:
        with pytest.raises(ValueError, match="incomplete download"):
            assets.download(url, target)
        assert not target.exists()
        assets.download(url, target)
        assert target.read_bytes() == BODY
        assert requests[1]["range"] == "bytes=3000-"
    finally:
        sock.close()


def test_partial_content_for_another_range_restarts_from_zero(tmp_path):
    def respond(headers):
        if "range" in headers:  # ignores the requested offset
            return _response("206 Partial Content", BODY[5000:], Content_Length=len(BODY) - 5000,
                             Content_Range=f"bytes 5000-{len(BODY) - 1}/{len(BODY)}")
        return _response("200 OK", BODY, Content_Length=len(BODY))

    url, requests, sock = _server(respond)
    target = tmp_path / "model.pth"
    target.with_suffix(".pth.part").write_bytes(BODY[:1000])
    try:
        assets.download(url, target)
        assert target.read_bytes() == BODY
        assert "range" not in requests[-1]
    finally:
        sock.close()


def test_unsatisfiable_range_restarts_instead_of_failing_forever(tmp_path):
    def respond(headers):
        if "range" in headers:
            return _response("416 Range Not Satisfiable", b"", Content_Length=0)
        return _response("200 OK", BODY, Content_Length=len(BODY))

    url, _, sock = _server(respond)
    target = tmp_path / "model.pth"
    target.with_suffix(".pth.part").write_bytes(BODY)  # complete, never renamed
    try:
        assets.download(url, target)
        assert target.read_bytes() == BODY
    finally:
        sock.close()

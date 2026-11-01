import http.client
import json
import threading
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from hackathon.runtime import Runtime
from hackathon.server import main, make_server


def jpeg_frame():
    # Distinct solid halves make preview mirroring observable after JPEG decoding.
    frame = np.zeros((8, 16, 3), dtype=np.uint8)
    frame[:, :8, 2] = 255
    frame[:, 8:, 0] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return frame, encoded.tobytes()


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response, data


def test_browser_runtime_waits_and_starts_workers_without_native_capture():
    runtime = Runtime(browser_camera=True, vocabulary={})
    runtime._capture = lambda: pytest.fail("native capture must not start")
    runtime._hands = lambda: None
    runtime._lips = lambda: None
    try:
        state = runtime.snapshot()
        assert state["browser_camera"] is True
        assert state["camera"] == "waiting"
        assert state["camera_error"] == ""
        runtime.start()
        assert [thread.name for thread in runtime.threads] == ["demo-hands", "demo-lips"]
    finally:
        runtime.close()

    native = Runtime(vocabulary={})
    native._capture = lambda: None
    native._hands = lambda: None
    native._lips = lambda: None
    try:
        native.start()
        assert [thread.name for thread in native.threads] == ["demo-capture", "demo-hands", "demo-lips"]
        assert native.snapshot()["browser_camera"] is False
    finally:
        native.close()


def test_receive_frame_feeds_both_workers_and_mirrors_preview():
    runtime = Runtime(browser_camera=True, vocabulary={})
    frame, _ = jpeg_frame()
    try:
        assert runtime.receive_frame(frame, 100) is True
        assert runtime.hand_frames.get_nowait() == (frame, 100)
        assert runtime.lip_frames.get_nowait() == (frame, 100)
        assert runtime.receive_frame(frame, 100) is False
        assert runtime.receive_frame(frame, 99) is False
        assert runtime.hand_frames.empty() and runtime.lip_frames.empty()
        assert runtime.snapshot()["camera"] == "ready"
        preview = cv2.imdecode(np.frombuffer(runtime.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert preview[0, 0, 0] >= 250
        assert preview[0, -1, 2] >= 250
    finally:
        runtime.close()


def test_rehearsal_and_browser_camera_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Runtime(rehearsal=True, browser_camera=True, vocabulary={})
    with patch("sys.argv", ["hackathon.server", "--rehearsal", "--browser-camera"]):
        with pytest.raises(SystemExit):
            main()


def test_browser_frame_http_contract_and_native_rejection():
    runtime = Runtime(browser_camera=True, vocabulary={})
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        response, body = request(port, "GET", "/api/state")
        state = json.loads(body)
        assert response.status == 200
        assert state["browser_camera"] is True
        assert state["camera"] == "waiting"
        token = state["token"]
        _, body = jpeg_frame()
        headers = {"Content-Type": "image/jpeg", "X-Lipflow-Token": token,
                   "Origin": f"http://127.0.0.1:{port}"}
        response, data = request(port, "POST", "/api/frame", body, headers)
        assert response.status == 200
        assert json.loads(data) == {"ok": True}
        assert runtime.hand_frames.qsize() == runtime.lip_frames.qsize() == 1
        response, preview = request(port, "GET", "/api/frame")
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/jpeg"
        assert preview == runtime.jpeg

        for bad_headers in ({"Content-Type": "image/jpeg"},
                            {**headers, "Origin": "https://example.com"}):
            response, _ = request(port, "POST", "/api/frame", body, bad_headers)
            assert response.status == 403
        response, _ = request(port, "POST", "/api/frame", body,
                              {"Content-Type": "application/json", "X-Lipflow-Token": token})
        assert response.status == 415
        response, _ = request(port, "POST", "/api/frame", b"not a jpeg", headers)
        assert response.status == 400
        response, _ = request(port, "POST", "/api/frame", b"x" * (512 * 1024 + 1), headers)
        assert response.status == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        runtime.close()

    for runtime in (Runtime(vocabulary={}), Runtime(rehearsal=True, vocabulary={})):
        server = make_server(runtime, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _, state_body = request(server.server_port, "GET", "/api/state")
            token = json.loads(state_body)["token"]
            _, body = jpeg_frame()
            response, _ = request(server.server_port, "POST", "/api/frame", body,
                                  {"Content-Type": "image/jpeg", "X-Lipflow-Token": token})
            assert response.status == 400
        finally:
            server.shutdown()
            server.server_close()
            thread.join(1)
            runtime.close()

"""Software contracts for the loopback server and runtime plumbing.

Fakes only: these tests check HTTP framing, error handling, preview encoding,
worker start-up and speech-state plumbing. They say nothing about camera
accuracy, recognition quality or audible speech.
"""
import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from hackathon import server as server_module
from hackathon.profiles import ProfileManager
from hackathon.runtime import ProfileSwitchConflict, Runtime
from hackathon.server import make_server, open_in_browser


@contextmanager
def serving(runtime):
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def connect(server):
    return http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)


def exchange(conn, method, path, body=None, headers=None):
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    return response, response.read()


def post_headers(server, token, content_type="application/json"):
    return {"Content-Type": content_type, "X-Lipflow-Token": token,
            "Origin": f"http://127.0.0.1:{server.server_port}"}


def silent_runtime(**kwargs):
    kwargs.setdefault("rehearsal", True)
    kwargs.setdefault("vocabulary", {})
    runtime = Runtime(**kwargs)
    runtime.speaker.close()
    runtime.speaker = Mock(available=True)
    return runtime


def test_one_keep_alive_connection_serves_many_requests_including_rejections():
    with serving(silent_runtime()) as server:
        conn = connect(server)
        response, body = exchange(conn, "GET", "/api/state")
        assert response.status == 200 and response.version == 11
        assert response.getheader("Connection") is None
        token = json.loads(body)["token"]
        sock = conn.sock
        for _ in range(20):
            response, body = exchange(conn, "GET", "/api/state")
            assert response.status == 200 and json.loads(body)["token"] == token
        # Rejections that never read the body must not corrupt the next request.
        bad = json.dumps({"revision": 0}).encode()
        for method, path, payload, headers, status in [
            ("POST", "/api/action/cancel", bad, {"Content-Type": "application/json"}, 403),
            ("POST", "/api/nope", bad, post_headers(server, token), 404),
            ("POST", "/api/action/cancel", b"x" * 9000, post_headers(server, token), 400),
            ("POST", "/api/profile/select", b"x" * 9000, post_headers(server, token), 400),
            ("GET", "/missing", None, {}, 404),
            ("POST", "/api/action/rehearse", json.dumps({"text": "typed", "label": ""}).encode(),
             post_headers(server, token), 200),
        ]:
            response, body = exchange(conn, method, path, payload, headers)
            assert response.status == status, (path, body)
            assert int(response.getheader("Content-Length")) == len(body)
            assert "Content-Security-Policy" in response.headers
            response, body = exchange(conn, "GET", "/api/state")
            assert response.status == 200
        assert json.loads(body)["decision"]["text"] == "typed"
        assert conn.sock is sock, "the connection must have been reused, not reopened"
        conn.close()


def test_personalize_redirect_has_empty_body_and_keeps_connection():
    with serving(silent_runtime()) as server:
        conn = connect(server)
        response, body = exchange(conn, "GET", "/record")
        sock = conn.sock
        assert response.status == 302 and body == b""
        assert response.getheader("Content-Length") == "0"
        assert response.getheader("Location").startswith("http://127.0.0.1:")
        response, _ = exchange(conn, "GET", "/api/state")
        assert response.status == 200 and conn.sock is sock is not None
        conn.close()


def test_wrong_host_with_body_is_drained():
    with serving(silent_runtime()) as server:
        conn = connect(server)
        response, _ = exchange(conn, "POST", "/api/action/cancel", b"{}" * 100,
                               {"Host": "example.com", "Content-Type": "application/json"})
        assert response.status == 403
        sock = conn.sock
        response, _ = exchange(conn, "GET", "/api/state")
        assert response.status == 200 and conn.sock is sock is not None
        conn.close()


def test_server_close_returns_promptly_with_idle_keep_alive_connections():
    runtime = silent_runtime()
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    connections = [connect(server) for _ in range(5)]
    for conn in connections:
        response, _ = exchange(conn, "GET", "/api/state")
        assert response.status == 200
    started = time.perf_counter()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    runtime.close()
    assert time.perf_counter() - started < 2
    # Old handlers must not keep answering on reused connections after a close.
    for conn in connections:
        with pytest.raises((http.client.HTTPException, OSError)):
            exchange(conn, "GET", "/api/state")
        conn.close()
    deadline = time.time() + 2
    while time.time() < deadline and server._connections:
        time.sleep(0.01)
    assert not server._connections


def test_idle_keep_alive_connection_times_out(monkeypatch):
    monkeypatch.setattr(server_module, "IDLE_TIMEOUT_S", 0.2)
    with serving(silent_runtime()) as server:
        conn = connect(server)
        exchange(conn, "GET", "/api/state")
        sock = conn.sock
        sock.settimeout(2)
        # The server closes an idle connection: the next read sees EOF.
        assert sock.recv(1) == b""
        conn.close()


@pytest.mark.parametrize("route", ["state", "action", "profile", "frame"])
def test_unexpected_exceptions_answer_generic_json_500(route, capfd):
    secret = "secret-internal-detail /private/path"
    runtime = silent_runtime(rehearsal=route != "frame", browser_camera=route == "frame")
    with serving(runtime) as server:
        conn = connect(server)
        _, body = exchange(conn, "GET", "/api/state")
        token = json.loads(body)["token"]
        if route == "state":
            runtime.snapshot = Mock(side_effect=KeyError(secret))
            response, body = exchange(conn, "GET", "/api/state")
        elif route == "action":
            runtime.action = Mock(side_effect=AttributeError(secret))
            response, body = exchange(conn, "POST", "/api/action/cancel", b"{}", post_headers(server, token))
        elif route == "profile":
            runtime.switch_profile = Mock(side_effect=AttributeError(secret))
            response, body = exchange(conn, "POST", "/api/profile/select", b'{"id":"x"}',
                                      post_headers(server, token))
        else:
            runtime.receive_frame = Mock(side_effect=KeyError(secret))
            ok, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
            response, body = exchange(conn, "POST", "/api/frame", jpeg.tobytes(),
                                      post_headers(server, token, "image/jpeg"))
        assert response.status == 500
        assert response.getheader("Content-Type").startswith("application/json")
        payload = json.loads(body)
        assert set(payload) == {"error"} and secret not in payload["error"]
        assert secret in capfd.readouterr().err  # traceback goes to the server log
        if route != "state":
            response, _ = exchange(conn, "GET", "/api/state")
            assert response.status == 200  # connection survives the 500
        conn.close()


def test_expected_error_statuses_are_unchanged(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    profile = manager.ensure_default_profile()
    runtime = silent_runtime(profile=profile, profile_manager=manager)
    with serving(runtime) as server:
        conn = connect(server)
        _, body = exchange(conn, "GET", "/api/state")
        token = json.loads(body)["token"]
        response, body = exchange(conn, "POST", "/api/action/speak", b'{"revision":-1}',
                                  post_headers(server, token))
        assert response.status == 400 and "result changed" in json.loads(body)["error"]
        runtime.switch_profile = Mock(side_effect=ProfileSwitchConflict("busy"))
        response, body = exchange(conn, "POST", "/api/profile/select", b'{"id":"x"}',
                                  post_headers(server, token))
        assert response.status == 409 and json.loads(body) == {"error": "busy"}
        runtime.switch_profile = Mock(side_effect=KeyError("unknown profile"))
        response, _ = exchange(conn, "POST", "/api/profile/select", b'{"id":"x"}',
                               post_headers(server, token))
        assert response.status == 400
        conn.close()


def mirrored_test_frame():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :8] = (255, 0, 0)    # left edge blue
    frame[:, -8:] = (0, 0, 255)   # right edge red
    return frame


def test_browser_frames_are_not_encoded_on_ingest_and_preview_is_cached():
    runtime = silent_runtime(rehearsal=False, browser_camera=True)
    frame = mirrored_test_frame()
    real_imencode = cv2.imencode
    calls = []

    def counting_imencode(*args, **kwargs):
        # Encoding must not hold the shared runtime lock.
        acquired = []

        def probe_lock():
            got = runtime.lock.acquire(timeout=1)
            if got:
                runtime.lock.release()
            acquired.append(got)

        probe = threading.Thread(target=probe_lock)
        probe.start()
        probe.join()
        calls.append(acquired[0])
        return real_imencode(*args, **kwargs)

    try:
        with patch("cv2.imencode", side_effect=counting_imencode):
            assert runtime.receive_frame(frame, 0)
            assert calls == []
            for ts in range(1, 30):
                assert runtime.receive_frame(frame, ts)
            assert calls == []
            first = runtime.preview_jpeg()
            assert runtime.preview_jpeg() is first and runtime.jpeg is first
            assert calls == [True]
            runtime.receive_frame(frame, 31)
            assert runtime.preview_jpeg() != b"" and calls == [True, True]
        preview = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
        assert preview[0, 0, 2] >= 200 and preview[0, -1, 0] >= 200  # mirrored
        runtime.action("camera_session", {"active": True})
        runtime.receive_frame(frame, 40)
        assert runtime.jpeg is not None
        runtime.action("camera_session", {"active": False})
        assert runtime.jpeg is None and runtime.preview_jpeg() is None
    finally:
        runtime.close()


def test_preview_cleared_while_encoding_is_not_returned():
    runtime = silent_runtime(rehearsal=False, browser_camera=True)
    real_imencode = cv2.imencode

    def clearing_imencode(*args, **kwargs):
        runtime.jpeg = None  # e.g. camera stop or profile switch mid-encode
        return real_imencode(*args, **kwargs)

    try:
        runtime.receive_frame(mirrored_test_frame(), 1)
        with patch("cv2.imencode", side_effect=clearing_imencode):
            assert runtime.preview_jpeg() is None
        assert runtime.preview_jpeg() is None
    finally:
        runtime.close()


def test_http_preview_503_then_mirrored_jpeg():
    runtime = silent_runtime(rehearsal=False, browser_camera=True)
    with serving(runtime) as server:
        conn = connect(server)
        response, _ = exchange(conn, "GET", "/api/frame")
        assert response.status == 503
        runtime.receive_frame(mirrored_test_frame(), 1)
        response, body = exchange(conn, "GET", "/api/frame")
        assert response.status == 200 and response.getheader("Content-Type") == "image/jpeg"
        assert body == runtime.jpeg
        conn.close()


def test_profile_switch_clears_preview(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    default = manager.ensure_default_profile()
    manager.create_profile("alice", "Alice")
    runtime = silent_runtime(profile=default, profile_manager=manager)
    try:
        with runtime.lock:
            runtime._set_preview_frame(mirrored_test_frame())
        assert runtime.jpeg is not None
        runtime.switch_profile("alice")
        assert runtime.jpeg is None
    finally:
        runtime.close()


class FakeCapture:
    def __init__(self, runtime, frames):
        self.runtime, self.frames = runtime, list(frames)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        if not self.frames:
            self.runtime.stop.set()
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


def test_server_camera_capture_uses_lazy_preview():
    runtime = silent_runtime(rehearsal=False)
    frame = mirrored_test_frame()
    try:
        with patch("cv2.VideoCapture", return_value=FakeCapture(runtime, [frame] * 5)), \
                patch("cv2.imencode", wraps=cv2.imencode) as imencode:
            runtime._capture(runtime._worker_generation, None)
            assert imencode.call_count == 0
            preview = cv2.imdecode(np.frombuffer(runtime.jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert imencode.call_count == 1
        assert preview[0, 0, 2] >= 200 and runtime.state["camera"] == "ready"
    finally:
        runtime.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_worker_type_error_is_not_rerun_with_default_arguments():
    runtime = silent_runtime(rehearsal=False, browser_camera=True, enable_lips=False)
    calls = []

    def failing_hands(*args):
        calls.append(args)
        raise TypeError("raised inside the worker")

    runtime._hands = failing_hands
    try:
        runtime.start()
        for thread in list(runtime.threads):
            thread.join(timeout=2)
        assert len(calls) == 1 and len(calls[0]) == 2
        assert calls[0][0] == runtime._worker_generation
    finally:
        runtime.close()


class FakeSpeaker:
    available = True

    def __init__(self, speaking=False, backend="espeak-ng"):
        self.speaking, self.backend, self.stops, self.spoken = speaking, backend, 0, []

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stops += 1
        self.speaking = False

    def close(self):
        pass


class SilentSpeakerLike:
    """Shape of scripts/web_check_fixture.py's fake: no stop/speaking/backend."""
    available = True

    def speak(self, text):
        pass

    def close(self):
        pass


def test_snapshot_reports_speech_state_strictly():
    runtime = silent_runtime()
    try:
        state = runtime.snapshot()  # Mock speaker: attributes are truthy Mocks
        assert state["speaking"] is False and state["speech_backend"] is None
        json.dumps(state)
        runtime.speaker = SilentSpeakerLike()
        state = runtime.snapshot()
        assert state["speaking"] is False and state["speech_backend"] is None
        runtime.speaker = FakeSpeaker(speaking=True)
        state = runtime.snapshot()
        assert state["speaking"] is True and state["speech_backend"] == "espeak-ng"
        json.dumps(state)
    finally:
        runtime.close()


def test_stop_speaking_is_idempotent_and_never_changes_the_message():
    runtime = silent_runtime()
    speaker = runtime.speaker = FakeSpeaker(speaking=True)
    try:
        runtime.action("rehearse", {"text": "keep this", "label": ""})
        before = runtime.snapshot()
        runtime.action("stop_speaking", {})
        runtime.action("stop_speaking", {"revision": -5})
        runtime.state["phase"] = "recording"   # allowed in any phase
        runtime.action("stop_speaking", {})
        runtime.state["phase"] = "idle"
        runtime._switching_profile = True      # and while a switch is in progress
        runtime.action("stop_speaking", {})
        runtime._switching_profile = False
        after = runtime.snapshot()
        assert speaker.stops == 4 and after["speaking"] is False
        assert after["revision"] == before["revision"]
        assert after["decision"] == before["decision"]
        assert speaker.spoken == []
        runtime.speaker = SilentSpeakerLike()
        runtime.action("stop_speaking", {})  # a speaker without stop() is harmless
    finally:
        runtime.close()


def test_stop_speaking_http_returns_state_payload():
    runtime = silent_runtime()
    speaker = runtime.speaker = FakeSpeaker(speaking=True)
    with serving(runtime) as server:
        conn = connect(server)
        _, body = exchange(conn, "GET", "/api/state")
        token = json.loads(body)["token"]
        response, body = exchange(conn, "POST", "/api/action/stop_speaking", b"{}", post_headers(server, token))
        state = json.loads(body)
        assert response.status == 200 and state["speaking"] is False
        assert state["token"] == token and "revision" in state and speaker.stops == 1
        conn.close()


def test_new_message_uses_speaking_property():
    runtime = silent_runtime()
    try:
        runtime.speaker = FakeSpeaker(speaking=True)
        with pytest.raises(ValueError):
            runtime.action("new_message", {"revision": runtime.revision})
        runtime.speaker = Mock(available=True)  # Mock.speaking is not True
        runtime.action("new_message", {"revision": runtime.revision})
    finally:
        runtime.close()


def test_open_in_browser_is_best_effort(capsys):
    opener = Mock()
    open_in_browser("http://127.0.0.1:1/", opener).join(2)
    opener.assert_called_once_with("http://127.0.0.1:1/")
    failing = Mock(side_effect=RuntimeError("no display"))
    open_in_browser("http://127.0.0.1:1/", failing).join(2)
    assert "open http://127.0.0.1:1/ manually" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [True, False])
def test_main_open_flag(flag, tmp_path, monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("QUIETBRIDGE_MOTION_MODEL", raising=False)
    real_make_server = server_module.make_server
    servers = []

    def fake_make_server(runtime, port):
        server = real_make_server(runtime, 0)
        assert server.server_address[0] == "127.0.0.1"
        server.serve_forever = Mock(side_effect=KeyboardInterrupt)
        servers.append(server)
        return server

    opened = threading.Event()
    opener = Mock(side_effect=lambda url: opened.set())
    argv = ["hackathon.server", "--rehearsal", "--port", "0"] + (["--open"] if flag else [])
    with patch("sys.argv", argv), patch.object(server_module, "make_server", fake_make_server), \
            patch.object(server_module.webbrowser, "open", opener):
        server_module.main()
        assert opened.wait(2 if flag else 0.3) is flag
    if flag:
        opener.assert_called_once_with(f"http://127.0.0.1:{servers[0].server_port}")
    else:
        opener.assert_not_called()


def test_main_rejects_motion_model_without_vocabulary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUIETBRIDGE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", str(tmp_path / "motion.npz"))
    with patch("sys.argv", ["hackathon.server", "--browser-camera", "--port", "0"]), \
            patch.object(server_module, "make_server") as make:
        with pytest.raises(SystemExit):
            server_module.main()
    make.assert_not_called()
    assert "no vocabulary" in capsys.readouterr().err


def test_switch_to_profile_without_vocabulary_for_explicit_motion_fails_visibly(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    default = manager.ensure_default_profile()
    manager.create_profile("alice", "Alice")
    motion = tmp_path / "cli_motion.npz"
    motion.write_bytes(b"not loaded by this test")
    vocab = {"CLI": {"phrase": "Cli", "matches": ["Cli"], "enabled": True, "verified": False}}
    runtime = silent_runtime(rehearsal=False, browser_camera=True, profile=default,
                             profile_manager=manager, explicit_motion_model=motion,
                             motion_model=motion, vocabulary=vocab)
    try:
        with pytest.raises(ValueError, match="no vocabulary"):
            runtime.switch_profile("alice")
        assert runtime.profile.id == default.id and runtime.vocabulary == vocab
        assert runtime._switching_profile is False
        assert "WATER" not in runtime.vocabulary  # stock meanings never applied
    finally:
        runtime.close()


def test_port_in_use_reports_the_bind_error():
    with serving(silent_runtime()) as server:
        other = silent_runtime()
        try:
            with pytest.raises(OSError) as raised:
                make_server(other, server.server_port)
            assert not isinstance(raised.value, AttributeError)
        finally:
            other.close()


def test_keep_alive_responses_are_not_delayed_by_nagle():
    # ~40 ms per response with Nagle + delayed ACK; a loose bound, not a benchmark.
    with serving(silent_runtime()) as server:
        assert server.RequestHandlerClass.disable_nagle_algorithm is True
        conn = connect(server)
        exchange(conn, "GET", "/api/state")
        started = time.perf_counter()
        for _ in range(30):
            response, _ = exchange(conn, "GET", "/api/state")
            assert response.status == 200
        assert time.perf_counter() - started < 0.6
        conn.close()

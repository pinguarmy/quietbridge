"""HTTP-level tests for hackathon.signs.lab using a fake inference engine.

No camera, no MediaPipe model, no downloads: the server is created with a
stub engine and a temporary data root, and exercised over real loopback HTTP.
"""
from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from hackathon.contracts import SignEvent
from hackathon.signs import lab
from hackathon.signs.lab import FrameResult, HandResult, create_server


class FakeEngine:
    """Deterministic engine: two hands (Left THUMB_UP, Right raw 'Victory')."""

    def __init__(self, hands=None, events=None):
        self._hands = hands
        self._events = events if events is not None else []
        self.timestamps = []
        self.closed = False

    def infer(self, frame_rgb, timestamp_ms: int) -> FrameResult:
        assert frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3
        if self.timestamps:
            assert timestamp_ms > self.timestamps[-1], "timestamps must strictly increase"
        self.timestamps.append(timestamp_ms)
        hands = self._hands
        if hands is None:
            hands = [
                HandResult("Left", "Thumb_Up", 0.91, [[0.1, 0.2, 0.3]] * 21),
                HandResult("Right", "Victory", 0.66, [[0.4, 0.5, 0.6]] * 21),
            ]
        return FrameResult(hands=hands, events=list(self._events))

    def close(self):
        self.closed = True


def _jpeg(width=32, height=24) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def server(tmp_path):
    engine = FakeEngine()
    srv = create_server(engine=engine, port=0, data_root=tmp_path / "seq")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)
    engine.close()


@pytest.fixture()
def conn(server):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    yield connection
    connection.close()


def _request(conn, server, method, path, body=None, headers=None, token=True):
    all_headers = {}
    if token:
        all_headers["X-QuietBridge-Token"] = server.token
    all_headers.update(headers or {})
    conn.request(method, path, body=body, headers=all_headers)
    response = conn.getresponse()
    data = response.read()
    return response, data


def _json(conn, server, method, path, body=None, headers=None, token=True):
    response, data = _request(conn, server, method, path, body, headers, token)
    return response, json.loads(data)


def _post_frame(conn, server):
    return _json(conn, server, "POST", "/api/frame", body=_jpeg(),
                 headers={"Content-Type": "image/jpeg"})


def _start(conn, server, label="DDE", split="train", session="s1"):
    return _json(conn, server, "POST", "/api/start",
                 body=json.dumps({"label": label, "split": split, "session": session}),
                 headers={"Content-Type": "application/json"})


# -- state / security ----------------------------------------------------


def test_state_needs_no_token_and_has_contract_fields(conn, server):
    response, data = _json(conn, server, "GET", "/api/state", token=False)
    assert response.status == 200
    assert data["token"] == server.token
    assert data["recording"] is False
    assert data["saved_count"] == 0
    assert data["last_saved"] == ""
    assert data["model_ready"] is True
    assert "THUMB_UP" in data["gesture_support"]
    assert isinstance(data["note"], str) and data["note"]


def test_post_requires_token(conn, server):
    response, data = _json(conn, server, "POST", "/api/stop", body=b"{}", token=False)
    assert response.status == 403
    assert "error" in data
    response, _ = _json(conn, server, "POST", "/api/stop", body=b"{}",
                        headers={"X-QuietBridge-Token": "wrong"}, token=False)
    assert response.status == 403


def test_host_and_origin_restrictions(conn, server):
    # http.client always sends a valid loopback Host; an explicit foreign one must fail.
    response, _ = _request(conn, server, "GET", "/api/state",
                           headers={"Host": "evil.example.com"}, token=False)
    assert response.status == 403
    response, _ = _json(conn, server, "POST", "/api/stop", body=b"{}",
                        headers={"Origin": "http://evil.example.com"})
    assert response.status == 403
    port = server.server_address[1]
    response, _ = _json(conn, server, "POST", "/api/stop", body=b"{}",
                        headers={"Origin": f"http://127.0.0.1:{port}"})
    assert response.status == 400  # origin accepted; fails only because not recording


def test_frame_rejects_wrong_content_type(conn, server):
    response, data = _json(conn, server, "POST", "/api/frame", body=_jpeg(),
                           headers={"Content-Type": "application/json"})
    assert response.status == 415
    assert "error" in data


def test_frame_rejects_oversize_body(conn, server):
    # The server closes the connection on an over-limit Content-Length; the
    # client may therefore see either the 413 response or a reset mid-write.
    try:
        response, _ = _json(conn, server, "POST", "/api/frame", body=b"x" * (513 * 1024),
                            headers={"Content-Type": "image/jpeg"})
        assert response.status == 413
    except (ConnectionResetError, BrokenPipeError):
        pass
    # The server itself is still healthy afterwards.
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    response, _ = _json(connection, server, "GET", "/api/state", token=False)
    connection.close()
    assert response.status == 200


def test_html_served_with_csp(conn, server, tmp_path, monkeypatch):
    html = tmp_path / "lab.html"
    html.write_text("<html><body>lab</body></html>")
    monkeypatch.setattr(lab, "LAB_HTML", html)
    response, data = _request(conn, server, "GET", "/", token=False)
    assert response.status == 200
    assert b"lab" in data
    csp = response.getheader("Content-Security-Policy")
    assert "unsafe-inline" in csp and "default-src 'self'" in csp


# -- frame inference -----------------------------------------------------


def test_frame_returns_raw_events_and_counts(conn, server):
    response, data = _post_frame(conn, server)
    assert response.status == 200
    assert data["raw"] == [{"label": "Thumb_Up", "score": pytest.approx(0.91)},
                           {"label": "Victory", "score": pytest.approx(0.66)}]
    assert data["hand_count"] == 2
    assert len(data["tracking"]) == 2
    assert data["tracking"][0]["handedness"] == "Left"
    assert data["tracking"][1]["landmarks"] == [[0.4, 0.5, 0.6]] * 21
    assert data["lips"]["status"] == "disabled"
    assert data["events"] == []
    assert data["recording"] is False
    assert data["recorded_frames"] == 0
    assert data["limit_reached"] is False
    assert data["process_ms"] >= 0


def test_frame_events_serialized_as_signevent_dicts(tmp_path):
    event = SignEvent(label="THUMB_UP", score=0.9, start_ms=1, end_ms=2, is_final=True)
    engine = FakeEngine(hands=[], events=[event])
    srv = create_server(engine=engine, port=0, data_root=tmp_path / "seq")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        response, data = _json(connection, srv, "POST", "/api/frame", body=_jpeg(),
                               headers={"Content-Type": "image/jpeg"})
        connection.close()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
    assert response.status == 200
    assert data["events"] == [{"label": "THUMB_UP", "score": 0.9,
                               "start_ms": 1, "end_ms": 2, "is_final": True}]


def test_mouth_routes_dispatch_and_refuse_overlapping_gestures(conn, server):
    class FakeMouth:
        def __init__(self):
            self.recording = False
            self.args = None
        def start(self, text, split, session):
            self.args = (text, split, session)
            self.recording = True
        def snapshot(self):
            return {"recording": self.recording, "frames": 0, "mouth_box": None}
        def stop(self):
            self.recording = False
        def discard(self):
            self.recording = False
        def close(self):
            pass
    server.mouth.close()
    server.mouth = FakeMouth()
    response, data = _json(conn, server, "POST", "/api/mouth/start",
        body=json.dumps({"text": "Hi, welcome to", "split": "train", "session": "mouth-v1"}),
        headers={"Content-Type": "application/json"})
    assert response.status == 200 and data["mouth"]["recording"] is True
    assert server.mouth.args == ("Hi, welcome to", "train", "mouth-v1")
    response, _ = _start(conn, server)
    assert response.status == 409
    response, data = _json(conn, server, "POST", "/api/mouth/stop", body=b"{}")
    assert response.status == 200 and data["mouth"]["recording"] is False
    response, _ = _start(conn, server)
    assert response.status == 200
    response, _ = _json(conn, server, "POST", "/api/mouth/start",
        body=json.dumps({"text": "Hi", "split": "train", "session": "mouth-v1"}))
    assert response.status == 400


# -- recording flow ------------------------------------------------------


def test_record_stop_saves_npz_with_both_hand_slots(conn, server, tmp_path):
    response, data = _start(conn, server)
    assert response.status == 200 and data["recording"] is True

    for _ in range(5):
        _, frame = _post_frame(conn, server)
    assert frame["recording"] is True
    assert frame["recorded_frames"] == 5

    response, data = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert response.status == 200
    assert data["recorded_frames"] == 5
    saved = Path(data["saved"])
    assert saved.parent == tmp_path / "seq" / "train"
    assert saved.name.startswith("s1_") and saved.suffix == ".npz"

    with np.load(saved, allow_pickle=False) as archive:
        assert archive["timestamps_ms"].shape == (5,)
        assert archive["timestamps_ms"].dtype == np.int64
        assert archive["landmarks"].shape == (5, 2, 21, 3)
        assert archive["landmarks"].dtype == np.float32
        assert archive["hand_present"].shape == (5, 2)
        assert archive["hand_present"].dtype == bool
        assert archive["hand_present"].all()
        handedness = archive["handedness"]
        assert handedness.shape == (5, 2)
        assert set(handedness[0].tolist()) == {"Left", "Right"}
        # Left hand in slot 0, Right in slot 1; no wrist normalization applied.
        assert archive["landmarks"][0, 0, 0, 0] == pytest.approx(0.1)
        assert archive["landmarks"][0, 1, 0, 0] == pytest.approx(0.4)
        assert str(archive["label"]) == "DDE"
        assert str(archive["split"]) == "train"
        assert str(archive["session_id"]) == "s1"
        assert len(str(archive["trial_id"])) == 36
        # Strictly increasing timestamps were used for inference and storage.
        ts = archive["timestamps_ms"]
        assert (np.diff(ts) > 0).all()

    _, state = _json(conn, server, "GET", "/api/state", token=False)
    assert state["recording"] is False
    assert state["saved_count"] == 1
    assert state["last_saved"] == str(saved)


def test_empty_hand_frames_still_saved_with_mask(conn, server, tmp_path):
    server.engine._hands = []  # no hands detected
    _start(conn, server)
    for _ in range(3):
        _post_frame(conn, server)
    response, data = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert response.status == 200
    with np.load(data["saved"], allow_pickle=False) as archive:
        assert archive["landmarks"].shape == (3, 2, 21, 3)
        assert not archive["hand_present"].any()
        assert np.isnan(archive["landmarks"]).all()
        assert (archive["handedness"] == "").all()


def test_start_rejects_reentry_and_bad_input(conn, server):
    _start(conn, server)
    response, data = _start(conn, server)
    assert response.status == 409
    _json(conn, server, "POST", "/api/discard", body=b"{}")

    response, _ = _start(conn, server, label="dde")  # must be safe UPPER_SNAKE
    assert response.status == 400
    response, _ = _start(conn, server, label="DDE;rm")
    assert response.status == 400
    response, _ = _start(conn, server, split="valid")
    assert response.status == 400
    response, _ = _start(conn, server, session="../evil")
    assert response.status == 400


def test_stop_without_frames_errors_and_saves_nothing(conn, server, tmp_path):
    _start(conn, server)
    response, data = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert response.status == 400
    assert "no frames" in data["error"]
    assert not list(tmp_path.rglob("*.npz"))
    _, state = _json(conn, server, "GET", "/api/state", token=False)
    assert state["recording"] is True  # still recording; user may retry or discard


def test_stop_without_recording_errors(conn, server):
    response, _ = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert response.status == 400


def test_discard_clears_buffer_without_disk_write(conn, server, tmp_path):
    _start(conn, server)
    for _ in range(3):
        _post_frame(conn, server)
    response, data = _json(conn, server, "POST", "/api/discard", body=b"{}")
    assert response.status == 200 and data["discarded"] is True
    assert not list(tmp_path.rglob("*.npz"))
    # Frames keep working for diagnostics after a discard.
    _, frame = _post_frame(conn, server)
    assert frame["recording"] is False and frame["hand_count"] == 2


def test_frame_limit_stops_appending_but_stop_still_saves(tmp_path):
    engine = FakeEngine()
    srv = create_server(engine=engine, port=0, data_root=tmp_path / "seq", max_frames=3)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        _start(connection, srv)
        last = None
        for _ in range(5):
            _, last = _post_frame(connection, srv)
        assert last["recorded_frames"] == 3
        assert last["limit_reached"] is True
        assert last["recording"] is True  # not auto-saved, not auto-stopped
        response, data = _json(connection, srv, "POST", "/api/stop", body=b"{}")
        connection.close()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
    assert response.status == 200
    with np.load(data["saved"], allow_pickle=False) as archive:
        assert archive["timestamps_ms"].shape == (3,)


def test_session_split_leakage_rejected(conn, server, tmp_path):
    _start(conn, server, split="train", session="s9")
    _post_frame(conn, server)
    _json(conn, server, "POST", "/api/stop", body=b"{}")
    response, data = _start(conn, server, split="test", session="s9")
    assert response.status == 400
    assert "split" in data["error"]


def test_only_npz_files_on_disk_no_images(conn, server, tmp_path):
    _start(conn, server)
    _post_frame(conn, server)
    _json(conn, server, "POST", "/api/stop", body=b"{}")
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and all(p.suffix == ".npz" for p in files)

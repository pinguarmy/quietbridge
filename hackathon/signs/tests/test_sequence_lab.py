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


def test_frame_rejects_oversize_length_before_body(conn, server):
    # Header-only request checks the actual 413 contract without OS-dependent
    # mid-upload resets masking the response.
    conn.putrequest("POST", "/api/frame")
    conn.putheader("X-QuietBridge-Token", server.token)
    conn.putheader("Content-Type", "image/jpeg")
    conn.putheader("Content-Length", str(513 * 1024))
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 413
    assert "too large" in json.loads(response.read())["error"]


def test_frame_rejects_oversize_body(conn, server):
    # The server closes the connection on an over-limit Content-Length; the
    # client may therefore see either the 413 response or a reset mid-write.
    try:
        response, _ = _json(conn, server, "POST", "/api/frame", body=b"x" * (513 * 1024),
                            headers={"Content-Type": "image/jpeg"})
        assert response.status == 413
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        pass
    # The server itself is still healthy afterwards.
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
    response, _ = _json(connection, server, "GET", "/api/state", token=False)
    connection.close()
    assert response.status == 200


@pytest.mark.parametrize("page_path", ["/", "/?return_to=http%3A%2F%2F127.0.0.1%3A8875%2F"])
def test_html_served_with_csp(conn, server, tmp_path, monkeypatch, page_path):
    html = tmp_path / "lab.html"
    html.write_text("<html><body>lab</body></html>")
    monkeypatch.setattr(lab, "LAB_HTML", html)
    response, data = _request(conn, server, "GET", page_path, token=False)
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
            self.root = server.mouth.root
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


def test_validation_split_saved_and_prevents_leakage_with_train_and_test(conn, server, tmp_path):
    # Record trial in validation
    response, data = _start(conn, server, label="HELLO", split="validation", session="val_sess")
    assert response.status == 200
    _post_frame(conn, server)
    response, stop_data = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert response.status == 200
    saved = Path(stop_data["saved"])
    assert saved.parent == tmp_path / "seq" / "validation"
    with np.load(saved, allow_pickle=False) as archive:
        assert str(archive["split"]) == "validation"
        assert str(archive["session_id"]) == "val_sess"

    # Attempting same session in train must fail
    res, err = _start(conn, server, label="HELLO", split="train", session="val_sess")
    assert res.status == 400
    assert "split leakage" in err["error"]

    # Attempting same session in test must fail
    res, err = _start(conn, server, label="HELLO", split="test", session="val_sess")
    assert res.status == 400
    assert "split leakage" in err["error"]


def test_profile_endpoints_crud_and_isolation(conn, server, tmp_path):
    # GET /api/profiles
    res, profs = _json(conn, server, "GET", "/api/profiles", token=False)
    assert res.status == 200
    assert "active_profile" in profs
    assert any(p["id"] == "default" for p in profs["profiles"])

    # POST /api/profile/create
    custom_gestures = [
        {"label": "CUSTOM_SIGN", "display_name": "Custom Sign", "concept": "Custom Idea", "description": "Move right"},
        {"label": "REST", "display_name": "Rest", "concept": "Rest", "description": "Relax"},
    ]
    custom_prompts = ["Custom lip phrase 1", "Custom lip phrase 2"]
    res, created = _json(conn, server, "POST", "/api/profile/create",
                         body=json.dumps({
                             "id": "alice",
                             "name": "Alice Profile",
                             "gestures": custom_gestures,
                             "lip_prompts": custom_prompts,
                         }))
    assert res.status == 200
    assert created["id"] == "alice"
    assert created["name"] == "Alice Profile"
    assert len(created["gestures"]) == 2
    assert created["lip_prompts"] == custom_prompts

    # Record gesture in alice profile
    _start(conn, server, label="CUSTOM_SIGN", split="train", session="alice_s1")
    _post_frame(conn, server)
    res, stop_data = _json(conn, server, "POST", "/api/stop", body=b"{}")
    assert res.status == 200
    alice_saved = Path(stop_data["saved"])
    assert "alice" in str(alice_saved)

    # GET /api/profile
    res, current = _json(conn, server, "GET", "/api/profile", token=False)
    assert res.status == 200
    assert current["id"] == "alice"
    assert current["stats"]["sequences"]["train"] == 1

    # POST /api/profile/update
    res, updated = _json(conn, server, "POST", "/api/profile/update",
                         body=json.dumps({
                             "id": "alice",
                             "name": "Alice Renamed",
                         }))
    assert res.status == 200
    assert updated["name"] == "Alice Renamed"

    # POST /api/profile/select
    res, selected = _json(conn, server, "POST", "/api/profile/select",
                          body=json.dumps({"id": "default"}))
    assert res.status == 200
    assert selected["id"] == "default"


def test_state_includes_profile_metadata(conn, server):
    res, state = _json(conn, server, "GET", "/api/state", token=False)
    assert res.status == 200
    assert "active_profile" in state
    assert "profile" in state
    assert state["profile"]["id"] == state["active_profile"]
    assert "gestures" in state["profile"]
    assert "lip_prompts" in state["profile"]


# -- paired recording flow -----------------------------------------------

from hackathon.signs.tests.test_mouth_capture import FakeTracker


def _paired_start(conn, server, profile="default", label="HELLO",
                  text="Hello, how are you?", split="train", session="pair_s1"):
    return _json(conn, server, "POST", "/api/paired/start",
                 body=json.dumps({
                     "profile": profile,
                     "label": label,
                     "text": text,
                     "split": split,
                     "session": session,
                 }),
                 headers={"Content-Type": "application/json"})


def test_paired_validation(conn, server):
    # Missing / empty profile
    res, err = _paired_start(conn, server, profile="")
    assert res.status == 400
    assert "profile" in err["error"]

    # Invalid profile
    res, err = _paired_start(conn, server, profile="non_existent_profile_xyz")
    assert res.status == 400

    # Invalid split
    res, err = _paired_start(conn, server, split="invalid_split")
    assert res.status == 400

    # Invalid session
    res, err = _paired_start(conn, server, session="")
    assert res.status == 400

    # Text not in lip prompts
    res, err = _paired_start(conn, server, text="Unknown prompt not configured")
    assert res.status == 400
    assert "lip prompts" in err["error"]

    # Label not in gestures and not REST
    res, err = _paired_start(conn, server, label="NOT_A_GESTURE")
    assert res.status == 400
    assert "gestures or REST" in err["error"]

    # REST label is explicitly allowed
    res, data = _paired_start(conn, server, label="REST", session="s_rest")
    assert res.status == 200
    assert data["paired"] is True
    assert data["label"] == "REST"
    _json(conn, server, "POST", "/api/paired/discard")

    # While paired recording is active, starting another should be 409
    res, data = _paired_start(conn, server, label="HELLO", session="s_conflict")
    assert res.status == 200
    res2, err2 = _paired_start(conn, server, label="HELLO", session="s_conflict2")
    assert res2.status == 409
    # Standalone start should also be 409
    res3, _ = _start(conn, server, session="s_conflict3")
    assert res3.status == 409
    _json(conn, server, "POST", "/api/paired/discard")


def test_paired_recording_contract(conn, server, monkeypatch):
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for",
                        lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))

    res, start_data = _paired_start(conn, server, profile="default", label="HELLO",
                                    text="Hello, how are you?", split="train", session="pair_contract_s1")
    assert res.status == 200
    assert start_data["recording"] is True
    assert start_data["paired"] is True
    pair_id = start_data["pair_id"]
    assert pair_id

    # Post 3 frames
    for _ in range(3):
        res_f, frame = _post_frame(conn, server)
        assert res_f.status == 200
        assert frame["paired"] is True
        assert frame["recording"] is True

    # Stop paired trial
    res, stop_data = _json(conn, server, "POST", "/api/paired/stop", body=b"{}")
    assert res.status == 200
    assert stop_data["pair_id"] == pair_id
    assert stop_data["saved_count"] == 1
    seq_path = Path(stop_data["saved"]["sequence"])
    mouth_path = Path(stop_data["saved"]["mouth"])
    assert seq_path.is_file()
    assert mouth_path.is_file()

    # Check Sequence NPZ contract
    with np.load(seq_path, allow_pickle=False) as seq_z:
        assert str(seq_z["pair_id"]) == pair_id
        assert str(seq_z["profile_id"]) == "default"
        assert str(seq_z["session_id"]) == "pair_contract_s1"
        assert str(seq_z["split"]) == "train"
        assert str(seq_z["label"]) == "HELLO"
        assert str(seq_z["expected_text"]) == "Hello, how are you?"
        seq_trial_id = str(seq_z["trial_id"])
        assert len(seq_trial_id) > 0
        assert seq_z["landmarks"].shape == (3, 2, 21, 3)
        assert seq_z["timestamps_ms"].shape == (3,)

    # Check Mouth NPZ contract
    with np.load(mouth_path, allow_pickle=False) as mouth_z:
        assert str(mouth_z["pair_id"]) == pair_id
        assert str(mouth_z["profile_id"]) == "default"
        assert str(mouth_z["session_id"]) == "pair_contract_s1"
        assert str(mouth_z["split"]) == "train"
        assert str(mouth_z["gesture_label"]) == "HELLO"
        assert str(mouth_z["text"]) == "Hello, how are you?"
        mouth_trial_id = str(mouth_z["trial_id"])
        assert len(mouth_trial_id) > 0
        assert mouth_trial_id != seq_trial_id  # distinct trial IDs
        assert mouth_z["rois"].shape == (20, 96, 96)
        assert mouth_z["timestamps_ms"].shape == (3,)
        assert mouth_z["mouth_open"].shape == mouth_z["timestamps_ms"].shape
        assert mouth_z["face_present"].shape == mouth_z["timestamps_ms"].shape
        assert mouth_z["mouth_open"].dtype == np.float32
        assert mouth_z["face_present"].dtype == bool

    # Cross-module contract: the fusion evaluator consumes the actual API output.
    from hackathon.evaluate_fusion import load_pairs
    pairs = load_pairs(server.profile_manager.get_profile("default"), "train")
    assert len(pairs) == 1
    assert pairs[0][0] == pair_id
    assert pairs[0][1]["trial_id"].item() != pairs[0][2]["trial_id"].item()
    assert len(pairs[0][2]["rois"]) == 20
    assert len(pairs[0][2]["timestamps_ms"]) == 3

    # Second trial with same session never overwrites
    res2, start_data2 = _paired_start(conn, server, profile="default", label="WATER",
                                      text="I would like some water.", split="train", session="pair_contract_s1")
    assert res2.status == 200
    pair_id2 = start_data2["pair_id"]
    assert pair_id2 != pair_id
    _post_frame(conn, server)
    res_stop2, stop_data2 = _json(conn, server, "POST", "/api/paired/stop", body=b"{}")
    assert res_stop2.status == 200
    seq_path2 = Path(stop_data2["saved"]["sequence"])
    mouth_path2 = Path(stop_data2["saved"]["mouth"])
    assert seq_path2 != seq_path
    assert mouth_path2 != mouth_path
    assert seq_path.is_file() and seq_path2.is_file()
    assert mouth_path.is_file() and mouth_path2.is_file()


def test_paired_discard(conn, server, monkeypatch):
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)

    res, _ = _paired_start(conn, server, session="s_discard_test")
    assert res.status == 200
    _post_frame(conn, server)
    res_d, disc = _json(conn, server, "POST", "/api/paired/discard", body=b"{}")
    assert res_d.status == 200
    assert disc["discarded"] is True
    assert disc["recording"] is False
    assert disc["paired"] is False

    res_st, st = _json(conn, server, "GET", "/api/state", token=False)
    assert st["recording"] is False
    assert st["paired"] is False


def test_paired_session_split_isolation(conn, server, monkeypatch):
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for",
                        lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))

    # Save a paired trial in train
    _paired_start(conn, server, split="train", session="sess_leak_guard")
    _post_frame(conn, server)
    _json(conn, server, "POST", "/api/paired/stop", body=b"{}")

    # Reject paired start in validation with same session
    res, err = _paired_start(conn, server, split="validation", session="sess_leak_guard")
    assert res.status == 400
    assert "split leakage" in err["error"]

    # Reject paired start in test with same session
    res, err = _paired_start(conn, server, split="test", session="sess_leak_guard")
    assert res.status == 400
    assert "split leakage" in err["error"]


@pytest.mark.parametrize('explicit_profile', [False, True])
def test_standalone_inputs_share_profile_session_split(conn, server, monkeypatch, explicit_profile):
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr('lipflow.dictation.clip_problem', lambda rec: None)
    monkeypatch.setattr('lipflow.dictation.rois_for', lambda rec: np.zeros((20,96,96), dtype=np.uint8))
    profile = server.profile_manager.get_active_profile()
    extra = {'profile':profile.id} if explicit_profile else {}

    def start(path, split, session):
        body = {'label':'HELLO', 'text':'hello', 'split':split, 'session':session, **extra}
        return _json(conn, server, 'POST', path, body=json.dumps(body),
                     headers={'Content-Type':'application/json'})

    response, _ = start('/api/start', 'train', 'gesture_session')
    assert response.status == 200
    assert _post_frame(conn, server)[0].status == 200
    assert _json(conn, server, 'POST', '/api/stop', body='{}')[0].status == 200
    response, error = start('/api/mouth/start', 'test', 'gesture_session')
    assert response.status == 400 and 'split leakage' in error['error']
    assert server.mouth.snapshot()['recording'] is False
    # Reusing a session for another input in the SAME split remains valid.
    assert start('/api/mouth/start', 'train', 'gesture_session')[0].status == 200
    _json(conn, server, 'POST', '/api/mouth/discard', body='{}')

    assert start('/api/mouth/start', 'validation', 'mouth_session')[0].status == 200
    assert _post_frame(conn, server)[0].status == 200
    stopped = _json(conn, server, 'POST', '/api/mouth/stop', body='{}')
    assert stopped[0].status == 200 and stopped[1]['mouth']['saved_count'] == 1
    response, error = start('/api/start', 'train', 'mouth_session')
    assert response.status == 400 and 'split leakage' in error['error']
    assert server.recording is None
    assert start('/api/start', 'validation', 'mouth_session')[0].status == 200
    _json(conn, server, 'POST', '/api/discard', body='{}')


def test_paired_stop_failure_preserves_diagnostics(conn, server, monkeypatch):
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)

    # 1. Zero frames: sequence has no frames captured
    _paired_start(conn, server, session="s_zero_frames")
    res, err = _json(conn, server, "POST", "/api/paired/stop", body=b"{}")
    assert res.status == 400
    assert "no frames captured" in err["error"]
    _json(conn, server, "POST", "/api/paired/discard")

    # 2. Mouth failure: clip_problem returns problem
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: ["Too short", "need more frames"])
    _paired_start(conn, server, session="s_mouth_fail")
    _post_frame(conn, server)
    res, err = _json(conn, server, "POST", "/api/paired/stop", body=b"{}")
    assert res.status == 400
    assert "mouth capture failed" in err["error"]
    assert "Too short" in err["error"]
    _json(conn, server, "POST", "/api/paired/discard")


def test_lab_html_contains_paired_ui(conn, server):
    # Verify the actual lab.html file served at GET / (without monkeypatching LAB_HTML)
    # has the paired tab, paired box, selectors, and clear collection disclaimer
    from hackathon.signs import lab as lab_module
    raw_html = lab_module.LAB_HTML.read_text(encoding="utf-8")
    assert 'id="pairedTab"' in raw_html
    assert 'id="pairedBox"' in raw_html
    assert 'id="pairedGestureSelect"' in raw_html
    assert 'id="pairedPromptSelect"' in raw_html
    assert "Paired data collection only" in raw_html or "no recognition" in raw_html.lower()
    assert "[hidden]{display:none!important}" in raw_html


def test_shutdown_waits_for_active_frame_before_closing_resources(tmp_path):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    close_done = threading.Event()
    closes = []

    class GatedEngine(FakeEngine):
        def infer(self, frame_rgb, timestamp_ms):
            entered.set()
            assert release.wait(3)
            result = super().infer(frame_rgb, timestamp_ms)
            finished.set()
            return result

    engine = GatedEngine()
    srv = create_server(engine=engine, port=0, data_root=tmp_path / 'seq')
    real_close = srv.mouth.close
    srv.mouth.close = lambda: (closes.append(finished.is_set()), real_close())
    serving = threading.Thread(target=srv.serve_forever, daemon=True)
    serving.start()
    responses = []

    def send_frame():
        conn = http.client.HTTPConnection('127.0.0.1', srv.server_port, timeout=4)
        try:
            response, _ = _request(conn, srv, 'POST', '/api/frame', body=_jpeg(),
                                   headers={'Content-Type':'image/jpeg'})
            responses.append(response.status)
        finally:
            conn.close()

    frame_thread = threading.Thread(target=send_frame)
    frame_thread.start()
    assert entered.wait(2)
    srv.shutdown()

    def close_server():
        srv.server_close()
        engine.close()
        close_done.set()

    closing = threading.Thread(target=close_server)
    closing.start()
    try:
        assert not close_done.wait(0.05), 'resources closed before active frame completed'
        assert not engine.closed
        assert closes == []
        # An existing HTTP/1.1 connection must not start a new recording while
        # shutdown is waiting for the active frame to leave the resource lock.
        class RejectedRequest:
            server = srv
            headers = {'Content-Length': '0'}
            def _check_host(self): return True
            def _drain_body(self): pass
            def _error(self, status, message): self.status = status
        request = RejectedRequest()
        assert not lab._Handler._check_post(request)
        assert request.status == 503
    finally:
        release.set()
        closing.join(timeout=4)
        frame_thread.join(timeout=4)
        serving.join(timeout=4)
    assert close_done.is_set()
    assert closes == [True]
    assert engine.closed
    assert responses == [200]

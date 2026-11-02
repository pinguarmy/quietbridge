"""End-to-end integration tests for per-user profile onboarding and customization.

Verifies:
1. Safe profile creation, selection, and customization (custom gestures & lip prompts).
2. Complete guided workflow across stages: Profile Setup -> Training -> Validation -> Final Test.
3. Strict session isolation (session_id cannot cross train, validation, or test).
4. Directory isolation for sequences and mouth clips per profile.
5. Non-destruction of existing files.
"""
from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from hackathon.profiles import ProfileManager
from hackathon.signs.lab import FrameResult, HandResult, create_server


class FakeEngine:
    def __init__(self):
        self.closed = False

    def infer(self, frame_rgb, timestamp_ms: int) -> FrameResult:
        hands = [
            HandResult("Left", "Thumb_Up", 0.95, [[0.1, 0.2, 0.3]] * 21),
            HandResult("Right", "Victory", 0.88, [[0.4, 0.5, 0.6]] * 21),
        ]
        return FrameResult(hands=hands, events=[])

    def close(self):
        self.closed = True


class FakeMouth:
    """Mock MouthCapture tracker for testing lab API dispatch."""

    def __init__(self, data_root=None):
        self.root = Path(data_root) if data_root else Path("/tmp/mouths")
        self.recording = False
        self.frames = 0
        self.args = None
        self.saved_count = 0
        self.last_saved = None

    def set_root(self, root):
        self.root = Path(root)

    def preview(self):
        return self.snapshot()

    def start(self, text, split, session):
        self.args = (text, split, session)
        self.recording = True
        self.frames = 0

    def feed(self, frame, timestamp_ms):
        if self.recording:
            self.frames += 1
        return [0.2, 0.3, 0.2, 0.3]

    def stop(self):
        self.recording = False
        text, split, session = self.args
        out_dir = self.root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{session}_trial1.npz"
        path.write_bytes(b"dummy_npz")
        self.last_saved = str(path)
        self.saved_count += 1
        return self.snapshot()

    def discard(self):
        self.recording = False
        return self.snapshot()

    def snapshot(self):
        return {
            "recording": self.recording,
            "frames": self.frames,
            "mouth_box": [0.2, 0.3, 0.2, 0.3] if self.recording else None,
            "face": True,
            "quality": 1.0,
            "last_saved": self.last_saved,
            "saved_count": self.saved_count,
        }

    def close(self):
        pass


def _jpeg(width=32, height=24) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def lab_env(tmp_path):
    engine = FakeEngine()
    pm = ProfileManager(root_dir=tmp_path / "qb_root")
    srv = create_server(engine=engine, port=0, profile_manager=pm)
    srv.mouth.close()
    srv.mouth = FakeMouth(data_root=pm.get_active_profile().mouths_dir)

    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        yield srv, conn, pm, tmp_path / "qb_root"
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
        engine.close()


def _request_json(conn, server, method, path, body=None, token=True):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-QuietBridge-Token"] = server.token
    raw_body = json.dumps(body).encode("utf-8") if isinstance(body, dict) else (body or b"")
    conn.request(method, path, body=raw_body, headers=headers)
    res = conn.getresponse()
    return res, json.loads(res.read())


def _post_frame(conn, server):
    headers = {
        "Content-Type": "image/jpeg",
        "X-QuietBridge-Token": server.token,
    }
    conn.request("POST", "/api/frame", body=_jpeg(), headers=headers)
    res = conn.getresponse()
    return res, json.loads(res.read())


def test_full_guided_workflow_across_stages(lab_env):
    srv, conn, pm, root_dir = lab_env

    # 1. Profile Setup & Customization
    custom_gestures = [
        {
            "label": "CUSTOM_HI",
            "display_name": "Wave Hi",
            "concept": "Hello and welcome",
            "description": "Raise hand and wave once gently",
            "hint": "Hold still 0.5s, wave, pause 0.5s",
            "emoji": "👋",
        },
        {
            "label": "REST",
            "display_name": "Rest",
            "concept": "Relaxed negative class",
            "description": "Relax hands down",
            "hint": "Ordinary relaxed state",
            "emoji": "😌",
        },
    ]
    custom_prompts = [
        "Welcome to QuietBridge live demo.",
        "Testing speech customization.",
    ]

    res, created = _request_json(conn, srv, "POST", "/api/profile/create", {
        "id": "sarah",
        "name": "Sarah Connor",
        "gestures": custom_gestures,
        "lip_prompts": custom_prompts,
    })
    assert res.status == 200
    assert created["id"] == "sarah"
    assert created["name"] == "Sarah Connor"
    assert len(created["gestures"]) == 2
    assert created["lip_prompts"] == custom_prompts

    # 2. Stage 2: Training Capture (split = "train")
    res, start_data = _request_json(conn, srv, "POST", "/api/start", {
        "label": "CUSTOM_HI",
        "split": "train",
        "session": "sarah_sess_train",
        "profile": "sarah",
    })
    assert res.status == 200 and start_data["recording"] is True
    for _ in range(3):
        _post_frame(conn, srv)
    res, stop_data = _request_json(conn, srv, "POST", "/api/stop", {})
    assert res.status == 200
    saved_train = Path(stop_data["saved"])
    assert "sarah" in str(saved_train)
    assert saved_train.parent == root_dir / "profiles" / "sarah" / "sequences" / "train"

    # Mouth capture in training
    res, _ = _request_json(conn, srv, "POST", "/api/mouth/start", {
        "text": custom_prompts[0],
        "split": "train",
        "session": "sarah_mouth_train",
        "profile": "sarah",
    })
    assert res.status == 200
    for _ in range(2):
        _post_frame(conn, srv)
    res, _ = _request_json(conn, srv, "POST", "/api/mouth/stop", {})
    assert res.status == 200

    # 3. Stage 3: Independent Validation (split = "validation")
    # Rule: session_id must NOT cross train and validation!
    res, err = _request_json(conn, srv, "POST", "/api/start", {
        "label": "CUSTOM_HI",
        "split": "validation",
        "session": "sarah_sess_train",  # Reusing train session id MUST be rejected
        "profile": "sarah",
    })
    assert res.status == 400
    assert "split leakage" in err["error"]

    # Fresh session ID for validation succeeds
    res, start_val = _request_json(conn, srv, "POST", "/api/start", {
        "label": "CUSTOM_HI",
        "split": "validation",
        "session": "sarah_sess_val_1",
        "profile": "sarah",
    })
    assert res.status == 200 and start_val["split"] == "validation"
    _post_frame(conn, srv)
    res, stop_val = _request_json(conn, srv, "POST", "/api/stop", {})
    assert res.status == 200
    saved_val = Path(stop_val["saved"])
    assert saved_val.parent == root_dir / "profiles" / "sarah" / "sequences" / "validation"
    with np.load(saved_val, allow_pickle=False) as z:
        assert str(z["split"]) == "validation"
        assert str(z["session_id"]) == "sarah_sess_val_1"

    # 4. Stage 4: Independent Final Test (split = "test")
    # Rule: session_id must NOT cross validation and test!
    res, err = _request_json(conn, srv, "POST", "/api/start", {
        "label": "CUSTOM_HI",
        "split": "test",
        "session": "sarah_sess_val_1",  # Reusing validation session id MUST be rejected
        "profile": "sarah",
    })
    assert res.status == 400
    assert "split leakage" in err["error"]

    # Fresh session ID for final test succeeds
    res, start_test = _request_json(conn, srv, "POST", "/api/start", {
        "label": "CUSTOM_HI",
        "split": "test",
        "session": "sarah_sess_test_1",
        "profile": "sarah",
    })
    assert res.status == 200 and start_test["split"] == "test"
    _post_frame(conn, srv)
    res, stop_test = _request_json(conn, srv, "POST", "/api/stop", {})
    assert res.status == 200
    saved_test = Path(stop_test["saved"])
    assert saved_test.parent == root_dir / "profiles" / "sarah" / "sequences" / "test"
    with np.load(saved_test, allow_pickle=False) as z:
        assert str(z["split"]) == "test"
        assert str(z["session_id"]) == "sarah_sess_test_1"

    # Check stats in profile metadata
    res, prof_data = _request_json(conn, srv, "GET", "/api/profile", token=False)
    assert res.status == 200
    stats = prof_data["stats"]["sequences"]
    assert stats["train"] == 1
    assert stats["validation"] == 1
    assert stats["test"] == 1


def test_profiles_isolated_from_each_other(lab_env):
    srv, conn, pm, root_dir = lab_env

    # Create Alice
    _request_json(conn, srv, "POST", "/api/profile/create", {
        "id": "alice",
        "name": "Alice",
        "gestures": [
            {"label": "ALICE_SIGN", "display_name": "A", "concept": "A", "description": "A"},
            {"label": "REST", "display_name": "R", "concept": "R", "description": "R"},
        ],
    })
    # Create Bob
    _request_json(conn, srv, "POST", "/api/profile/create", {
        "id": "bob",
        "name": "Bob",
        "gestures": [
            {"label": "BOB_SIGN", "display_name": "B", "concept": "B", "description": "B"},
            {"label": "REST", "display_name": "R", "concept": "R", "description": "R"},
        ],
    })

    # Select Alice and record
    _request_json(conn, srv, "POST", "/api/profile/select", {"id": "alice"})
    _request_json(conn, srv, "POST", "/api/start", {
        "label": "ALICE_SIGN",
        "split": "train",
        "session": "alice_s",
    })
    _post_frame(conn, srv)
    _request_json(conn, srv, "POST", "/api/stop", {})

    alice_dir = root_dir / "profiles" / "alice" / "sequences" / "train"
    bob_dir = root_dir / "profiles" / "bob" / "sequences" / "train"

    assert len(list(alice_dir.glob("*.npz"))) == 1
    assert not list(bob_dir.glob("*.npz"))

"""Recorder flow for capture-quality review: the person decides, nothing is silently dropped."""
from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from quietbridge.signs import capture_quality as cq
from quietbridge.signs.lab import FrameResult, HandResult, create_server
from quietbridge.signs.tests.test_capture_quality import gesture_try
from quietbridge.signs.tests.test_mouth_capture import FakeTracker
from quietbridge.signs.tests.test_sequence_lab import _jpeg


class MovingEngine:
    """Replays one synthetic right-hand try, one frame per call."""

    def __init__(self, move=0.3, n=40):
        lm, hp, _ = gesture_try(n=n, move=move)
        self.frames = [lm[f, 1].tolist() if hp[f, 1] else None for f in range(n)]
        self.calls = 0

    def infer(self, frame_rgb, timestamp_ms):
        landmarks = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        hands = [HandResult("Right", "", 0.0, landmarks)] if landmarks is not None else []
        return FrameResult(hands=hands, events=[])

    def close(self):
        pass


@pytest.fixture()
def lab(tmp_path):
    engine = MovingEngine()
    srv = create_server(engine=engine, port=0, data_root=tmp_path / "seq")
    clock = iter(range(1000, 10_000_000, 40))  # 25 fps camera clock
    srv.next_timestamp_ms = lambda: next(clock)
    srv.mouth._tracker_factory = FakeTracker
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield srv, conn, engine
    conn.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def call(lab, method, path, body=None, content_type="application/json"):
    srv, conn, _ = lab
    headers = {"X-QuietBridge-Token": srv.token, "Content-Type": content_type}
    conn.request(method, path, body=body if isinstance(body, (bytes, type(None))) else json.dumps(body),
                 headers=headers)
    response = conn.getresponse()
    return response.status, json.loads(response.read())


def frames(lab, n):
    for _ in range(n):
        assert call(lab, "POST", "/api/frame", _jpeg(), "image/jpeg")[0] == 200


def saved_files(srv):
    return sorted(Path(srv.data_root).glob("*/*.npz"))


def stored_quality(path):
    with np.load(path, allow_pickle=False) as data:
        return cq.from_npz(data)


def test_poor_try_waits_for_a_decision_then_save_anyway_keeps_it(lab):
    srv, _, engine = lab
    assert call(lab, "POST", "/api/start", {"label": "HELLO", "split": "train", "session": "s1"})[0] == 200
    frames(lab, 4)  # far too short
    status, body = call(lab, "POST", "/api/stop", {"review": True})
    assert status == 200 and body["review_required"] is True
    assert "too_few_frames" in cq.warning_codes(body["quality"])
    assert saved_files(srv) == [] and srv.recording is not None  # nothing saved, nothing dropped
    frames(lab, 3)  # the camera keeps sending frames while the person reads the warning
    assert len(srv.recording.timestamps_ms) == 4
    status, body = call(lab, "POST", "/api/stop", {"accept_quality": True})
    assert status == 200 and body["saved"]
    (path,) = saved_files(srv)
    review = stored_quality(path)
    assert review["decision"] == "saved_anyway"
    assert review["measured"]["frames"] == 4 and review["thresholds"] == cq.THRESHOLDS


def test_discard_and_retry_saves_nothing(lab):
    srv, _, _ = lab
    call(lab, "POST", "/api/start", {"label": "HELLO", "split": "train", "session": "s1"})
    frames(lab, 3)
    assert call(lab, "POST", "/api/stop", {"review": True})[1]["review_required"] is True
    status, body = call(lab, "POST", "/api/discard", {})
    assert status == 200 and body["discarded"] is True
    assert saved_files(srv) == [] and srv.recording is None


def test_clean_try_saves_at_once_with_its_measurements(lab):
    srv, _, _ = lab
    call(lab, "POST", "/api/start", {"label": "HELLO", "split": "train", "session": "s1"})
    frames(lab, 40)
    status, body = call(lab, "POST", "/api/stop", {"review": True})
    assert status == 200 and "review_required" not in body and body["saved"]
    review = stored_quality(saved_files(srv)[0])
    assert review["decision"] == "no_warnings" and review["warnings"] == []
    assert review["measured"]["frames"] == 40


def test_stop_without_review_flag_keeps_old_behaviour_and_records_quality(lab):
    srv, _, _ = lab
    call(lab, "POST", "/api/start", {"label": "HELLO", "split": "train", "session": "s1"})
    frames(lab, 3)
    status, body = call(lab, "POST", "/api/stop", b"{}")
    assert status == 200 and body["saved"]
    assert stored_quality(saved_files(srv)[0])["decision"] == "not_reviewed"


def test_review_flags_must_be_booleans(lab):
    call(lab, "POST", "/api/start", {"label": "HELLO", "split": "train", "session": "s1"})
    frames(lab, 2)
    status, body = call(lab, "POST", "/api/stop", {"review": "yes"})
    assert status == 400 and "boolean" in body["error"]


def test_lip_try_with_hidden_mouth_waits_for_a_decision(lab, monkeypatch):
    srv, _, _ = lab
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))
    assert call(lab, "POST", "/api/mouth/start", {"text": "hello", "split": "train", "session": "m1"})[0] == 200
    tracker = srv.mouth._tracker
    for i in range(20):
        tracker.see_face = i >= 12  # mouth hidden for the first 12 of 20 frames
        frames(lab, 1)
    status, body = call(lab, "POST", "/api/mouth/stop", {"review": True})
    assert status == 200 and body["review_required"] is True
    assert cq.warning_codes(body["quality"]) == ["mouth_not_visible"]
    assert body["mouth"]["awaiting_review"] is True and srv.mouth.last_saved is None
    frames(lab, 2)
    assert body["mouth"]["frames"] == srv.mouth.snapshot()["frames"]  # frozen while reviewing
    status, body = call(lab, "POST", "/api/mouth/stop", {"accept_quality": True})
    assert status == 200 and body["mouth"]["saved_count"] == 1
    review = stored_quality(srv.mouth.last_saved)
    assert review["decision"] == "saved_anyway" and review["measured"]["face_ratio"] == pytest.approx(0.4)


def test_paired_review_combines_both_inputs(lab, monkeypatch):
    srv, _, _ = lab
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))
    profile = srv.profile_manager.get_active_profile()
    body = {"profile": profile.id, "label": "HELLO", "text": profile.lip_prompts[0],
            "split": "train", "session": "p1"}
    assert call(lab, "POST", "/api/paired/start", body)[0] == 200
    frames(lab, 4)
    status, review = call(lab, "POST", "/api/paired/stop", {"review": True})
    assert status == 200 and review["review_required"] is True
    assert len(review["quality"]["parts"]) == 2 and "too_few_frames" in cq.warning_codes(review["quality"])
    status, saved = call(lab, "POST", "/api/paired/stop", {"accept_quality": True})
    assert status == 200
    assert stored_quality(saved["saved"]["sequence"])["decision"] == "saved_anyway"
    assert stored_quality(saved["saved"]["mouth"])["kind"] == "lips"

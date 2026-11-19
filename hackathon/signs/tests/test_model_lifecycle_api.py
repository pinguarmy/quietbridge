"""Loopback lab model API contracts, without training or external services."""
import http.client
import json
import threading

import pytest

from hackathon.signs.lab import create_server


class FakeModels:
    def __init__(self):
        self.calls = []

    def status(self, profile):
        return {"profile": profile.id, "candidates": [], "active": {}, "jobs": []}

    def start_training(self, profile, kind):
        self.calls.append(("train", profile.id, kind))

    def promote(self, profile, kind, candidate_id):
        self.calls.append(("promote", profile.id, kind, candidate_id))

    def rollback(self, profile, kind):
        self.calls.append(("rollback", profile.id, kind))

    def evaluate_language(self, profile, candidate_id):
        self.calls.append(("evaluate", profile.id, candidate_id))


@pytest.fixture
def api(tmp_path):
    server = create_server(engine=object(), port=0, data_root=tmp_path / "seq")
    server.models = FakeModels()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    yield server, connection
    connection.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request(api, method, path, data=None, token=True):
    server, conn = api
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-QuietBridge-Token"] = server.token
    conn.request(method, path, json.dumps(data) if data is not None else None, headers)
    response = conn.getresponse()
    return response.status, json.loads(response.read())


def test_model_actions_use_active_profile_and_return_status(api):
    server, _ = api
    profile = server.profile_manager.get_active_profile().id
    code, status = request(api, "GET", "/api/models")
    assert code == 200 and status["profile"] == profile
    for action, payload in (("train", {"kind": "motion"}),
                            ("promote", {"kind": "lips", "candidate_id": "candidate-1"}),
                            ("rollback", {"kind": "language"}),
                            ("evaluate", {"kind": "language", "candidate_id": "candidate-2"})):
        code, status = request(api, "POST", "/api/models/" + action, payload)
        assert code == 200 and status["profile"] == profile
    assert server.models.calls == [("train", profile, "motion"),
                                    ("promote", profile, "lips", "candidate-1"),
                                    ("rollback", profile, "language"),
                                    ("evaluate", profile, "candidate-2")]


def test_model_actions_require_token_and_no_recording(api):
    server, _ = api
    code, _ = request(api, "POST", "/api/models/train", {"kind": "motion"}, token=False)
    assert code == 403 and not server.models.calls
    server.paired_recording = {"profile_id": "example"}
    code, body = request(api, "POST", "/api/models/train", {"kind": "motion"})
    assert code == 409 and "error" in body and not server.models.calls


@pytest.fixture
def real_api(tmp_path):
    server = create_server(engine=object(), port=0, data_root=tmp_path / "seq")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    yield server, connection
    connection.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _promotable_motion_candidate(profile, candidate_id, content):
    """A finished motion candidate on disk, as the trainer would leave it."""
    directory = profile.profile_dir / "models" / "candidates" / candidate_id
    directory.mkdir(parents=True)
    model, report = directory / "motion.npz", directory / "motion.report.json"
    model.write_bytes(content)
    report.write_text(json.dumps({"validation_trials": 2, "validation_ungated": {"trials": 2}}))
    (directory / "candidate.json").write_text(json.dumps({
        "id": candidate_id, "kind": "motion", "status": "completed", "error": None,
        "model_path": str(model), "report_path": str(report)}))


def test_restore_state_comes_from_saved_history(real_api):
    server, _ = real_api
    profile = server.profile_manager.get_active_profile()
    code, status = request(real_api, "GET", "/api/models")
    assert code == 200
    # Nothing saved yet: the page must not offer a restore that can only fail.
    assert status["restore"]["motion"] == {"available": False, "restores": None}
    assert status["restore"]["lips"]["available"] is False

    first, second = "a" * 32, "b" * 32
    _promotable_motion_candidate(profile, first, b"first")
    _promotable_motion_candidate(profile, second, b"second")
    code, status = request(real_api, "POST", "/api/models/promote", {"kind": "motion", "candidate_id": first})
    assert code == 200
    # Restoring now goes back to "no saved model", i.e. the stock behaviour.
    assert status["restore"]["motion"] == {"available": True, "restores": "stock"}
    code, status = request(real_api, "POST", "/api/models/promote", {"kind": "motion", "candidate_id": second})
    assert status["restore"]["motion"] == {"available": True, "restores": "previous_model"}

    code, status = request(real_api, "POST", "/api/models/rollback", {"kind": "motion"})
    assert code == 200
    assert (profile.profile_dir / "models" / "motion.npz").read_bytes() == b"first"
    assert status["restore"]["motion"] == {"available": True, "restores": "stock"}
    code, status = request(real_api, "POST", "/api/models/rollback", {"kind": "motion"})
    assert not (profile.profile_dir / "models" / "motion.npz").exists()
    assert status["restore"]["motion"] == {"available": False, "restores": None}
    code, body = request(real_api, "POST", "/api/models/rollback", {"kind": "motion"})
    assert code == 400 and "No previous model" in body["error"]


def test_lifecycle_restore_field_takes_precedence(api):
    server, _ = api
    server.models.status = lambda profile: {"candidates": [], "active": {}, "jobs": [],
                                            "restore": {"motion": {"available": True, "restores": "stock"}}}
    code, status = request(api, "GET", "/api/models")
    assert code == 200 and status["restore"] == {"motion": {"available": True, "restores": "stock"}}

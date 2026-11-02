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

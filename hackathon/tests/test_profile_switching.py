"""Tests for hot profile switching without restarting process.

Validates:
- GET /api/profiles and POST /api/profile/select HTTP API contracts.
- Conflict rejection (409) when recording, continuous session active/finalizing,
  or browser camera is running without interrupting recording.
- Validation rejection (400) on invalid or non-existent profile IDs or missing models.
- Clean stop and join of old workers, worker generation protection against state leaks,
  clearing old queues/tracking/results, and preserving Speaker/CleanupClient/server.
- Proper loading of motion.npz, lip.pth, lm.pth, custom vocabulary, and identity reporting.
- Explicit CLI model precedence over profile candidates.
- Safe routing of corrections to active profile.
"""
import copy
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from hackathon.contracts import Decision, SignEvent
from hackathon.profiles import (
    ProfileManager,
    profile_language_model,
    profile_lip_model,
    profile_motion_model,
)
from hackathon.runtime import ProfileSwitchConflict, Runtime
from hackathon.server import make_server


def make_fake_motion_model(path: Path, labels=("HELLO", "WATER")):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        templates=np.zeros((len(labels), 30, 2, 21, 3), dtype=np.float32),
        masks=np.ones((len(labels), 30, 2), dtype=bool),
        labels=np.array(labels),
        reject_threshold=0.8,
        margin_threshold=0.1,
        loo_report=np.array("{}"),
        calibration_report=np.array("{}"),
        label_thresholds=np.array("{}"),
        meta=np.array(json.dumps({"format": "motion-template-v1", "labels": list(labels)})),
    )


def http_get(url: str, path: str):
    with urlopen(f"{url}{path}", timeout=5) as res:
        return res.status, json.loads(res.read().decode())


def http_post(url: str, path: str, data: dict, token: str):
    req = Request(
        f"{url}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "X-Lipflow-Token": token},
    )
    with urlopen(req, timeout=5) as res:
        return res.status, json.loads(res.read().decode())


def http_post_raw(url: str, path: str, data: dict, token: str):
    req = Request(
        f"{url}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "X-Lipflow-Token": token},
    )
    try:
        with urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read().decode())
    except HTTPError as err:
        return err.code, json.loads(err.read().decode())


@pytest.fixture
def profile_env(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    default_p = manager.ensure_default_profile()

    alice_gestures = [
        {"label": "WAVE", "display_name": "Wave", "concept": "Hello friend", "description": "Hand wave"},
        {"label": "REST", "display_name": "Rest", "concept": "Rest", "description": "Relax"},
    ]
    alice_p = manager.create_profile("alice", "Alice Profile", gestures=alice_gestures, lip_prompts=["Hi there."])

    bob_gestures = [
        {"label": "THUMBS_UP", "display_name": "Thumbs Up", "concept": "Good", "description": "Thumb up"},
        {"label": "REST", "display_name": "Rest", "concept": "Rest", "description": "Relax"},
    ]
    bob_p = manager.create_profile("bob", "Bob Profile", gestures=bob_gestures, lip_prompts=["Looking good."])

    return manager, default_p, alice_p, bob_p


def test_api_get_profiles(profile_env):
    manager, default_p, alice_p, bob_p = profile_env
    runtime = Runtime(rehearsal=True, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        status, data = http_get(url, "/api/profiles")
        assert status == 200
        assert data["active_profile"] == "default"
        profile_ids = [p["id"] for p in data["profiles"]]
        assert "default" in profile_ids
        assert "alice" in profile_ids
        assert "bob" in profile_ids
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_api_profile_select_gate_conditions_409(profile_env):
    manager, default_p, alice_p, bob_p = profile_env
    runtime = Runtime(browser_camera=True, rehearsal=False, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]

        # Case 1: Browser camera is running -> 409
        runtime.action("camera_session", {"active": True})
        assert runtime._camera_session_active is True
        status, err = http_post_raw(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 409
        assert "browser camera" in err["error"].lower()
        # Ensure camera session was NOT interrupted
        assert runtime._camera_session_active is True
        assert runtime.profile.id == "default"

        # Stop camera session
        runtime.action("camera_session", {"active": False})

        # Case 2: Runtime phase is not idle (e.g. recording) -> 409
        with runtime.lock:
            runtime.state["phase"] = "recording"
            runtime.state["recording"] = True
        status, err = http_post_raw(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 409
        assert "recording or decoding" in err["error"].lower()
        # Verify recording was NOT cancelled or interrupted
        assert runtime.state["phase"] == "recording"
        assert runtime.state["recording"] is True
        assert runtime.profile.id == "default"

        # Case 3: Session active or finalizing -> 409
        with runtime.lock:
            runtime.state["phase"] = "idle"
            runtime.state["recording"] = False
            runtime.state["session_finalizing"] = True
        status, err = http_post_raw(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 409
        assert "continuous session" in err["error"].lower()
        assert runtime.state["session_finalizing"] is True
        assert runtime.profile.id == "default"

        # Case 4: Auto recognition active -> 409
        with runtime.lock:
            runtime.state["session_finalizing"] = False
            runtime.state["auto_recognition"] = True
        status, err = http_post_raw(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 409
        assert "continuous session" in err["error"].lower()
        assert runtime.profile.id == "default"

    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_api_profile_select_validation_400(profile_env):
    manager, default_p, alice_p, bob_p = profile_env
    runtime = Runtime(rehearsal=True, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]

        # Empty id
        status, err = http_post_raw(url, "/api/profile/select", {"id": ""}, token)
        assert status == 400
        assert "non-empty" in err["error"].lower()

        # Unsafe id
        status, err = http_post_raw(url, "/api/profile/select", {"id": "../evil"}, token)
        assert status == 400
        assert "invalid" in err["error"].lower()

        # Nonexistent profile
        status, err = http_post_raw(url, "/api/profile/select", {"id": "unknown_user"}, token)
        assert status == 400
        assert "not found" in err["error"].lower()

        # Profile remains default
        assert runtime.profile.id == "default"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_api_profile_select_success_and_vocabulary_update(profile_env):
    manager, default_p, alice_p, bob_p = profile_env
    runtime = Runtime(rehearsal=True, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]
        assert runtime.profile.id == "default"

        # Switch to Alice (rehearsal mode)
        status, resp = http_post(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 200
        assert resp["ok"] is True
        assert resp["active_profile"] == "alice"
        assert resp["profile_name"] == "Alice Profile"
        assert runtime.profile.id == "alice"
        assert manager.active_profile_id == "alice"

        # Verify active_profile.txt on disk
        assert manager.active_file.read_text().strip() == "alice"

        # Switch to Bob
        status, resp = http_post(url, "/api/profile/select", {"id": "bob"}, token)
        assert status == 200
        assert resp["active_profile"] == "bob"
        assert resp["profile_name"] == "Bob Profile"
        assert runtime.profile.id == "bob"
        assert manager.active_profile_id == "bob"

        # Switch back to Default
        status, resp = http_post(url, "/api/profile/select", {"id": "default"}, token)
        assert status == 200
        assert resp["active_profile"] == "default"
        assert runtime.profile.id == "default"
        assert manager.active_profile_id == "default"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_profile_switching_with_personal_models_and_identity(profile_env):
    manager, default_p, alice_p, bob_p = profile_env

    # Setup Alice with trained motion model and lip model files
    alice_motion = profile_motion_model(alice_p)
    make_fake_motion_model(alice_motion, labels=("WAVE", "REST"))
    alice_lip = profile_lip_model(alice_p)
    alice_lip.parent.mkdir(parents=True, exist_ok=True)
    alice_lip.write_bytes(b"FAKE_VSR_WEIGHTS_ALICE")
    alice_lm = profile_language_model(alice_p)
    alice_lm.write_bytes(b"FAKE_LM_WEIGHTS_ALICE")

    # Start runtime in rehearsal mode on default profile
    runtime = Runtime(rehearsal=True, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]
        assert runtime.motion_model_path is None
        assert runtime.state["gesture_model"] == "stock"

        # Switch to Alice (has custom motion model and custom lip model)
        status, resp = http_post(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 200
        assert runtime.profile.id == "alice"
        assert runtime.motion_model_path == alice_motion
        assert runtime.lip_model_path == alice_lip
        assert runtime.lip_language_model_path == alice_lm
        assert runtime.state["gesture_model"] == "personal motion-v2 (experimental)"
        assert set(runtime.vocabulary.keys()) == {"WAVE"}
        assert resp["lip_model"]["mode"] == "personal"
        assert resp["lip_model"]["vsr_path"] == str(alice_lip)
        assert "vsr_sha256" in resp["lip_model"]

        # Switch to Bob (untrained profile, no models)
        status, resp = http_post(url, "/api/profile/select", {"id": "bob"}, token)
        assert status == 200
        assert runtime.profile.id == "bob"
        assert runtime.motion_model_path is None
        assert runtime.lip_model_path is None
        assert runtime.lip_language_model_path is None
        assert runtime.state["gesture_model"] == "stock"
        assert "no trained motion model" in runtime.state["profile_warning"]
        assert resp["lip_model"]["mode"] == "stock"

    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_explicit_cli_model_precedence(profile_env, tmp_path):
    manager, default_p, alice_p, bob_p = profile_env

    # Alice has her own candidate models
    alice_motion = profile_motion_model(alice_p)
    make_fake_motion_model(alice_motion, labels=("WAVE", "REST"))
    alice_lip = profile_lip_model(alice_p)
    alice_lip.parent.mkdir(parents=True, exist_ok=True)
    alice_lip.write_bytes(b"ALICE_LIP")

    # Explicit CLI models
    cli_motion = tmp_path / "cli_motion.npz"
    make_fake_motion_model(cli_motion, labels=("CLI_SIGN", "REST"))
    cli_lip = tmp_path / "cli_lip.pth"
    cli_lip.write_bytes(b"CLI_LIP")

    runtime = Runtime(
        rehearsal=True,
        profile=default_p,
        profile_manager=manager,
        explicit_motion_model=cli_motion,
        explicit_lip_model=cli_lip,
    )
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]

        # Initial models are the explicit CLI ones
        assert runtime.motion_model_path == cli_motion
        assert runtime.lip_model_path == cli_lip

        # Switch to Alice: explicit CLI paths must prevail over Alice's candidate models
        status, resp = http_post(url, "/api/profile/select", {"id": "alice"}, token)
        assert status == 200
        assert runtime.profile.id == "alice"
        assert runtime.motion_model_path == cli_motion  # Not alice_motion!
        assert runtime.lip_model_path == cli_lip  # Not alice_lip!
        assert resp["lip_model"]["vsr_path"] == str(cli_lip)

    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()


def test_worker_lifecycle_stop_and_join_no_leak(profile_env):
    manager, default_p, alice_p, bob_p = profile_env

    # Create fake workers to verify generation tracking and clean termination
    stopped_workers = []
    worker_runs = []

    def fake_hands(worker_gen, worker_stop):
        worker_runs.append(("hands", worker_gen))
        while not worker_stop.is_set():
            time.sleep(0.01)
        stopped_workers.append(("hands", worker_gen))

    def fake_lips(worker_gen, worker_stop):
        worker_runs.append(("lips", worker_gen))
        while not worker_stop.is_set():
            time.sleep(0.01)
        stopped_workers.append(("lips", worker_gen))

    runtime = Runtime(browser_camera=True, rehearsal=False, profile=default_p, profile_manager=manager)
    runtime._hands = fake_hands
    runtime._lips = fake_lips
    runtime.start()

    # Wait for workers to start
    time.sleep(0.05)
    assert len(worker_runs) == 2
    initial_gen = runtime._worker_generation

    # Populate queues and results to verify they get cleared
    runtime.hand_frames.put(("fake_frame", 100))
    runtime.lip_frames.put(("fake_frame", 100))
    runtime.hand_control.put(("cmd", "val", 1))
    runtime.tracking = {"hands": [{"label": "Left"}], "mouth": "fake_mouth"}
    runtime.events.append(SignEvent("HELLO", 0.9, 0, 100, True))

    # Keep speaker and cleanup intact
    speaker_ref = runtime.speaker
    cleanup_ref = runtime.cleanup

    # Perform profile switch
    runtime.switch_profile("alice")

    # Verify old workers stopped and joined
    assert ("hands", initial_gen) in stopped_workers
    assert ("lips", initial_gen) in stopped_workers

    # Verify new workers started with incremented generation
    new_gen = runtime._worker_generation
    assert new_gen > initial_gen
    assert ("hands", new_gen) in worker_runs
    assert ("lips", new_gen) in worker_runs

    # Verify queues and tracking were drained / reset
    assert runtime.hand_frames.empty()
    assert runtime.lip_frames.empty()
    assert runtime.hand_control.empty()
    assert runtime.tracking == {"hands": [], "mouth": None}
    assert len(runtime.events) == 0

    # Verify Speaker and CleanupClient instances were PRESERVED (not closed)
    assert runtime.speaker is speaker_ref
    assert runtime.cleanup is cleanup_ref

    # Clean shutdown
    runtime.close()


def test_corrections_recorded_to_active_profile(profile_env):
    manager, default_p, alice_p, bob_p = profile_env
    from hackathon.corrections import CorrectionStore

    runtime = Runtime(rehearsal=True, profile=default_p, profile_manager=manager)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    try:
        _, state_data = http_get(url, "/api/state")
        token = state_data["token"]

        # Switch to Alice
        http_post(url, "/api/profile/select", {"id": "alice"}, token)
        assert runtime.profile.id == "alice"

        # Produce a rehearsal result
        _, state_after = http_post(
            url, "/api/action/rehearse", {"text": "I need watr", "label": ""}, token
        )
        rev = state_after["revision"]

        # Save correction with learn=True
        status, corr_res = http_post(
            url,
            "/api/action/correct",
            {"revision": rev, "text": "I need water", "learn": True},
            token,
        )
        assert status == 200
        assert corr_res["decision"]["text"] == "I need water"

        # Check Alice's corrections store
        alice_store = CorrectionStore(alice_p)
        examples = alice_store.list()
        assert len(examples) == 1
        assert examples[0]["original"] == "I need watr"
        assert examples[0]["corrected"] == "I need water"

        # Verify Bob's corrections store is untouched
        bob_store = CorrectionStore(bob_p)
        assert len(bob_store.list()) == 0

    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()

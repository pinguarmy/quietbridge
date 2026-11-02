"""Unit tests for hackathon.profiles (profile management, isolation, validation)."""
import os
from pathlib import Path
import pytest

from hackathon.profiles import (
    DEFAULT_GESTURES,
    DEFAULT_LIP_PROMPTS,
    SPLITS,
    ProfileManager,
    check_session_split,
    get_default_root,
    is_safe_profile_id,
    profile_language_model,
    profile_lip_model,
    profile_motion_model,
    profile_vocabulary,
    validate_gestures,
    validate_lip_prompts,
)


def test_safe_profile_id_validation():
    assert is_safe_profile_id("default")
    assert is_safe_profile_id("peter")
    assert is_safe_profile_id("alice_123-test")
    assert not is_safe_profile_id("")
    assert not is_safe_profile_id(".")
    assert not is_safe_profile_id("..")
    assert not is_safe_profile_id("../evil")
    assert not is_safe_profile_id("evil/path")
    assert not is_safe_profile_id("evil\\path")
    assert not is_safe_profile_id("name with spaces")
    assert not is_safe_profile_id("-leading-dash")
    assert not is_safe_profile_id("a" * 65)


def test_validate_gestures():
    valid = [
        {"label": "CUSTOM_ONE", "display_name": "One", "concept": "Concept 1", "description": "Do something"},
        {"label": "CUSTOM_TWO", "display_name": "Two", "concept": "Concept 2", "description": "Do another"},
    ]
    res = validate_gestures(valid)
    assert len(res) == 2
    assert res[0]["label"] == "CUSTOM_ONE"
    assert res[0]["emoji"] == "✋"

    with pytest.raises(ValueError, match="safe UPPER_SNAKE_CASE"):
        validate_gestures([{"label": "lowercase"}])
    with pytest.raises(ValueError, match="Duplicate gesture label"):
        validate_gestures([{"label": "HELLO"}, {"label": "HELLO"}])
    with pytest.raises(ValueError, match="non-empty list"):
        validate_gestures([])


@pytest.mark.parametrize("template", ["no placeholder", "{gesture} {gesture}",
                                      "{{gesture}}", "{gesture!r}", "{gesture} {other}",
                                      "{gesture} {", 42])
def test_invalid_intent_template(template):
    with pytest.raises(ValueError, match="intent_template"):
        validate_gestures([{"label": "WATER", "intent_template": template}])


def test_optional_intent_template_round_trip(tmp_path):
    pm = ProfileManager(root_dir=tmp_path)
    profile = pm.create_profile("alice", "Alice", gestures=[
        {"label": "WATER", "concept": "water", "intent_template": " Please bring {gesture} "},
        {"label": "YES", "intent_template": ""},
        {"label": "NO"},
    ])
    assert profile.gestures[0]["intent_template"] == "Please bring {gesture}"
    assert profile.gestures[1]["intent_template"] == ""
    assert "intent_template" not in profile.gestures[2]
    assert pm.get_profile("alice").gestures == profile.gestures
    assert profile_vocabulary(profile)["WATER"]["intent_template"] == "Please bring {gesture}"
    assert "intent_template" not in profile_vocabulary(profile)["YES"]
    with pytest.raises(ValueError, match="intent_template"):
        pm.update_profile("alice", gestures=[{"label": "WATER", "intent_template": "{other}"}])
    assert pm.get_profile("alice").gestures == profile.gestures


def test_validate_lip_prompts():
    valid = ["Hello world", "Another prompt"]
    assert validate_lip_prompts(valid) == ["Hello world", "Another prompt"]

    with pytest.raises(ValueError, match="non-empty list"):
        validate_lip_prompts([])
    with pytest.raises(ValueError, match="1-300 characters"):
        validate_lip_prompts([""])
    with pytest.raises(ValueError, match="1-300 characters"):
        validate_lip_prompts(["a" * 301])


def test_default_root_redirection(tmp_path, monkeypatch):
    monkeypatch.delenv("QUIETBRIDGE_DATA_ROOT", raising=False)
    monkeypatch.delenv("QUIETBRIDGE_ROOT", raising=False)
    default_root = get_default_root()
    assert default_root == Path.home() / ".local" / "share" / "quietbridge"

    monkeypatch.setenv("QUIETBRIDGE_DATA_ROOT", str(tmp_path / "custom_root"))
    assert get_default_root() == tmp_path / "custom_root"


def test_profile_manager_creation_and_defaults(tmp_path):
    pm = ProfileManager(root_dir=tmp_path)
    # Automatically creates "default" profile with sensible defaults
    default_p = pm.get_active_profile()
    assert default_p.id == "default"
    assert default_p.name == "Default Profile"
    assert len(default_p.gestures) == len(DEFAULT_GESTURES)
    assert default_p.lip_prompts == DEFAULT_LIP_PROMPTS
    assert (default_p.sequences_dir / "train").is_dir()
    assert (default_p.sequences_dir / "validation").is_dir()
    assert (default_p.sequences_dir / "test").is_dir()
    assert (default_p.mouths_dir / "train").is_dir()
    assert (default_p.mouths_dir / "validation").is_dir()
    assert (default_p.mouths_dir / "test").is_dir()


def test_profile_isolation_and_switching(tmp_path):
    pm = ProfileManager(root_dir=tmp_path)
    alice = pm.create_profile(
        "alice",
        "Alice Smith",
        gestures=[
            {"label": "ALICE_SIGN", "display_name": "Alice Sign", "concept": "Greeting", "description": "Wave left"},
            {"label": "REST", "display_name": "Rest", "concept": "Rest", "description": "Relax hand"},
        ],
        lip_prompts=["Prompt for Alice 1", "Prompt for Alice 2"],
    )
    assert alice.id == "alice"
    assert (alice.sequences_dir / "train").is_dir()
    assert (alice.mouths_dir / "train").is_dir()
    assert alice.sequences_dir != pm.get_profile("default").sequences_dir

    pm.set_active_profile("alice")
    assert pm.active_profile_id == "alice"
    assert pm.get_active_profile().id == "alice"

    # Profile listing
    profiles = pm.list_profiles(include_stats=True)
    ids = [p["id"] for p in profiles]
    assert "default" in ids and "alice" in ids


def test_profile_update(tmp_path):
    pm = ProfileManager(root_dir=tmp_path)
    pm.create_profile("bob", "Bob")
    updated = pm.update_profile(
        "bob",
        name="Robert",
        lip_prompts=["Updated prompt"],
    )
    assert updated.name == "Robert"
    assert updated.lip_prompts == ["Updated prompt"]

    reloaded = pm.get_profile("bob")
    assert reloaded.name == "Robert"
    assert reloaded.lip_prompts == ["Updated prompt"]


def test_existing_data_preserved_not_deleted(tmp_path):
    # Pre-populate some existing files in tmp_path
    legacy_seq = tmp_path / "sequences" / "train"
    legacy_seq.mkdir(parents=True)
    (legacy_seq / "old_recording.npz").write_text("dummy")

    legacy_mouths = tmp_path / "mouths" / "train"
    legacy_mouths.mkdir(parents=True)
    (legacy_mouths / "old_mouth.npz").write_text("dummy")

    pm = ProfileManager(root_dir=tmp_path)
    pm.create_profile("test_user", "Test User")

    # Verify legacy files were NOT touched or deleted
    assert (legacy_seq / "old_recording.npz").is_file()
    assert (legacy_mouths / "old_mouth.npz").is_file()


def test_session_split_leakage_rejection(tmp_path):
    seq_dir = tmp_path / "seq"
    (seq_dir / "train").mkdir(parents=True)
    (seq_dir / "train" / "session1_trial1.npz").write_text("data")

    # Trying session1 in validation or test must fail
    with pytest.raises(ValueError, match="already has recordings in train"):
        check_session_split(seq_dir, "validation", "session1")

    with pytest.raises(ValueError, match="already has recordings in train"):
        check_session_split(seq_dir, "test", "session1")

    # Clean in train
    check_session_split(seq_dir, "train", "session1")

    # Clean in other session
    check_session_split(seq_dir, "validation", "session2")


def test_profile_vocabulary_and_candidate_paths_stay_custom_and_unverified(tmp_path):
    profile = ProfileManager(root_dir=tmp_path).create_profile(
        "alice", "Alice", gestures=[
            {"label": "WATER", "display_name": "Water sign", "concept": "water"},
            {"label": "REST", "display_name": "Rest", "concept": "rest"},
        ])
    vocabulary = profile_vocabulary(profile)
    assert vocabulary == {"WATER": {
        "display_name": "Water sign", "phrase": "water",
        "matches": ["water", "Water sign"], "enabled": True, "verified": False,
    }}
    assert profile_motion_model(profile) == profile.profile_dir / "models" / "motion.npz"
    assert profile_lip_model(profile) == profile.profile_dir / "models" / "lip.pth"
    assert profile_language_model(profile) == profile.profile_dir / "models" / "lm.pth"
    assert not profile_motion_model(profile).exists()

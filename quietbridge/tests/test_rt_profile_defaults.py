"""Default profile content is plain language and never rewrites saved profiles."""
import json

from quietbridge.profiles import DEFAULT_GESTURES, ProfileManager, profile_vocabulary, validate_gestures

TECHNICAL = ("REST", "NEUTRAL", "negative class", "demo", "fixed custom", "0.5 seconds", "15 seconds")


def test_default_gesture_text_is_plain_and_honest():
    assert validate_gestures([dict(g) for g in DEFAULT_GESTURES]) == DEFAULT_GESTURES
    for gesture in DEFAULT_GESTURES:
        for field in ("display_name", "concept", "description", "hint"):
            for word in TECHNICAL:
                assert word not in gesture[field], (gesture["label"], field, word)
        if gesture["label"] != "REST":
            assert "not a standard sign" in gesture["hint"]
    rest = next(g for g in DEFAULT_GESTURES if g["label"] == "REST")
    assert "silent" in rest["hint"]


def test_default_labels_keep_their_ids_and_drop_the_demo_movement():
    assert [g["label"] for g in DEFAULT_GESTURES] == ["HELLO", "WATER", "YES", "NO", "REST"]


def test_default_meanings_are_usable_message_text():
    # The concept is the text a recognized gesture contributes to a message.
    vocabulary = profile_vocabulary(type("P", (), {"gestures": DEFAULT_GESTURES})())
    assert vocabulary["HELLO"]["phrase"] == "Hello"
    assert "REST" not in vocabulary
    assert all("/" not in entry["phrase"] for entry in vocabulary.values())


def test_saved_profile_text_loads_unchanged(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    folder = tmp_path / "profiles" / "old"
    folder.mkdir(parents=True)
    old_gestures = [
        {"label": "WELCOME", "display_name": "Welcome", "concept": "Welcome / Inward welcome",
         "description": "d", "hint": "This is a fixed custom demo motion, not a standard sign.", "emoji": "x"},
        {"label": "REST", "display_name": "Rest", "concept": "Rest / Relaxed negative class",
         "description": "d", "hint": "REST is its own label; it is not NEUTRAL.", "emoji": "y"},
    ]
    (folder / "profile.json").write_text(json.dumps(
        {"id": "old", "name": "Old", "gestures": old_gestures, "lip_prompts": ["Hi"]}), encoding="utf-8")
    loaded = manager.get_profile("old")
    assert loaded.gestures == old_gestures and loaded.lip_prompts == ["Hi"]


def test_profile_file_without_gestures_keeps_the_original_label_set(tmp_path):
    manager = ProfileManager(root_dir=tmp_path)
    folder = tmp_path / "profiles" / "bare"
    folder.mkdir(parents=True)
    (folder / "profile.json").write_text(json.dumps({"id": "bare"}), encoding="utf-8")
    labels = [g["label"] for g in manager.get_profile("bare").gestures]
    assert labels == ["HELLO", "WELCOME", "WATER", "YES", "NO", "REST"]

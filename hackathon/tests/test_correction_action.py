from unittest.mock import Mock

import pytest

from hackathon.corrections import CorrectionStore
from hackathon.profiles import Profile
from hackathon.runtime import Runtime


def runtime_with_profile(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    runtime = Runtime(rehearsal=True, vocabulary={}, profile=profile, profile_name="Alice")
    runtime.speaker = Mock(available=True)
    return runtime, profile


def test_correction_can_learn_but_never_speaks_automatically(tmp_path):
    runtime, profile = runtime_with_profile(tmp_path)
    try:
        runtime.action("rehearse", {"text": "I need help"})
        old_revision = runtime.revision
        runtime.action("correct", {"revision": old_revision, "text": "I need some help", "learn": True})
        state = runtime.snapshot()
        assert state["decision"]["text"] == "I need some help"
        assert state["decision"]["status"] == "accepted"
        assert state["decision"]["reason"] == "corrected"
        assert state["revision"] > old_revision
        runtime.speaker.speak.assert_not_called()
        entries = CorrectionStore(profile).list()
        assert len(entries) == 1 and entries[0]["corrected"] == "I need some help"
        runtime.action("speak", {"revision": state["revision"], "text": "I need some help"})
        runtime.speaker.speak.assert_called_once_with("I need some help")
    finally:
        runtime.close()


def test_correction_without_learning_does_not_write_training_data(tmp_path):
    runtime, profile = runtime_with_profile(tmp_path)
    try:
        runtime.action("rehearse", {"text": "wrong"})
        runtime.action("correct", {"revision": runtime.revision, "text": "right", "learn": False})
        assert CorrectionStore(profile).list() == []
        assert runtime.state["last_correction"] == "Correction applied locally"
    finally:
        runtime.close()


def test_learning_requires_profile_and_correction_uses_current_revision(tmp_path):
    runtime = Runtime(rehearsal=True, vocabulary={})
    try:
        runtime.action("rehearse", {"text": "wrong"})
        with pytest.raises(ValueError, match="profile"):
            runtime.action("correct", {"revision": runtime.revision, "text": "right", "learn": True})
        with pytest.raises(ValueError, match="result changed"):
            runtime.action("correct", {"revision": runtime.revision - 1, "text": "right", "learn": False})
    finally:
        runtime.close()

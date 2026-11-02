import pytest

from hackathon.profiles import ProfileManager, profile_motion_model
from hackathon.runtime import Runtime
from hackathon.server import profile_configuration
from lipflow.vsr import LipReader


def test_explicit_personal_models_fail_before_loading_stock(tmp_path):
    missing_vsr = tmp_path / "missing-vsr.pth"
    with pytest.raises(FileNotFoundError, match="Personal lip model"):
        LipReader(personal=False, personal_vsr_path=missing_vsr)
    missing_lm = tmp_path / "missing-lm.pth"
    with pytest.raises(FileNotFoundError, match="Personal language model"):
        LipReader(personal=False, personal_lm_path=missing_lm)


def test_utterance_limit_is_configurable_and_validated():
    runtime = Runtime(rehearsal=True, vocabulary={}, max_utterance_seconds=45)
    try:
        assert runtime.max_utterance_ms == 45000
    finally:
        runtime.close()
    with pytest.raises(ValueError, match="at least one second"):
        Runtime(rehearsal=True, vocabulary={}, max_utterance_seconds=0.5)


def test_untrained_profile_uses_stock_vocabulary_with_visible_warning(tmp_path):
    profile = ProfileManager(root_dir=tmp_path).create_profile("alice", "Alice")
    vocabulary, motion, warning = profile_configuration(profile)
    assert vocabulary is None  # Runtime loads the compatible stock vocabulary.
    assert motion is None
    assert "using stock gestures" in warning


def test_profile_candidate_pair_uses_custom_vocabulary(tmp_path):
    profile = ProfileManager(root_dir=tmp_path).create_profile(
        "alice", "Alice", gestures=[
            {"label": "HELLO", "display_name": "Wave", "concept": "Hello"},
            {"label": "REST", "display_name": "Rest", "concept": "Rest"},
        ])
    path = profile_motion_model(profile)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"candidate")
    vocabulary, motion, warning = profile_configuration(profile)
    assert motion == path
    assert set(vocabulary) == {"HELLO"}
    assert warning == ""

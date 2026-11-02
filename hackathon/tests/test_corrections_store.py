import numpy as np
import pytest

from hackathon.corrections import CorrectionStore
from hackathon.profiles import Profile


def test_correction_requires_explicit_learning_and_persists(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    source = profile.mouths_dir / "train" / "clip.npz"
    np.savez(source, text=np.array("wrong"))
    store = CorrectionStore(profile)
    with pytest.raises(ValueError, match="Choose"):
        store.record(original="wrong", corrected="right")
    entry = store.record(original="wrong", corrected="right", candidates=["wrong", "write"],
                         source_paths=[source], learn=True)
    assert entry["sources"] == ["mouths/train/clip.npz"]
    assert entry["learn"] is True
    assert store.list() == [entry]


def test_corrections_reject_empty_same_or_external_sources(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    store = CorrectionStore(profile)
    with pytest.raises(ValueError, match="does not change"):
        store.record(original="same", corrected="same", learn=True)
    with pytest.raises(ValueError, match="1–2000"):
        store.record(original="wrong", corrected="", learn=True)
    external = tmp_path / "external.npz"
    np.savez(external, text=np.array("outside"))
    with pytest.raises(ValueError, match="inside this profile"):
        store.record(original="wrong", corrected="right", source_paths=[external], learn=True)
    assert store.list() == []

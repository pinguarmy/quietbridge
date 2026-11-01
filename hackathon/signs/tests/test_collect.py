import numpy as np
import pytest

from hackathon.signs.collect import save_trial


def test_save_trial_writes_independent_feature_only_npz(tmp_path):
    features = np.arange(12 * 63, dtype=np.float32).reshape(12, 63)
    first = save_trial(tmp_path, "train", "participant-a", "THUMB_UP", features)
    second = save_trial(tmp_path, "train", "participant-a", "THUMB_UP", features)

    assert first != second
    assert first.parent == tmp_path / "train"
    with np.load(first, allow_pickle=False) as saved:
        assert set(saved.files) == {"features", "label", "session_id", "trial_id"}
        np.testing.assert_array_equal(saved["features"], features)
        assert saved["features"].dtype == np.float32
        assert saved["label"].item() == "THUMB_UP"
        assert saved["session_id"].item() == "participant-a"
        assert saved["trial_id"].item() == first.stem.removeprefix("participant-a_")


def test_session_cannot_cross_splits(tmp_path):
    saved = save_trial(tmp_path, "train", "same-session", "OPEN_PALM", np.ones((8, 63)))
    with pytest.raises(ValueError, match="already has trials"):
        save_trial(tmp_path, "test", "same-session", "OPEN_PALM", np.ones((8, 63)))
    assert saved.exists()
    assert list((tmp_path / "test").glob("*.npz")) == []


def test_short_positive_trial_is_not_saved_or_overwritten(tmp_path):
    existing = save_trial(tmp_path, "train", "p", "THUMB_DOWN", np.ones((8, 63)))
    before = existing.read_bytes()

    with pytest.raises(ValueError, match="at least 8"):
        save_trial(tmp_path, "train", "p", "THUMB_DOWN", np.ones((7, 63)))

    assert existing.read_bytes() == before
    assert len(list((tmp_path / "train").glob("*.npz"))) == 1


def test_neutral_trial_allows_zero_frames(tmp_path):
    path = save_trial(tmp_path, "test", "neutral-session", "NEUTRAL", [])
    with np.load(path, allow_pickle=False) as saved:
        assert saved["features"].shape == (0, 63)
        assert saved["features"].dtype == np.float32
        assert saved["label"].item() == "NEUTRAL"

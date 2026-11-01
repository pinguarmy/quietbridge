from io import BytesIO

import pytest

from hackathon.signs import assets


def test_model_path_override_is_expanded(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", "~/demo/gesture.task")
    assert assets.default_model_path().is_absolute()
    assert assets.default_model_path().name == "gesture.task"


def test_download_verifies_before_writing(monkeypatch, tmp_path):
    path = tmp_path / "gesture.task"
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: BytesIO(b"bad data"))
    with pytest.raises(ValueError, match="checksum"):
        assets.download_model(path)
    assert not path.exists()


def test_existing_file_is_not_overwritten(monkeypatch, tmp_path):
    path = tmp_path / "gesture.task"
    path.write_bytes(b"my local model")
    with pytest.raises(ValueError, match="not overwriting"):
        assets.download_model(path)
    assert path.read_bytes() == b"my local model"

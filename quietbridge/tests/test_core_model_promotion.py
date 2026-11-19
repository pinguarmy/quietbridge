"""Promotion is all-or-nothing: a failed promotion leaves no rollback point."""
import json
from pathlib import Path

import pytest

from quietbridge import model_lifecycle
from quietbridge.tests.test_model_lifecycle import training  # noqa: F401  (fixture)


def _promotable_motion(lifecycle, profile, calls, payload):
    candidate = lifecycle.start_training(profile, "motion")
    Path(candidate["model_path"]).write_bytes(payload)
    Path(candidate["report_path"]).write_text(json.dumps(
        {"validation_trials": 2, "validation_ungated": {"accuracy": 1.0}}))
    calls[-1][2].returncode = 0
    return candidate


def test_failed_promotion_does_not_poison_rollback(training, monkeypatch):  # noqa: F811
    lifecycle, profile, calls = training
    first = _promotable_motion(lifecycle, profile, calls, b"first")
    second = _promotable_motion(lifecycle, profile, calls, b"second")
    third = _promotable_motion(lifecycle, profile, calls, b"third")
    lifecycle.promote(profile, "motion", first["id"])
    lifecycle.promote(profile, "motion", second["id"])
    active = Path(lifecycle.status(profile)["active"]["motion"])
    history = active.parent / "history" / "motion"
    before = sorted(history.iterdir())

    real_copy = model_lifecycle.shutil.copyfile

    def disk_full(source, target):
        Path(target).write_bytes(b"par")  # partial data, then failure
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(model_lifecycle.shutil, "copyfile", disk_full)
    with pytest.raises(OSError):
        lifecycle.promote(profile, "motion", third["id"])
    monkeypatch.setattr(model_lifecycle.shutil, "copyfile", real_copy)

    assert active.read_bytes() == b"second"
    assert sorted(history.iterdir()) == before
    lifecycle.rollback(profile, "motion")
    assert active.read_bytes() == b"first"


def test_promote_refuses_record_pointing_at_another_file(training):  # noqa: F811
    lifecycle, profile, calls = training
    candidate = _promotable_motion(lifecycle, profile, calls, b"evaluated")
    elsewhere = profile.profile_dir / "elsewhere.npz"
    elsewhere.write_bytes(b"other")
    meta = Path(candidate["model_path"]).parent / "candidate.json"
    record = json.loads(meta.read_text())
    record.update(status="completed", model_path=str(elsewhere))
    meta.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not eligible"):
        lifecycle.promote(profile, "motion", candidate["id"])
    assert lifecycle.status(profile)["active"]["motion"] is None

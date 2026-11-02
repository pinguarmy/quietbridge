"""Candidate lifecycle uses fake subprocesses and isolated profile directories."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from hackathon import model_lifecycle
from hackathon.profiles import Profile


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stderr = io.BytesIO(b"")

    def poll(self):
        return self.returncode


@pytest.fixture
def training(monkeypatch, tmp_path):
    calls = []

    def popen(argv, **kwargs):
        process = FakeProcess()
        calls.append((argv, kwargs, process))
        return process

    monkeypatch.setattr(model_lifecycle.subprocess, "Popen", popen)
    profile = Profile(id="alice", name="Alice", root_dir=tmp_path)
    for split in ("train", "validation", "test"):
        for root in (profile.mouths_dir, profile.sequences_dir):
            folder = root / split
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / "trial.npz", session_id=np.array(split), trial_id=np.array(split),
                     split=np.array(split), text=np.array("hello"), rois=np.zeros((4, 96, 96), dtype=np.uint8),
                     label=np.array("HELLO"), landmarks=np.zeros((4, 2, 21, 3)),
                     hand_present=np.ones((4, 2), dtype=bool), timestamps_ms=np.arange(4),
                     handedness=np.array([["Left", "Right"]] * 4))
    return model_lifecycle.ModelLifecycle(), profile, calls


@pytest.mark.parametrize("kind,module,data_attr,suffix", [
    ("motion", "hackathon.signs.train_motion", "sequences_dir", ".npz"),
    ("lips", "hackathon.signs.train_lips", "mouths_dir", ".pth"),
    ("language", "hackathon.signs.train_language", "mouths_dir", ".pth"),
])
def test_trainer_arguments_and_unique_candidates(training, kind, module, data_attr, suffix):
    lifecycle, profile, calls = training
    first = lifecycle.start_training(profile, kind)
    with pytest.raises(ValueError, match="already running"):
        lifecycle.start_training(profile, kind)
    calls[0][2].returncode = 0
    second = lifecycle.start_training(profile, kind)
    assert first["id"] != second["id"]
    argv, kwargs, _ = calls[0]
    assert argv[1:3] == ["-m", module]
    assert argv[argv.index("--data") + 1] == str(getattr(profile, data_attr))
    output = Path(argv[argv.index("--output") + 1])
    assert output == Path(first["model_path"]) and output.suffix == suffix
    assert output.is_relative_to(profile.profile_dir / "models" / "candidates")
    assert not output.exists() and not Path(first["report_path"]).exists()
    assert kwargs.get("shell") is False and "--overwrite" not in argv
    if kind == "motion":
        assert argv[argv.index("--session-prefix") + 1] == ""
    assert len(lifecycle.status(profile)["jobs"]) == 1


@pytest.mark.parametrize("kind,root_attr", [("motion", "sequences_dir"), ("lips", "mouths_dir"), ("language", "mouths_dir")])
def test_preflight_requires_independent_validation(training, kind, root_attr):
    lifecycle, profile, calls = training
    validation = getattr(profile, root_attr) / "validation" / "trial.npz"
    validation.unlink()
    with pytest.raises(ValueError):
        lifecycle.start_training(profile, kind)
    assert not calls


@pytest.mark.parametrize("kind,root_attr", [("motion", "sequences_dir"), ("lips", "mouths_dir"), ("language", "mouths_dir")])
def test_preflight_rejects_session_leakage(training, kind, root_attr):
    lifecycle, profile, calls = training
    root = getattr(profile, root_attr)
    path = root / "validation" / "trial.npz"
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    fields["session_id"] = np.array("train")
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="session"): 
        lifecycle.start_training(profile, kind)
    assert not calls


def test_concurrent_duplicate_requests_start_only_one_process(training):
    lifecycle, profile, calls = training

    def start():
        try:
            return lifecycle.start_training(profile, "motion")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(calls) == 1


def test_completed_candidate_requires_report_and_explicit_promotion_then_rollback(training):
    lifecycle, profile, calls = training
    candidate = lifecycle.start_training(profile, "motion")
    output = Path(candidate["model_path"])
    output.write_bytes(b"candidate")
    calls[0][2].returncode = 0
    with pytest.raises(ValueError):
        lifecycle.promote(profile, "motion", candidate["id"])
    Path(candidate["report_path"]).write_text(json.dumps({"validation_trials": 2, "validation_ungated": {"accuracy": 1.0}}))
    status = lifecycle.status(profile)
    assert status["candidates"][0]["promotable"]
    assert status["active"]["motion"] is None
    lifecycle.promote(profile, "motion", candidate["id"])
    active = Path(lifecycle.status(profile)["active"]["motion"])
    assert active.read_bytes() == b"candidate" and output.read_bytes() == b"candidate"
    lifecycle.rollback(profile, "motion")
    assert not active.exists() and output.read_bytes() == b"candidate"


def test_failed_job_cannot_promote_and_profile_isolation(training):
    lifecycle, profile, calls = training
    candidate = lifecycle.start_training(profile, "lips")
    calls[0][2].returncode = 1
    status = lifecycle.status(profile)
    assert status["candidates"][0]["status"] == "failed"
    assert status["jobs"] == []
    with pytest.raises(ValueError):
        lifecycle.promote(profile, "lips", candidate["id"])
    other = Profile(id="bob", name="Bob", root_dir=profile.root_dir)
    assert lifecycle.status(other)["candidates"] == []
    with pytest.raises(ValueError):
        lifecycle.promote(other, "lips", candidate["id"])

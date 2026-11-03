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


def _setup_valid_language_candidate(training):
    lifecycle, profile, calls = training
    candidate = lifecycle.start_training(profile, "language")
    calls[0][2].returncode = 0
    Path(candidate["model_path"]).write_bytes(b"candidate-lm")
    Path(candidate["report_path"]).write_text(
        json.dumps({"counts": {"validation": 3}, "selection": {"selected": "candidate"}, "saved": True})
    )
    return lifecycle, profile, candidate, calls


def test_language_candidate_without_comparison_not_promotable(training):
    lifecycle, profile, candidate, _ = _setup_valid_language_candidate(training)
    status = lifecycle.status(profile)
    cand_record = status["candidates"][0]
    assert cand_record["status"] == "completed"
    assert cand_record["promotable"] is False
    with pytest.raises(ValueError, match="not eligible for promotion"):
        lifecycle.promote(profile, "language", candidate["id"])


def test_language_evaluation_arguments_and_visual_model_flag(training):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)
    # 1. No promoted lip.pth -> visual model omitted (stock)
    lifecycle.evaluate_language(profile, candidate["id"])
    assert len(calls) == 2
    argv, kwargs, proc = calls[1]
    assert argv[1:3] == ["-m", "hackathon.evaluate_language"]
    assert argv[argv.index("--data") + 1] == str(profile.mouths_dir)
    assert argv[argv.index("--candidate") + 1] == str(candidate["model_path"])
    cand_dir = Path(candidate["model_path"]).parent
    assert argv[argv.index("--output") + 1] == str(cand_dir / "comparison.json")
    assert "--visual-model" not in argv
    assert kwargs.get("shell") is False

    # Complete this eval
    proc.returncode = 0
    comp_file = cand_dir / "comparison.json"
    comp_file.write_text(json.dumps({
        "candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
        "visual_model": "stock", "visual_model_sha256": None,
        "variants": {
            "stock": {"metrics": {"wer": 0.5, "target_wer": 0.2}},
            "candidate": {"metrics": {"wer": 0.3, "target_wer": 0.1}}
        }
    }))
    assert lifecycle.status(profile)["candidates"][0]["promotable"] is True

    # 2. When promoted lip.pth exists -> pass --visual-model
    lip_path = profile.profile_dir / "models" / "lip.pth"
    lip_path.parent.mkdir(parents=True, exist_ok=True)
    lip_path.write_bytes(b"visual-lip")

    second_candidate = lifecycle.start_training(profile, "language")
    calls[2][2].returncode = 0
    Path(second_candidate["model_path"]).write_bytes(b"cand-2")
    Path(second_candidate["report_path"]).write_text(
        json.dumps({"counts": {"validation": 2}, "selection": {"selected": "candidate"}, "saved": True})
    )
    lifecycle.evaluate_language(profile, second_candidate["id"])
    assert len(calls) == 4
    argv_second = calls[3][0]
    assert "--visual-model" in argv_second
    assert argv_second[argv_second.index("--visual-model") + 1] == str(lip_path)


def test_language_evaluation_reuses_current_and_preserves_invalid_report(training):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)
    cand_dir = Path(candidate["model_path"]).parent
    comp_file = cand_dir / "comparison.json"
    comp_file.write_text(json.dumps({
        "candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
        "visual_model": "stock", "visual_model_sha256": None,
        "variants": {
            "stock": {"metrics": {"wer": 0.4, "target_wer": 0.2}},
            "candidate": {"metrics": {"wer": 0.2, "target_wer": 0.1}}
        }
    }))
    # Calling evaluate_language reuses existing valid comparison without launching subprocess
    res = lifecycle.evaluate_language(profile, candidate["id"])
    assert len(calls) == 1  # Only the training call, no new Popen call
    assert res["eval_status"] == "completed"
    assert res["promotable"] is True

    # Now create another candidate with an invalid/corrupted comparison.json
    cand2 = lifecycle.start_training(profile, "language")
    calls[1][2].returncode = 0
    Path(cand2["model_path"]).write_bytes(b"cand-2")
    Path(cand2["report_path"]).write_text(
        json.dumps({"counts": {"validation": 2}, "selection": {"selected": "candidate"}, "saved": True})
    )
    cand2_dir = Path(cand2["model_path"]).parent
    cand2_comp = cand2_dir / "comparison.json"
    cand2_comp.write_text("corrupted json")

    result = lifecycle.evaluate_language(profile, cand2["id"])
    assert len(calls) == 3
    assert cand2_comp.read_text() == "corrupted json"
    assert Path(result["comparison_path"]) != cand2_comp
    assert result["eval_status"] == "running" and not result["promotable"]


def test_language_evaluation_failure_safe_summary(training, monkeypatch):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)

    def popen_fail(argv, **kwargs):
        proc = FakeProcess()
        proc.stderr = io.BytesIO(b"RuntimeError: Out of memory in beam search\n")
        proc.returncode = 1
        calls.append((argv, kwargs, proc))
        return proc

    monkeypatch.setattr(model_lifecycle.subprocess, "Popen", popen_fail)
    lifecycle.evaluate_language(profile, candidate["id"])
    status = lifecycle.status(profile)
    cand = status["candidates"][0]
    assert cand["eval_status"] == "failed"
    assert "RuntimeError (details withheld)" in cand["eval_error"]
    assert cand["promotable"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "0.1"])
def test_invalid_error_rates_never_pass_promotion(value):
    report = {"variants": {"stock": {"metrics": {"wer": .4}},
                           "candidate": {"metrics": {"wer": value}}}}
    assert not model_lifecycle._comparison_gate_passed(report)


def test_missing_candidate_target_rate_does_not_hide_regression():
    report = {"variants": {"stock": {"metrics": {"wer": .4, "target_wer": .2}},
                           "candidate": {"metrics": {"wer": .1}}}}
    assert not model_lifecycle._comparison_gate_passed(report)


@pytest.mark.parametrize("changed", ["candidate", "visual"])
def test_language_comparison_cannot_promote_different_weights(training, changed):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)
    directory = Path(candidate["model_path"]).parent
    model_path = Path(candidate["model_path"])
    report = {"candidate_sha256": model_lifecycle._file_sha256(model_path),
              "visual_model": "stock", "visual_model_sha256": None,
              "variants": {"stock": {"metrics": {"wer": .4}}, "candidate": {"metrics": {"wer": .2}}}}
    comparison = directory / "comparison.json"
    comparison.write_text(json.dumps(report))
    assert lifecycle.status(profile)["candidates"][0]["promotable"]
    target = model_path if changed == "candidate" else profile.profile_dir / "models" / "lip.pth"
    target.write_bytes(b"changed weights")
    described = lifecycle.status(profile)["candidates"][0]
    assert not described["promotable"] and not described["comparison_current"]
    assert "weights" in described["comparison_warning"]
    before = comparison.read_bytes()
    with pytest.raises(ValueError, match="not eligible"):
        lifecycle.promote(profile, "language", candidate["id"])
    reevaluation = lifecycle.evaluate_language(profile, candidate["id"])
    assert reevaluation["eval_status"] == "running"
    assert Path(reevaluation["comparison_path"]) != comparison
    assert comparison.read_bytes() == before
    assert len(calls) == 2
    updated = {**report, "candidate_sha256": model_lifecycle._file_sha256(model_path)}
    if changed == "visual":
        updated.update(visual_model=str(target), visual_model_sha256=model_lifecycle._file_sha256(target))
    fresh_report = Path(reevaluation["comparison_path"])
    fresh_report.write_text(json.dumps(updated))
    calls[-1][2].returncode = 0
    current = lifecycle.status(profile)["candidates"][0]
    assert current["comparison_current"] and current["promotable"]
    assert lifecycle.evaluate_language(profile, candidate["id"])["promotable"]
    assert len(calls) == 2
    assert comparison.read_bytes() == before


def test_comparison_is_not_promotable_before_evaluator_finishes(training):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)
    evaluation = lifecycle.evaluate_language(profile, candidate["id"])
    report = {"candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
              "visual_model": "stock", "visual_model_sha256": None,
              "variants": {"stock": {"metrics": {"wer": .4}}, "candidate": {"metrics": {"wer": .2}}}}
    Path(evaluation["comparison_path"]).write_text(json.dumps(report))
    assert lifecycle.status(profile)["candidates"][0]["eval_status"] == "running"
    assert not lifecycle.status(profile)["candidates"][0]["promotable"]
    calls[-1][2].returncode = 0
    assert lifecycle.status(profile)["candidates"][0]["promotable"]


def test_personal_visual_report_matches_content_not_just_path(training):
    lifecycle, profile, candidate, _ = _setup_valid_language_candidate(training)
    visual = profile.profile_dir / "models" / "lip.pth"
    visual.write_bytes(b"visual v1")
    report = {"candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
              "visual_model": str(visual), "visual_model_sha256": model_lifecycle._file_sha256(visual),
              "variants": {"stock": {"metrics": {"wer": .4}}, "candidate": {"metrics": {"wer": .2}}}}
    (Path(candidate["model_path"]).parent / "comparison.json").write_text(json.dumps(report))
    assert lifecycle.status(profile)["candidates"][0]["promotable"]
    visual.write_bytes(b"visual v2")
    assert not lifecycle.status(profile)["candidates"][0]["promotable"]


def test_language_promotion_gate_wer_and_target_wer(training):
    lifecycle, profile, candidate, calls = _setup_valid_language_candidate(training)
    cand_dir = Path(candidate["model_path"]).parent
    comp_file = cand_dir / "comparison.json"

    # Case 1: Overall WER degraded (candidate > stock) -> not promotable
    comp_file.write_text(json.dumps({
        "candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
        "visual_model": "stock", "visual_model_sha256": None,
        "variants": {
            "stock": {"metrics": {"wer": 0.2, "target_wer": 0.1}},
            "candidate": {"metrics": {"wer": 0.3, "target_wer": 0.05}}
        }
    }))
    assert lifecycle.status(profile)["candidates"][0]["promotable"] is False
    with pytest.raises(ValueError, match="not eligible for promotion"):
        lifecycle.promote(profile, "language", candidate["id"])

    # Case 2: Overall WER passed, but target_wer increased -> not promotable
    comp_file.write_text(json.dumps({
        "candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
        "visual_model": "stock", "visual_model_sha256": None,
        "variants": {
            "stock": {"metrics": {"wer": 0.3, "target_wer": 0.1}},
            "candidate": {"metrics": {"wer": 0.2, "target_wer": 0.2}}
        }
    }))
    assert lifecycle.status(profile)["candidates"][0]["promotable"] is False
    with pytest.raises(ValueError, match="not eligible for promotion"):
        lifecycle.promote(profile, "language", candidate["id"])

    # Case 3: Both WER <= stock and target_wer <= stock -> promotable & promote succeeds
    comp_file.write_text(json.dumps({
        "candidate_sha256": model_lifecycle._file_sha256(Path(candidate["model_path"])),
        "visual_model": "stock", "visual_model_sha256": None,
        "variants": {
            "stock": {"metrics": {"wer": 0.3, "target_wer": 0.1}},
            "candidate": {"metrics": {"wer": 0.25, "target_wer": 0.1}}
        }
    }))
    status = lifecycle.status(profile)
    assert status["candidates"][0]["promotable"] is True
    lifecycle.promote(profile, "language", candidate["id"])
    active_lm = Path(lifecycle.status(profile)["active"]["language"])
    assert active_lm.read_bytes() == b"candidate-lm"

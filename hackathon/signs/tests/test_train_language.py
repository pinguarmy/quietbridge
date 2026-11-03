from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.signs import train_language
from lipflow import train_lm


def make_npz(
    folder: Path,
    name: str,
    text: str,
    session_id: str,
    trial_id: str,
    split: str | None = None,
    with_rois: bool = True,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    kwargs = {
        "text": np.array(text),
        "session_id": np.array(session_id),
        "trial_id": np.array(trial_id),
    }
    if split is not None:
        kwargs["split"] = np.array(split)
    if with_rois:
        kwargs["rois"] = np.zeros((4, 96, 96), dtype=np.uint8)
    np.savez(path, **kwargs)
    return path


def test_load_text_metadata_does_not_load_rois(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    make_npz(train_dir, "t1", "HELLO HOW ARE YOU", "sess-train", "trial-1", split="train", with_rois=True)
    make_npz(val_dir, "v1", "PLEASE WAIT A MOMENT", "sess-val", "trial-2", split="validation", with_rois=True)

    accessed_keys: list[str] = []
    orig_getitem = np.lib.npyio.NpzFile.__getitem__

    def spy_getitem(self, key):
        accessed_keys.append(key)
        return orig_getitem(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy_getitem)

    data = train_language.load_text_metadata(tmp_path)
    assert "rois" not in accessed_keys
    assert len(data["train"]) == 1
    assert data["train"][0]["text"] == "HELLO HOW ARE YOU"
    assert len(data["validation"]) == 1
    assert data["validation"][0]["text"] == "PLEASE WAIT A MOMENT"


@pytest.mark.parametrize("leaked_key", ["session_id", "trial_id"])
def test_cross_split_leakage_rejected(tmp_path, leaked_key):
    train_vals = {"session_id": "train-session", "trial_id": "train-trial"}
    val_vals = {"session_id": "val-session", "trial_id": "val-trial"}
    val_vals[leaked_key] = train_vals[leaked_key]

    make_npz(tmp_path / "train", "t1", "HELLO WORLD", train_vals["session_id"], train_vals["trial_id"], split="train")
    make_npz(tmp_path / "validation", "v1", "HELLO AGAIN", val_vals["session_id"], val_vals["trial_id"], split="validation")

    with pytest.raises(ValueError, match=leaked_key):
        train_language.load_text_metadata(tmp_path)


def test_empty_train_rejected(tmp_path):
    make_npz(tmp_path / "validation", "v1", "HELLO AGAIN", "sess-val", "trial-1", split="validation")
    with pytest.raises(ValueError, match="no training NPZs found"):
        train_language.load_text_metadata(tmp_path)


def test_empty_validation_rejected(tmp_path):
    make_npz(tmp_path / "train", "t1", "HELLO WORLD", "sess-train", "trial-1", split="train")
    with pytest.raises(ValueError, match="no validation NPZs found"):
        train_language.load_text_metadata(tmp_path)


def test_mismatched_split_metadata_rejected(tmp_path):
    make_npz(tmp_path / "train", "t1", "HELLO WORLD", "sess-train", "trial-1", split="validation")
    make_npz(tmp_path / "validation", "v1", "HELLO AGAIN", "sess-val", "trial-2", split="validation")
    with pytest.raises(ValueError, match="split metadata 'validation' does not match directory 'train'"):
        train_language.load_text_metadata(tmp_path)


def test_refuses_to_overwrite_existing_candidate(tmp_path):
    existing_pth = tmp_path / "candidate.pth"
    existing_pth.write_bytes(b"dummy")
    args = argparse.Namespace(
        data=str(tmp_path),
        output=str(existing_pth),
        epochs=1,
        lr=3e-5,
        bs=32,
        seed=0,
        max_general_loss=1.15,
        device=None,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite existing file"):
        train_language.run(args)


def test_refuses_to_overwrite_existing_report(tmp_path):
    target_pth = tmp_path / "candidate.pth"
    report_file = tmp_path / "candidate.report.json"
    report_file.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        data=str(tmp_path),
        output=str(target_pth),
        epochs=1,
        lr=3e-5,
        bs=32,
        seed=0,
        max_general_loss=1.15,
        device=None,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite existing file"):
        train_language.run(args)


def test_refuses_non_pth_output_extension(tmp_path):
    target = tmp_path / "candidate.bin"
    args = argparse.Namespace(
        data=str(tmp_path),
        output=str(target),
        epochs=1,
        lr=3e-5,
        bs=32,
        seed=0,
        max_general_loss=1.15,
        device=None,
    )
    with pytest.raises(ValueError, match="--output must name a new .pth file"):
        train_language.run(args)


def test_successful_candidate_training_creates_pth_and_report(tmp_path, monkeypatch):
    train_dir = tmp_path / "mouths" / "train"
    val_dir = tmp_path / "mouths" / "validation"
    make_npz(train_dir, "t1", "HELLO HOW ARE YOU", "sess-train", "trial-1", split="train")
    make_npz(val_dir, "v1", "PLEASE WAIT A MOMENT", "sess-val", "trial-2", split="validation")

    output_pth = tmp_path / "output" / "candidate.pth"

    def mock_train(**kwargs):
        assert kwargs["output_path"] == output_pth
        assert kwargs["phrases"] == ["HELLO HOW ARE YOU"]
        assert kwargs["validation_phrases"] == ["PLEASE WAIT A MOMENT"]
        output_pth.parent.mkdir(parents=True, exist_ok=True)
        output_pth.write_bytes(b"mock_weights")
        return {
            "sentences": 2,
            "train": 1,
            "val": 1,
            "before": {"yours": 50.0, "general": 40.0},
            "after": {"yours": 30.0, "general": 41.0},
            "saved": True,
            "seconds": 1.23,
        }

    monkeypatch.setattr("lipflow.train_lm.train", mock_train)

    args = argparse.Namespace(
        data=str(tmp_path / "mouths"),
        output=str(output_pth),
        epochs=3,
        lr=3e-5,
        bs=32,
        seed=0,
        max_general_loss=1.15,
        device=None,
    )
    report = train_language.run(args)

    assert output_pth.is_file()
    report_file = output_pth.with_suffix(".report.json")
    assert report_file.is_file()

    loaded_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded_report["saved"] is True
    assert loaded_report["deployment"] is False
    assert loaded_report["auto_deploy"] is False
    assert loaded_report["counts"]["train"] == 1
    assert loaded_report["counts"]["validation"] == 1
    assert loaded_report["perplexity"]["before"]["yours"] == 50.0
    assert loaded_report["perplexity"]["after"]["yours"] == 30.0
    assert loaded_report["selection"]["selected"] == "candidate"
    assert loaded_report["selection"]["deployment"] is False


def test_explicit_corrections_join_language_training_only(tmp_path, monkeypatch):
    from hackathon.corrections import CorrectionStore
    from hackathon.profiles import Profile
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    make_npz(profile.mouths_dir / "train", "t1", "HELLO", "sess-train", "trial-1", split="train")
    make_npz(profile.mouths_dir / "validation", "v1", "WAIT", "sess-val", "trial-2", split="validation")
    CorrectionStore(profile).record(original="HELLO", corrected="HELLO THERE", learn=True)
    output = profile.profile_dir / "models" / "lm.pth"
    captured = {}

    def mock_train(**kwargs):
        captured.update(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"weights")
        return {"train": 2, "val": 1, "before": {}, "after": {}, "saved": True}

    monkeypatch.setattr("lipflow.train_lm.train", mock_train)
    args = argparse.Namespace(data=str(profile.mouths_dir), output=str(output), epochs=1,
                              lr=3e-5, bs=32, seed=0, max_general_loss=1.15, device=None)
    report = train_language.run(args)
    assert captured["phrases"] == ["HELLO", "HELLO THERE"]
    assert captured["validation_phrases"] == ["WAIT"]
    assert report["counts"]["corrections"] == 1
    assert report["counts"]["corrections_skipped"] == 0
    assert report["corrections"]["total"] == 1
    assert report["corrections"]["learned"] == 1
    assert report["corrections"]["provenance_unknown"] is True
    assert report["corrections"]["correction_provenance_complete"] is False
    assert len(report["corrections"]["warnings"]) == 1


def test_heldout_corrections_excluded_from_language_training(tmp_path, monkeypatch):
    from hackathon.profiles import Profile
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    make_npz(profile.mouths_dir / "train", "t1", "HELLO", "sess-train", "trial-1", split="train")
    make_npz(profile.mouths_dir / "validation", "v1", "WAIT", "sess-val", "trial-2", split="validation")

    # Manually add a legacy held-out validation correction entry
    jsonl_path = profile.profile_dir / "corrections.jsonl"
    entry = {
        "id": "e_heldout",
        "profile_id": "alice",
        "original": "WAIT",
        "corrected": "WAIT A MINUTE",
        "learn": True,
        "sources": ["mouths/validation/v1.npz"],
    }
    jsonl_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    output = profile.profile_dir / "models" / "lm.pth"
    captured = {}

    def mock_train(**kwargs):
        captured.update(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"weights")
        return {"train": 1, "val": 1, "before": {}, "after": {}, "saved": True}

    monkeypatch.setattr("lipflow.train_lm.train", mock_train)
    args = argparse.Namespace(data=str(profile.mouths_dir), output=str(output), epochs=1,
                              lr=3e-5, bs=32, seed=0, max_general_loss=1.15, device=None)
    report = train_language.run(args)
    # The heldout phrase must NOT enter training phrases!
    assert "WAIT A MINUTE" not in captured["phrases"]
    assert captured["phrases"] == ["HELLO"]
    assert report["counts"]["corrections"] == 0
    assert report["counts"]["corrections_skipped"] == 1
    assert report["corrections"]["total"] == 1
    assert report["corrections"]["learned"] == 0
    assert report["corrections"]["skipped"] == 1
    assert report["corrections"]["source_coverage"]["heldout_rejected"] == 1
    assert any("held-out" in w for w in report["corrections"]["warnings"])



def test_rejected_candidate_does_not_create_pth_and_records_rejection(tmp_path, monkeypatch):
    train_dir = tmp_path / "mouths" / "train"
    val_dir = tmp_path / "mouths" / "validation"
    make_npz(train_dir, "t1", "HELLO HOW ARE YOU", "sess-train", "trial-1", split="train")
    make_npz(val_dir, "v1", "PLEASE WAIT A MOMENT", "sess-val", "trial-2", split="validation")

    output_pth = tmp_path / "output" / "candidate.pth"

    def mock_train(**kwargs):
        return {
            "sentences": 2,
            "train": 1,
            "val": 1,
            "before": {"yours": 50.0, "general": 40.0},
            "after": {"yours": 50.0, "general": None},
            "saved": False,
            "seconds": 0.5,
        }

    monkeypatch.setattr("lipflow.train_lm.train", mock_train)

    args = argparse.Namespace(
        data=str(tmp_path / "mouths"),
        output=str(output_pth),
        epochs=3,
        lr=3e-5,
        bs=32,
        seed=0,
        max_general_loss=1.15,
        device=None,
    )
    report = train_language.run(args)

    assert not output_pth.exists()  # candidate output must NOT be created
    report_file = output_pth.with_suffix(".report.json")
    assert report_file.is_file()
    loaded_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded_report["saved"] is False
    assert loaded_report["deployment"] is False
    assert loaded_report["auto_deploy"] is False
    assert loaded_report["selection"]["selected"] == "stock"


def test_train_lm_refuses_to_overwrite_existing_output_path(tmp_path):
    existing = tmp_path / "candidate.pth"
    existing.write_bytes(b"existing_weights")
    with pytest.raises(FileExistsError, match="Candidate output path already exists"):
        train_lm.train(output_path=existing)


def test_train_lm_raises_on_empty_phrases():
    with pytest.raises(ValueError, match="No valid training sentences"):
        train_lm.train(phrases=[], validation_phrases=["HELLO HOW ARE YOU"])


def test_train_lm_raises_on_empty_validation_phrases():
    with pytest.raises(ValueError, match="No valid validation sentences"):
        train_lm.train(phrases=["HELLO HOW ARE YOU"], validation_phrases=[])


def test_train_lm_validation_split_separation(tmp_path, monkeypatch):
    """Verify that when validation_phrases is passed, val and tr are kept separate and not randomly split."""
    train_phrases = ["FIRST TRAINING PHRASE", "SECOND TRAINING PHRASE"]
    val_phrases = ["VALIDATION PROMPT ONLY"]

    class FakeLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t):
            return torch.tensor(1.0, requires_grad=True), torch.tensor(1.0), torch.tensor(1)

    fake_lm = FakeLM()
    monkeypatch.setattr("espnet.asr.asr_utils.get_model_conf", lambda *a: argparse.Namespace(backend="pytorch"))
    monkeypatch.setattr("espnet.nets.lm_interface.dynamic_import_lm", lambda *a: lambda *args, **kwargs: fake_lm)
    monkeypatch.setattr("espnet.asr.asr_utils.torch_load", lambda *a: None)

    call_count = 0

    def fake_perplexity(lm, seqs, eos, device):
        nonlocal call_count
        call_count += 1
        return 10.0 if call_count > 2 else 50.0

    monkeypatch.setattr("lipflow.train_lm.perplexity", fake_perplexity)

    out_file = tmp_path / "new_lm.pth"
    res = train_lm.train(
        epochs=1,
        phrases=train_phrases,
        validation_phrases=val_phrases,
        output_path=out_file,
        device="cpu",
    )

    assert res["train"] == 2
    assert res["val"] == 1
    assert res["sentences"] == 3
    assert res["saved"] is True
    assert out_file.is_file()


def test_train_lm_forgets_guard_rejects_saving(tmp_path, monkeypatch):
    """Verify that general sample regression prevents saving."""
    class FakeLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t):
            return torch.tensor(1.0, requires_grad=True), torch.tensor(1.0), torch.tensor(1)

    fake_lm = FakeLM()
    monkeypatch.setattr("espnet.asr.asr_utils.get_model_conf", lambda *a: argparse.Namespace(backend="pytorch"))
    monkeypatch.setattr("espnet.nets.lm_interface.dynamic_import_lm", lambda *a: lambda *args, **kwargs: fake_lm)
    monkeypatch.setattr("espnet.asr.asr_utils.torch_load", lambda *a: None)

    step = 0

    def fake_perplexity(lm, seqs, eos, device):
        nonlocal step
        step += 1
        if step == 1:
            return 50.0
        elif step == 2:
            return 40.0
        elif step == 3:
            return 10.0
        else:
            return 100.0  # general text regressed severely

    monkeypatch.setattr("lipflow.train_lm.perplexity", fake_perplexity)

    out_file = tmp_path / "regressed.pth"
    res = train_lm.train(
        epochs=1,
        phrases=["TRAINING SENTENCE HERE"],
        validation_phrases=["VALIDATION SENTENCE HERE"],
        output_path=out_file,
        device="cpu",
    )

    assert res["saved"] is False
    assert not out_file.exists()


def test_train_lm_default_preserves_old_signature_and_missing_phrases_error(tmp_path, monkeypatch):
    monkeypatch.setattr("lipflow.train_lm.PHRASES", str(tmp_path / "nonexistent_phrases.txt"))
    with pytest.raises(FileNotFoundError, match="No phrases yet. Run `lipflow import-wispr` first."):
        train_lm.train()


def test_main_cli(tmp_path, monkeypatch, capsys):
    train_dir = tmp_path / "mouths" / "train"
    val_dir = tmp_path / "mouths" / "validation"
    make_npz(train_dir, "t1", "HELLO HOW ARE YOU", "sess-train", "trial-1", split="train")
    make_npz(val_dir, "v1", "PLEASE WAIT A MOMENT", "sess-val", "trial-2", split="validation")

    output_pth = tmp_path / "candidate.pth"
    monkeypatch.setattr("lipflow.train_lm.train", lambda **kw: {
        "sentences": 2, "train": 1, "val": 1,
        "before": {"yours": 40.0, "general": 30.0},
        "after": {"yours": 25.0, "general": 30.5},
        "saved": True, "seconds": 0.1,
    })

    train_language.main(["--data", str(tmp_path / "mouths"), "--output", str(output_pth)])
    captured = capsys.readouterr()
    cli_out = json.loads(captured.out)
    assert cli_out["saved"] is True
    assert cli_out["auto_deploy"] is False
    assert cli_out["output"] == str(output_pth)

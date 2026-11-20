"""train_lips run() contracts with fake readers: data minimums, ordering, hashes, experiments."""
from __future__ import annotations

import argparse
import hashlib
import json

import pytest
import torch

from quietbridge.signs import train_lips
from quietbridge.signs.tests.test_train_lips import save_trial


def dataset(root, n_train=12, n_val=6, n_test=4):
    phrases = ["HELLO THERE", "PLEASE WAIT", "I NEED HELP"]
    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        for i in range(n):
            save_trial(root, split, f"{split}-{i}", phrases[i % 3], f"{split}-session", f"{split}-{i}")


def args(tmp_path, **overrides):
    values = dict(data=str(tmp_path / "data"), output=str(tmp_path / "out" / "lip.pth"), epochs=1, beam=1,
                  lr=1e-4, scope="frontend", device="cpu", seed=3, no_final_test=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class Events(list):
    pass


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    events = Events()
    tokenizer = tmp_path / "tok.model"
    tokenizer.write_bytes(b"tok")
    monkeypatch.setattr(train_lips, "TOKENIZER", tokenizer)

    class Reader:
        def __init__(self, *a, name="stock", **k):
            self.name = name
            self.model = torch.nn.Module()
            self.model.register_parameter("w", torch.nn.Parameter(torch.zeros(2)))
            events.append(("load", name))

        def read(self, rois, fast=False):
            events.append(("decode", self.name, self._split))
            return ("HELLO THERE" if self.name == "candidate" else "WRONG"), 0.01

    class EncoderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.frontend = torch.nn.Linear(2, 2)

    def finetune(fit, epochs, lr, scope, reader, seed):
        assert all(item["split"] == "train" for item in fit)
        events.append(("finetune", len(fit), seed))
        candidate = Reader(name="candidate")
        candidate.model = EncoderModel()
        return candidate

    original_evaluate = train_lips.evaluate

    def evaluate(reader, items, cache=None):
        reader._split = items[0]["split"] if items else None
        return original_evaluate(reader, items, cache)

    import lipflow.train_vsr
    import lipflow.vsr
    monkeypatch.setattr(lipflow.vsr, "LipReader", Reader)
    monkeypatch.setattr(lipflow.train_vsr, "finetune", finetune)
    monkeypatch.setattr(train_lips, "evaluate", evaluate)
    return events


def test_too_little_data_stops_before_loading_the_model_and_says_what_to_record(tmp_path, fakes):
    dataset(tmp_path / "data", n_train=4, n_val=2)
    with pytest.raises(train_lips.InsufficientData) as error:
        train_lips.run(args(tmp_path))
    assert "record 6 more lip practice tries" in str(error.value)
    assert "record 3 more lip check tries" in str(error.value)
    assert fakes == []  # no model was loaded
    report = json.loads((tmp_path / "out" / "lip.report.json").read_text())
    assert report["failures"] and report["data_requirements"]["minimums"] == {
        "fit_clips": train_lips.MIN_FIT_CLIPS, "validation_clips": train_lips.MIN_VALIDATION_CLIPS}
    assert not (tmp_path / "out" / "lip.pth").exists()


def test_selection_is_frozen_before_the_single_final_test_pass_and_report_binds_weights(tmp_path, fakes):
    dataset(tmp_path / "data")
    report = train_lips.run(args(tmp_path))
    decodes = [e for e in fakes if e[0] == "decode"]
    first_test = next(i for i, e in enumerate(fakes) if e[0] == "decode" and e[2] == "test")
    assert all(e[0] != "decode" or e[2] != "test" for e in fakes[:first_test])
    selection_point = fakes.index(("load", "stock"), 1)  # fresh stock reader for test after selection
    assert selection_point < first_test
    assert ("finetune", 12, 3) in fakes[:first_test]  # seed reaches the fine-tune
    assert sum(1 for e in decodes if e[1] == "candidate" and e[2] == "test") == 4  # once per test clip
    assert report["selection"]["selected"] == "candidate"
    output = tmp_path / "out" / "lip.pth"
    assert report["candidate_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["config"]["seed"] == 3 and "train clips only" in report["config"]["augmentation"]
    assert report["data_requirements"]["fit_clips"] == 12
    assert report["capture_quality"]["train"]["checked"] == 12
    assert report["tokenizer_sha256"] == hashlib.sha256(b"tok").hexdigest()


def test_experiment_runs_skip_the_final_test_and_are_marked(tmp_path, fakes):
    dataset(tmp_path / "data")
    report = train_lips.run(args(tmp_path, no_final_test=True))
    assert not any(e[0] == "decode" and e[2] == "test" for e in fakes)
    assert report["selection"]["experiment_only"] is True
    assert "test" not in report["candidate"] and "not evaluated" in report["final_test"]


def test_cli_exposes_seed_and_experiment_flags(tmp_path, fakes, capsys):
    dataset(tmp_path / "data")
    train_lips.main(["--data", str(tmp_path / "data"), "--output", str(tmp_path / "out" / "lip.pth"),
                     "--epochs", "1", "--seed", "9", "--no-final-test"])
    report = json.loads((tmp_path / "out" / "lip.report.json").read_text())
    assert report["config"]["seed"] == 9 and report["selection"]["experiment_only"] is True


def test_recipe_experiment_chooses_on_validation_and_refuses_reports_with_test(tmp_path):
    from quietbridge.signs import lip_recipe_experiment as exp

    def write(name, wer, selected="candidate", test=False):
        folder = tmp_path / name
        folder.mkdir()
        candidate = {"val": {"metrics": {"wer": wer, "wer_words": 40}}}
        if test:
            candidate["test"] = {"metrics": {"wer": 0.0}}
        (folder / "lip.report.json").write_text(json.dumps({
            "data": "d", "failures": [], "selection": {"selected": selected, "experiment_only": True},
            "candidate": candidate, "baseline": {"val": {"metrics": {"wer": 0.5}}}}))

    write("current_default", 0.40)
    write("frontend_6ep", 0.35)
    write("frontend_encoder1_6ep", 0.30, selected="stock")  # lower WER but failed the gate
    write("frontend_encoder1_3ep", 0.35)
    summary = exp.summarize(tmp_path)
    assert summary["chosen"] == "frontend_6ep"  # tie with a later config keeps the earlier one
    assert any(r["status"] == "missing" for r in summary["rows"])
    write("frontend_6ep_lr5e-5", 0.10, test=True)
    with pytest.raises(ValueError, match="no-final-test"):
        exp.summarize(tmp_path)
    cmds = [exp._command("d", tmp_path / "o.pth", c, 0, final=False) for c in exp.CONFIGS]
    assert all("--no-final-test" in c for c in cmds)
    assert "--no-final-test" not in exp._command("d", tmp_path / "o.pth", exp.CONFIGS[0], 0, final=True)

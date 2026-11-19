from pathlib import Path

import numpy as np
import pytest

from quietbridge.signs import train_lips


def save_trial(root: Path, split: str, name: str, text: str, session_id: str, trial_id: str):
    folder = root / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    np.savez(path, rois=np.zeros((4, 96, 96), dtype=np.uint8), text=np.array(text),
             session_id=np.array(session_id), trial_id=np.array(trial_id), split=np.array(split))
    return path


def test_load_trial_metadata_without_pickle(tmp_path):
    path = save_trial(tmp_path, "train", "a", "HI WELCOME", "train-session", "trial-a")
    data = train_lips.load_dataset(tmp_path)
    item = data["train"][0]
    assert item["path"] == str(path)
    assert item["text"] == "HI WELCOME"
    assert item["session_id"] == "train-session"
    assert item["trial_id"] == "trial-a"
    assert item["rois"].shape == (4, 96, 96)
    assert item["rois"].dtype == np.uint8


@pytest.mark.parametrize("leaked_key", ["session_id", "trial_id"])
def test_refuses_cross_split_leakage(tmp_path, leaked_key):
    train_values = {"session_id": "train-session", "trial_id": "train-trial"}
    test_values = {"session_id": "test-session", "trial_id": "test-trial"}
    test_values[leaked_key] = train_values[leaked_key]
    save_trial(tmp_path, "train", "a", "TARGET", **train_values)
    save_trial(tmp_path, "test", "b", "CONTRAST", **test_values)
    with pytest.raises(ValueError, match=leaked_key):
        train_lips.load_dataset(tmp_path)


def test_partition_is_path_sorted_and_holds_out_one_per_prompt(tmp_path):
    for prompt in ("FULL WELCOME PHRASE", "SHORT CONTRAST"):
        for name in ("c", "a", "b"):
            save_trial(tmp_path, "train", f"{prompt.replace(' ', '_')}_{name}", prompt,
                       f"session-{name}", f"{prompt}-{name}")
    fit, val = train_lips.partition_train(train_lips.load_dataset(tmp_path)["train"])
    assert len(fit) == 4
    assert len(val) == 2
    assert all(Path(x["path"]).stem.endswith("_c") for x in val)
    assert len({x["text"] for x in val}) == 2


def test_candidate_gate_requires_wer_gain_and_no_false_welcome_rise():
    stock = {"wer": 0.5, "contrast_false_welcome": 0.1}
    better = {"wer": 0.4, "contrast_false_welcome": 0.1}
    assert train_lips.select_candidate(stock, better)[0]
    assert not train_lips.select_candidate(stock, {"wer": 0.5, "contrast_false_welcome": 0.1})[0]
    assert not train_lips.select_candidate(stock, {"wer": 0.3, "contrast_false_welcome": 0.2})[0]


def test_metrics_include_raw_wer_exact_phrase_and_contrast_false_welcome():
    target = "HI WELCOME TO OPENAI DDE COMMUNITY HACKATHON"
    rows = [
        {"text": target, "hypothesis": target},
        {"text": "A DIFFERENT SENTENCE", "hypothesis": target},
    ]
    result = train_lips.metrics(rows, target)
    assert result["count"] == 2
    assert result["fullphrase_exact"] == 1.0
    assert result["contrast_false_welcome"] == 1.0
    assert result["target_wer"] == 0.0
    assert result["wer_words"] > 0


def test_validation_is_loaded_and_cannot_reuse_train_session(tmp_path):
    save_trial(tmp_path, "train", "train", "TARGET", "train-session", "train-trial")
    save_trial(tmp_path, "validation", "val", "TARGET", "validation-session", "validation-trial")
    data = train_lips.load_dataset(tmp_path)
    assert len(data["validation"]) == 1
    save_trial(tmp_path, "test", "test", "CONTRAST", "validation-session", "test-trial")
    with pytest.raises(ValueError, match="validation and test"):
        train_lips.load_dataset(tmp_path)


def test_training_requires_an_independent_validation_directory(tmp_path):
    save_trial(tmp_path, "train", "train", "TARGET", "train-session", "train-trial")
    data = train_lips.load_dataset(tmp_path)
    with pytest.raises(ValueError, match="Independent validation"):
        train_lips.independent_partition(data)
    save_trial(tmp_path, "validation", "val", "TARGET", "validation-session", "validation-trial")
    fit, val = train_lips.independent_partition(train_lips.load_dataset(tmp_path))
    assert len(fit) == len(val) == 1


def test_candidate_gate_rejects_target_regression_despite_overall_gain():
    stock = {"wer": 0.5, "target_wer": 0.4, "contrast_false_welcome": 0.0}
    candidate = {"wer": 0.4, "target_wer": 0.6, "contrast_false_welcome": 0.0}
    assert train_lips.select_candidate(stock, candidate) == (
        False, "rejected: target-prompt WER regressed")


def test_rates_without_a_denominator_are_null_not_zero():
    target = "PLEASE CALL MY SISTER"
    # Only target-prompt rows: there were no contrast prompts to falsely match.
    only_target = train_lips.metrics([{"text": target, "hypothesis": target}], target)
    assert only_target["contrast_false_welcome"] is None
    assert only_target["contrast_false_welcome_counts"] == [0, 0]
    # Only contrast rows: the full-phrase rate was not measured.
    only_contrast = train_lips.metrics([{"text": "GOOD MORNING", "hypothesis": "GOOD MORNING"}], target)
    assert only_contrast["fullphrase_exact"] is None
    # No rows at all (for example an empty final-test split) is not a perfect 0 WER.
    empty = train_lips.metrics([], target)
    assert empty["wer"] is None and empty["fullphrase_exact"] is None
    assert empty["contrast_false_welcome"] is None


def test_selection_rejects_unscorable_validation_and_skips_unmeasured_contrast():
    assert train_lips.select_candidate({"wer": None}, {"wer": None}) == (
        False, "rejected: validation has no scorable words")
    stock = {"wer": 0.5, "target_wer": 0.4, "contrast_false_welcome": None}
    better = {"wer": 0.4, "target_wer": 0.3, "contrast_false_welcome": None}
    assert train_lips.select_candidate(stock, better)[0] is True


def test_copied_recording_in_one_split_is_rejected(tmp_path):
    save_trial(tmp_path, "train", "a", "HELLO THERE", "s-train", "trial-a")
    save_trial(tmp_path, "train", "a-copy", "HELLO THERE", "s-train", "trial-a")
    with pytest.raises(ValueError, match="appears twice in train"):
        train_lips.load_dataset(tmp_path)

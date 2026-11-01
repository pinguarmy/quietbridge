from pathlib import Path

import numpy as np
import pytest

from hackathon.signs import train_lips


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
    assert result["wer_words"] > 0

"""Regression checks for honest replay metrics and labeled inputs."""
import json

import numpy as np

from lipflow import tune


def test_missing_prediction_never_uses_gold_as_candidate(tmp_path):
    np.savez_compressed(tmp_path / "no-raw.npz", rois=np.zeros((15, 96, 96), dtype=np.uint8),
                        text="A known reference that has not been predicted", raw="")
    items = tune.load_dataset(str(tmp_path))
    assert items == [] or items[0]["candidates"] != [items[0]["reference"].upper()]


def test_history_generated_text_is_not_ground_truth(tmp_path):
    log = tmp_path / "history.jsonl"
    log.write_text(json.dumps({"raw": ["A WRONG GUESS"], "text": "An invented output"}) + "\n")
    items = tune.load_dataset(str(log))
    assert not items or items[0]["reference"] != "An invented output"


def test_semantic_contraction_does_not_flag_meaning_change():
    assert not tune.detect_semantic_risks("I do not want this", "I don't want this")["negation_risk"]


def test_sentence_capitalization_is_not_a_missing_entity():
    assert not tune.detect_semantic_risks("Please send the document", "Could you please send the document?")["entity_risk"]

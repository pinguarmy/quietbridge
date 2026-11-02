"""Tests for motion evaluation reports, using controlled predictions."""

import pytest

from hackathon.signs.motion import REST_LABEL
from hackathon.signs.train_motion import evaluate


class Predictions:
    def predict(self, trial):
        return trial["prediction"]


def test_evaluation_separates_positive_recall_rest_and_mislabels():
    trials = [
        {"label": "HELLO", "prediction": ("HELLO", 0.9)},
        {"label": "HELLO", "prediction": ("WELCOME", 0.9)},
        {"label": "WELCOME", "prediction": (None, 0.0)},
        {"label": REST_LABEL, "prediction": (None, 0.0)},
        {"label": REST_LABEL, "prediction": ("HELLO", 0.7)},
    ]
    result = evaluate(Predictions(), trials)
    assert result["accuracy"] == pytest.approx(2 / 5)
    assert result["false_positives"] == 2  # REST emission plus positive mislabel
    assert result["misses"] == 1
    assert result["mislabels"] == 1
    assert (result["positive_correct"], result["positive_trials"]) == (1, 3)
    assert result["positive_recall"] == pytest.approx(1 / 3)
    assert result["rest_silent_rate"] == pytest.approx(1 / 2)
    assert result["confusion"]["HELLO -> WELCOME"] == 1

    gated = evaluate(Predictions(), trials, score_gate=0.75)
    assert gated["false_positives"] == 1
    assert gated["mislabels"] == 1
    assert gated["positive_recall"] == pytest.approx(1 / 3)
    assert gated["rest_silent_rate"] == 1
    assert gated["accuracy"] == pytest.approx(3 / 5)


def test_missing_classes_have_undefined_rates():
    empty = evaluate(Predictions(), [])
    assert empty["positive_recall"] is None
    assert empty["rest_silent_rate"] is None
    positive = evaluate(Predictions(), [{"label": "HELLO", "prediction": ("HELLO", 0.75)}], 0.75)
    assert positive["positive_recall"] == 1
    assert positive["rest_silent_rate"] is None
    rest = evaluate(Predictions(), [{"label": REST_LABEL, "prediction": (None, 0)}])
    assert rest["positive_recall"] is None
    assert rest["rest_silent_rate"] == 1

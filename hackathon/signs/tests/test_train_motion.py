"""Tests for motion evaluation reports, using controlled predictions."""

import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.signs.motion import MotionModel, REST_LABEL
from hackathon.signs.tests.test_motion import make_trial
from hackathon.signs.train_motion import evaluate, main


class Predictions:
    def predict(self, trial):
        return trial["prediction"]


def _save_npz(path: Path, trial: dict):
    np.savez(
        path,
        label=np.array(trial["label"]),
        landmarks=np.asarray(trial["landmarks"], dtype=np.float64),
        hand_present=np.asarray(trial["hand_present"], dtype=bool),
        timestamps_ms=np.asarray(trial["timestamps_ms"], dtype=np.float64),
        handedness=np.asarray(trial["handedness"]).astype(str),
        session_id=np.array(trial["session_id"]),
        trial_id=np.array(trial["trial_id"]),
    )


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


def test_train_motion_fixed_order_and_report_structure(tmp_path):
    data_dir = tmp_path / "data"
    for split in ("train", "validation", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    labels = ["HELLO", "WELCOME", REST_LABEL]
    for l in labels:
        for s in range(3):
            t = make_trial(l, seed=s, session="peter-motion-v2-train", trial_id=f"train-{l}-{s}")
            _save_npz(data_dir / "train" / f"{l}-{s}.npz", t)
        for s in range(2):
            t = make_trial(l, seed=10 + s, session="peter-motion-v2-val", trial_id=f"val-{l}-{s}")
            _save_npz(data_dir / "validation" / f"{l}-{s}.npz", t)
        for s in range(2):
            t = make_trial(l, seed=20 + s, session="peter-motion-v2-test", trial_id=f"test-{l}-{s}")
            _save_npz(data_dir / "test" / f"{l}-{s}.npz", t)

    output = tmp_path / "model.npz"
    ret = main(["--data", str(data_dir), "--output", str(output)])
    assert ret == 0
    assert output.exists()

    report_file = tmp_path / "model.report.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text())

    # Check distinct sections for train LOO, validation pre/post calibration, and final test
    assert "loo" in report
    assert "reject_threshold" in report["loo"]
    assert "margin_threshold" in report["loo"]

    assert "calibration" in report
    assert report["calibration"]["calibrated"] is True
    assert "coverage_rate" in report["calibration"]
    assert "calibrated_labels" in report["calibration"]

    assert "validation_pre_calibration" in report
    assert "positive_recall" in report["validation_pre_calibration"]
    assert "rest_silent_rate" in report["validation_pre_calibration"]
    assert "mislabels" in report["validation_pre_calibration"]

    assert "validation_post_calibration" in report
    assert "positive_recall" in report["validation_post_calibration"]
    assert "rest_silent_rate" in report["validation_post_calibration"]
    assert "mislabels" in report["validation_post_calibration"]

    assert "validation_pre_calibration_diagnostics" in report
    assert len(report["validation_pre_calibration_diagnostics"]) == 6
    assert "validation_post_calibration_diagnostics" in report
    assert len(report["validation_post_calibration_diagnostics"]) == 6

    assert "test_ungated" in report
    assert "test_diagnostics" in report
    assert len(report["test_diagnostics"]) == 6

    loaded = MotionModel.load(output)
    assert loaded.label_thresholds_ is not None
    assert loaded.calibration_report_["calibrated"] is True


def test_train_motion_legacy_no_validation(tmp_path):
    data_dir = tmp_path / "data"
    for split in ("train", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    labels = ["HELLO", REST_LABEL]
    for l in labels:
        for s in range(3):
            t = make_trial(l, seed=s, session="peter-motion-v2-train", trial_id=f"train-{l}-{s}")
            _save_npz(data_dir / "train" / f"{l}-{s}.npz", t)
        t_test = make_trial(l, seed=20, session="peter-motion-v2-test", trial_id=f"test-{l}")
        _save_npz(data_dir / "test" / f"{l}.npz", t_test)

    output = tmp_path / "legacy_model.npz"
    ret = main(["--data", str(data_dir), "--output", str(output), "--allow-legacy-no-validation"])
    assert ret == 0
    report = json.loads((tmp_path / "legacy_model.report.json").read_text())
    assert report["validation_pre_calibration"] is None
    assert report["validation_post_calibration"] is None
    assert report["test_ungated"] is not None
    assert report["thresholds_calibrated_on"] == "train leave-one-trial-out only"

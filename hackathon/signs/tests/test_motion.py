"""Tests for the personal motion-v2 dynamic gesture model (synthetic data)."""

import json
import math

import numpy as np
import pytest

from hackathon.signs.motion import (
    MotionModel,
    MotionModelError,
    REST_LABEL,
    check_session_leakage,
    masked_distance,
    preprocess_trial,
)

N = 40
DT_MS = 66.0
LABELS = ["HELLO", "WELCOME", "OPENAI", "DDE", "HACKATHON", REST_LABEL]


def _hand_block(center, spread=0.12):
    """21 landmarks: wrist at center, middle MCP 0.25 away (scale anchor)."""
    rng_offsets = np.linspace(0, spread, 21)
    lm = np.zeros((21, 3))
    lm[:, 0] = center[0] + rng_offsets * 0.3
    lm[:, 1] = center[1] + rng_offsets
    lm[:, 2] = center[2]
    lm[0] = center                      # wrist exactly at center
    lm[9] = center + np.array([0.0, 0.25, 0.0])  # hand scale = 0.25
    return lm


def _trajectory(kind, n=N):
    t = np.linspace(0.0, 1.0, n)
    base = np.array([0.4, 0.6, 0.0])
    moves = {
        "up": np.stack([np.zeros(n), -0.4 * t, np.zeros(n)], axis=1),
        "down": np.stack([np.zeros(n), 0.4 * t, np.zeros(n)], axis=1),
        "left": np.stack([-0.4 * t, np.zeros(n), np.zeros(n)], axis=1),
        "right": np.stack([0.4 * t, np.zeros(n), np.zeros(n)], axis=1),
        "circle": np.stack([0.2 * np.sin(2 * np.pi * t),
                            0.2 * (1 - np.cos(2 * np.pi * t)),
                            np.zeros(n)], axis=1),
        "rest": np.zeros((n, 3)),
    }
    return base + moves[kind]


_KIND = dict(zip(LABELS, ["up", "down", "circle", "left", "right", "rest"]))


def make_trial(label, seed=0, n=N, session="sess-train", trial_id=None):
    rng = np.random.default_rng(seed)
    traj = _trajectory(_KIND[label], n)
    lm = np.full((n, 2, 21, 3), np.nan)
    hp = np.zeros((n, 2), dtype=bool)
    handed = np.full((n, 2), "", dtype="U5")
    noise = rng.normal(0, 0.005, size=(n, 21, 3))
    for f in range(n):
        lm[f, 0] = _hand_block(traj[f]) + noise[f]
        hp[f, 0] = True
        handed[f, 0] = "Left"
    return {
        "label": label,
        "landmarks": lm,
        "hand_present": hp,
        "timestamps_ms": np.arange(n) * DT_MS,
        "handedness": handed,
        "session_id": session,
        "trial_id": trial_id or f"{label}-{seed}",
    }


def make_train():
    return [make_trial(label, seed=s) for label in LABELS for s in range(4)]


def test_fit_predict_held_out():
    model = MotionModel().fit(make_train())
    for label in LABELS:
        pred, score = model.predict(make_trial(label, seed=100, session="sess-new"))
        if label == REST_LABEL:
            assert pred is None and score == 0.0  # REST must not produce output
        else:
            assert pred == label
            assert 0.0 < score <= 1.0


def test_temporal_order_matters():
    model = MotionModel().fit(make_train())
    trial = make_trial("HELLO", seed=7, session="sess-new")
    _, d1_normal, _ = model._match(trial)
    reversed_trial = dict(trial)
    idx = np.arange(N)[::-1]
    reversed_trial["landmarks"] = trial["landmarks"][idx]
    reversed_trial["hand_present"] = trial["hand_present"][idx]
    reversed_trial["handedness"] = trial["handedness"][idx]
    _, d1_reversed, _ = model._match(reversed_trial)
    assert d1_reversed > d1_normal * 2  # time-reversed motion is a worse match


def test_dropout_mask_interpolates_and_still_matches():
    model = MotionModel().fit(make_train())
    trial = make_trial("WELCOME", seed=11, session="sess-new")
    drop = slice(12, 28)  # long mid-sequence dropout for both hands
    trial["hand_present"][drop] = False
    trial["landmarks"][drop] = np.nan
    trial["handedness"][drop] = ""
    feat, mask = preprocess_trial(trial)
    assert mask[:8, 0].all() and mask[-8:, 0].all()
    assert not mask[12:20, 0].any()  # interior dropout remains missing
    assert 0.5 <= model.diagnose(trial)["joint_coverage"] < 0.7
    pred, score = model.predict(trial)
    assert pred == "WELCOME"


def test_out_of_distribution_rejected():
    model = MotionModel().fit(make_train())
    rng = np.random.default_rng(0)
    n = N
    ood = {
        "landmarks": rng.uniform(0, 1, size=(n, 2, 21, 3)),
        "hand_present": np.ones((n, 2), dtype=bool),
        "timestamps_ms": np.arange(n) * DT_MS,
        "handedness": np.tile(np.array(["Left", "Right"], dtype="U5"), (n, 1)),
        "session_id": "sess-ood",
        "trial_id": "ood-0",
    }
    pred, score = model.predict(ood)
    assert pred is None and score == 0.0


def test_save_load_roundtrip(tmp_path):
    model = MotionModel().fit(make_train())
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = MotionModel.load(path)
    for label in LABELS:
        trial = make_trial(label, seed=55, session="sess-new")
        assert loaded.predict(trial) == model.predict(trial)


def test_session_leakage_detected():
    train = [make_trial("HELLO", seed=1, session="shared")]
    test = [make_trial("HELLO", seed=2, session="shared")]
    with pytest.raises(MotionModelError):
        check_session_leakage(train, test)
    check_session_leakage(train, [make_trial("HELLO", seed=3, session="other")])


@pytest.mark.parametrize("label", ["", "  ", " hello ", None, 42])
def test_invalid_label_rejected(label):
    bad = make_trial("HELLO", seed=1)
    bad["label"] = label
    with pytest.raises(MotionModelError):
        MotionModel().fit([bad])


def test_custom_label_roundtrip(tmp_path):
    trials = [make_trial("HELLO", seed=i) for i in range(4)]
    for trial in trials:
        trial["label"] = "My gesture"
    model = MotionModel().fit(trials + [make_trial("WELCOME", seed=i) for i in range(4)])
    path = tmp_path / "custom.npz"
    model.save(path)
    assert MotionModel.load(path).predict(trials[0])[0] == "My gesture"


def test_interior_dropout_below_coverage_rejected():
    model = MotionModel().fit(make_train())
    trial = make_trial("HELLO", seed=0)
    trial["hand_present"][5:35] = False
    trial["landmarks"][5:35] = np.nan
    _, mask = preprocess_trial(trial)
    assert not mask[6:26, 0].any()
    detail = model.diagnose(trial)
    assert detail["joint_coverage"] < 0.5
    assert detail["reject_reason"] == "insufficient_coverage"
    assert model.predict(trial) == (None, 0.0)


def test_masked_distance_no_overlap_is_inf():
    trial = make_trial("HELLO", seed=1)
    feat, mask = preprocess_trial(trial)
    empty = np.zeros_like(mask)
    assert masked_distance(feat, mask, feat, empty) == math.inf


def test_joint_coverage_threshold_and_diagnostics(tmp_path):
    model = MotionModel().fit(make_train())
    trial = make_trial("HELLO", seed=0)
    detail = model.diagnose(trial)
    assert detail["raw_label"] == "HELLO"
    assert detail["accepted"] and detail["reject_reason"] is None
    assert detail["joint_coverage"] == 1.0
    assert model.classify(trial) == (detail["raw_label"], detail["score"], True)

    feat, mask = preprocess_trial(trial)
    half = mask.copy()
    half[16:] = False
    assert masked_distance(feat, mask, feat, half) == 0.0
    half[15] = False
    assert masked_distance(feat, mask, feat, half) == math.inf

    sparse = make_trial("HELLO", seed=0)
    sparse["hand_present"][10:] = False
    sparse["landmarks"][10:] = np.nan
    result = model.diagnose(sparse)
    assert result["raw_label"] is None
    assert result["reject_reason"] == "insufficient_coverage"
    assert result["joint_coverage"] < 0.5
    assert model.predict(sparse) == (None, 0.0)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = MotionModel.load(path)
    assert loaded.reject_threshold_ == model.reject_threshold_
    assert loaded.margin_threshold_ == model.margin_threshold_
    assert loaded.diagnose(sparse) == result


def test_diagnose_rejected_distance_and_rest():
    model = MotionModel().fit(make_train())
    trial = make_trial("HELLO", seed=0)
    trial["landmarks"][:, 0, :, 0] += 10
    result = model.diagnose(trial)
    assert result["reject_reason"] == "distance"
    assert not result["accepted"]
    rest = model.diagnose(make_trial(REST_LABEL, seed=0))
    assert rest["raw_label"] == REST_LABEL and rest["accepted"]
    assert model.predict(make_trial(REST_LABEL, seed=0)) == (None, 0.0)


def test_calibrate_on_validation_tightens_to_eliminate_false_alarms():
    model = MotionModel().fit(make_train())
    # Normal validation trials
    val_hello_1 = make_trial("HELLO", seed=50, session="sess-val")
    val_hello_2 = make_trial("HELLO", seed=51, session="sess-val")

    # Construct an intruder: true REST, but landmarks perturbed from HELLO
    # so distance is close enough to be accepted under train LOO threshold (~0.45)
    intruder = make_trial("HELLO", seed=52, session="sess-val")
    intruder["label"] = REST_LABEL
    intruder["trial_id"] = "rest-intruder"
    intruder["landmarks"][:, 0, :, 0] += 0.007  # distance ~0.027, within train LOO threshold (0.0296)

    # Before calibration: intruder is falsely accepted as HELLO by train LOO threshold
    pred_before, score_before = model.predict(intruder)
    assert pred_before == "HELLO"
    assert score_before > 0.0

    # Calibrate on validation split
    validation_trials = [val_hello_1, val_hello_2, intruder]
    model.calibrate(validation_trials, min_samples=2)

    # After calibration: intruder must be rejected (silent)
    pred_after, score_after = model.predict(intruder)
    assert pred_after is None
    assert score_after == 0.0

    # Legitimate HELLO trials remain accepted
    pred_hello, score_hello = model.predict(val_hello_1)
    assert pred_hello == "HELLO"
    assert score_hello > 0.0

    # Verify calibration metadata
    report = model.calibration_report_
    assert report["calibrated"] is True
    assert "HELLO" in report["calibrated_labels"]
    hello_meta = report["per_label"]["HELLO"]
    assert hello_meta["status"] == "calibrated"
    assert hello_meta["source"] == "validation"
    assert hello_meta["wrong_accepts"] == 0
    assert hello_meta["positive_correct"] == 2
    assert hello_meta["reject_threshold"] < model.loo_report_["reject_threshold"]


def test_test_trials_not_participating_in_calibration():
    # Two identical models fitted and calibrated on identical train & val splits
    val_trials = [make_trial("HELLO", seed=s, session="sess-val") for s in (50, 51)]
    m1 = MotionModel().fit(make_train()).calibrate(val_trials, min_samples=2)
    m2 = MotionModel().fit(make_train()).calibrate(val_trials, min_samples=2)

    # Evaluating test trials must not mutate model thresholds
    test_normal = [make_trial("HELLO", seed=100, session="sess-test-normal")]
    test_noisy = [make_trial("HELLO", seed=999, session="sess-test-noisy")]
    test_noisy[0]["landmarks"] += 5.0  # extreme out-of-distribution test trial

    m1.predict(test_normal[0])
    m2.predict(test_noisy[0])

    assert m1.reject_threshold_ == m2.reject_threshold_
    assert m1.margin_threshold_ == m2.margin_threshold_
    assert m1.label_thresholds_ == m2.label_thresholds_
    assert m1.calibration_report_ == m2.calibration_report_


def test_old_model_npz_backward_compatibility(tmp_path):
    model = MotionModel().fit(make_train())
    path = tmp_path / "legacy_v1.npz"

    # Save exactly the fields present in old legacy NPZ format (no calibration keys)
    np.savez(
        path,
        templates=model.templates_,
        masks=model.masks_,
        labels=model.labels_,
        reject_threshold=np.float64(model.reject_threshold_),
        margin_threshold=np.float64(model.margin_threshold_),
        loo_report=np.array(json.dumps(model.loo_report_)),
        meta=np.array("legacy"),
    )

    loaded = MotionModel.load(path)
    assert loaded.label_thresholds_ is None
    assert loaded.calibration_report_ == {}
    assert loaded.reject_threshold_ == model.reject_threshold_
    assert loaded.margin_threshold_ == model.margin_threshold_

    for label in LABELS:
        t = make_trial(label, seed=60, session="sess-eval")
        assert loaded.predict(t) == model.predict(t)
        assert loaded.diagnose(t) == model.diagnose(t)


def test_small_sample_fallback_and_coverage_report():
    model = MotionModel().fit(make_train())
    # Provide validation samples only for HELLO (2 trials) and OPENAI (1 trial)
    val_trials = [
        make_trial("HELLO", seed=50, session="sess-val"),
        make_trial("HELLO", seed=51, session="sess-val"),
        make_trial("OPENAI", seed=52, session="sess-val"),
    ]
    model.calibrate(val_trials, min_samples=2)
    report = model.calibration_report_

    assert report["calibrated"] is True
    assert report["calibrated_labels"] == ["HELLO"]
    assert "OPENAI" in report["fallback_labels"]
    assert "WELCOME" in report["fallback_labels"]

    # Coverage: only HELLO out of 6 labels was calibrated
    assert report["coverage_rate"] == pytest.approx(1 / 6)

    # OPENAI had 1 sample (< min_samples 2), so fallback
    openai_meta = report["per_label"]["OPENAI"]
    assert openai_meta["status"] == "fallback_train_loo"
    assert openai_meta["reason"] == "insufficient_samples"
    assert openai_meta["reject_threshold"] == model.loo_report_["reject_threshold"]
    assert openai_meta["margin_threshold"] == model.loo_report_["margin_threshold"]

    # WELCOME had 0 samples, so fallback
    welcome_meta = report["per_label"]["WELCOME"]
    assert welcome_meta["status"] == "fallback_train_loo"
    assert welcome_meta["reason"] == "no_samples"
    assert welcome_meta["reject_threshold"] == model.loo_report_["reject_threshold"]


def test_calibrate_priority_zero_wrong_accepts_over_recall():
    model = MotionModel().fit(make_train())
    # 2 true HELLO trials: one close, one further away
    good_1 = make_trial("HELLO", seed=50, session="sess-val")
    good_2 = make_trial("HELLO", seed=51, session="sess-val")
    good_2["landmarks"][:, 0, :, 0] += 0.015  # distance ~0.037

    # Intruder REST trial with distance between good_1 and good_2 (~0.027)
    intruder = make_trial("HELLO", seed=52, session="sess-val")
    intruder["label"] = REST_LABEL
    intruder["trial_id"] = "intruder-rest"
    intruder["landmarks"][:, 0, :, 0] += 0.007

    val_trials = [good_1, good_2, intruder]
    model.calibrate(val_trials, min_samples=2)

    # Intruder must be rejected
    assert model.predict(intruder) == (None, 0.0)
    assert not model.diagnose(intruder)["accepted"]

    # good_1 (closer than intruder) must be accepted
    assert model.predict(good_1)[0] == "HELLO"

    # Calibration report confirms 0 wrong accepts
    hello_meta = model.label_thresholds_["HELLO"]
    assert hello_meta["wrong_accepts"] == 0


def test_save_load_calibrated_model_roundtrip(tmp_path):
    model = MotionModel().fit(make_train())
    val_trials = [make_trial("HELLO", seed=50, session="sess-val"),
                  make_trial("HELLO", seed=51, session="sess-val")]
    model.calibrate(val_trials, min_samples=2)

    path = tmp_path / "calibrated_model.npz"
    model.save(path)
    loaded = MotionModel.load(path)

    assert loaded.label_thresholds_ == model.label_thresholds_
    assert loaded.calibration_report_ == model.calibration_report_
    assert loaded.reject_threshold_ == model.reject_threshold_
    assert loaded.margin_threshold_ == model.margin_threshold_

    trial = make_trial("HELLO", seed=50, session="sess-val")
    assert loaded.predict(trial) == model.predict(trial)

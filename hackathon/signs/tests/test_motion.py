"""Tests for the personal motion-v2 dynamic gesture model (synthetic data)."""

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

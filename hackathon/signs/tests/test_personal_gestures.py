"""Tests for personal features, template model, held-out honesty and CLI.

All data here is synthetic clusters; these tests check contract behaviour
(Shapes, leakage guards, rejection, round-trip), never real recognition
accuracy.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hackathon.signs.features import FEATURE_DIM, hand_features
from hackathon.signs.personal import (
    ALL_LABELS,
    NEUTRAL_LABEL,
    POSITIVE_LABELS,
    RUNTIME_SCORE_FLOOR,
    PersonalGestureModel,
    assert_disjoint_sessions,
    evaluate_heldout,
)
from hackathon.signs import train as train_cli


# ------------------------------------------------------------------- features
def _hand_points(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(21, 3)).astype(np.float64)


def test_features_shape_dtype_and_wrist_origin():
    pts = _hand_points()
    feats = hand_features(pts, "Right")
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32
    assert np.allclose(feats[:3], 0.0)  # wrist is the origin


def test_features_translation_and_scale_invariant():
    pts = _hand_points()
    base = hand_features(pts, "Right")
    moved = hand_features(pts + np.array([3.0, -2.0, 1.0]), "Right")
    scaled = hand_features(pts * 5.0, "Right")
    assert np.allclose(base, moved, atol=1e-5)
    assert np.allclose(base, scaled, atol=1e-5)


def test_features_left_mirror_unifies_handedness():
    right = _hand_points()
    left = right.copy()
    left[:, 0] = -left[:, 0]  # same hand shape, mirrored as a left hand
    assert np.allclose(hand_features(right, "Right"),
                       hand_features(left, "Left"), atol=1e-5)


def test_features_no_rotation_up_down_stays_distinct():
    # Flipping the hand upside down (y sign) must change the features; a
    # rotation-normalising implementation would wrongly erase this.
    pts = _hand_points()
    flipped = pts.copy()
    flipped[:, 1] = -flipped[:, 1]
    assert not np.allclose(hand_features(pts, "Right"),
                           hand_features(flipped, "Right"), atol=1e-4)


def test_features_accepts_mediapipe_landmark_objects():
    pts = _hand_points()
    landmarks = [SimpleNamespace(x=float(x), y=float(y), z=float(z))
                 for x, y, z in pts]
    assert np.allclose(hand_features(landmarks, "right"),
                       hand_features(pts, "Right"))


def test_features_degenerate_inputs_raise():
    pts = _hand_points()
    with pytest.raises(ValueError):
        hand_features(pts[:20], "Right")  # wrong landmark count
    with pytest.raises(ValueError):
        hand_features(np.zeros((21, 3)), "Right")  # collapsed: zero scale
    bad = pts.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        hand_features(bad, "Right")
    with pytest.raises(ValueError):
        hand_features(pts, "lefty")


# --------------------------------------------------------------- synthetic data
def _make_trial(base: np.ndarray, label: str, session: str, trial_id: str,
                n_frames: int = 10, noise: float = 0.02,
                seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    features = base + rng.normal(scale=noise, size=(n_frames, FEATURE_DIM))
    return {
        "features": features.astype(np.float32),
        "label": label,
        "session_id": session,
        "trial_id": trial_id,
    }


def _label_bases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {label: rng.normal(size=FEATURE_DIM) for label in ALL_LABELS}


def _make_training_set(n_per_label: int = 5, session_prefix: str = "train"):
    bases = _label_bases()
    trials = []
    for label in ALL_LABELS:
        for i in range(n_per_label):
            trials.append(_make_trial(
                bases[label], label, f"{session_prefix}-{label}-{i}",
                f"{session_prefix}-{label}-{i}",
                seed=1000 + ALL_LABELS.index(label) * 100 + i))
    return trials, bases


def _fitted_model():
    trials, bases = _make_training_set()
    model = PersonalGestureModel().fit(trials)
    return model, trials, bases


# ----------------------------------------------------------------------- model
def test_fit_rejects_duplicate_trial_ids():
    trials, _ = _make_training_set()
    # Re-adding one recording under a new session must not count twice.
    repeated = dict(trials[0])
    repeated["session_id"] = "another-session"
    with pytest.raises(ValueError, match="duplicate trial_id"):
        PersonalGestureModel().fit(trials + [repeated])


def test_fit_requires_five_independent_trials_per_label():
    trials, _ = _make_training_set()
    # Drop one THUMB_UP trial: only 4 independent ones remain.
    reduced = [t for t in trials
               if not (t["label"] == "THUMB_UP" and t["trial_id"].endswith("-4"))]
    with pytest.raises(ValueError, match="not enough independent trials"):
        PersonalGestureModel().fit(reduced)


def test_fit_rejects_positive_trial_with_zero_frames():
    trials, _ = _make_training_set()
    empty = _make_trial(_label_bases()["THUMB_UP"], "THUMB_UP", "s", "empty",
                        n_frames=0)
    with pytest.raises(ValueError, match="no hand frames"):
        PersonalGestureModel().fit(trials + [empty])


def test_fit_keeps_no_hand_neutral_as_metadata_only():
    trials, _ = _make_training_set()
    no_hand = _make_trial(np.zeros(FEATURE_DIM), NEUTRAL_LABEL, "s-neutral",
                          "neutral-nohand", n_frames=0)
    model = PersonalGestureModel().fit(trials + [no_hand])
    assert model.no_hand_trial_ids_ == ["neutral-nohand"]
    assert model.templates_.shape[0] == len(trials)  # not a feature row


def test_predict_positive_neutral_and_unknown():
    model, _trials, bases = _fitted_model()

    label, score = model.predict(bases["THUMB_UP"].astype(np.float32))
    assert label == "THUMB_UP"
    assert 0.0 < score <= 1.0  # heuristic similarity, not a probability

    # NEUTRAL is a supported negative: never returned as a label.
    neutral_label, neutral_score = model.predict(bases[NEUTRAL_LABEL])
    assert neutral_label is None
    assert neutral_score == 0.0

    # Far from every template: distance rejection, same None/0.0 contract.
    rng = np.random.default_rng(123)
    unknown = bases["THUMB_UP"] + rng.normal(scale=50.0, size=FEATURE_DIM)
    rejected_label, rejected_score = model.predict(unknown.astype(np.float32))
    assert rejected_label is None
    assert rejected_score == 0.0


def test_save_load_roundtrip(tmp_path):
    model, _trials, bases = _fitted_model()
    path = tmp_path / "model.npz"
    model.save(path)

    # Loads with pickling disabled: the file must be plain numeric/strings.
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) >= {"templates", "template_labels", "threshold"}

    loaded = PersonalGestureModel.load(path)
    assert loaded.threshold_ == model.threshold_
    for label in POSITIVE_LABELS:
        assert loaded.predict(bases[label]) == model.predict(bases[label])


# ------------------------------------------------------- leakage and evaluation
def test_session_overlap_between_train_and_test_is_refused():
    train_trials, _ = _make_training_set()
    leaked = [dict(train_trials[0])]  # same session_id on the test side
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_sessions(train_trials, leaked)


def test_evaluate_heldout_applies_the_runtime_score_floor():
    train_trials, bases = _make_training_set()
    model = PersonalGestureModel().fit(train_trials)

    test_trials = []
    for label in POSITIVE_LABELS:
        for i in range(2):
            test_trials.append(_make_trial(
                bases[label], label, f"test-{label}-{i}", f"test-{label}-{i}",
                seed=2000 + ALL_LABELS.index(label) * 100 + i))
    # One no-hand NEUTRAL trial: emitting nothing is the correct behaviour.
    test_trials.append(_make_trial(np.zeros(FEATURE_DIM), NEUTRAL_LABEL,
                                   "test-neutral-nh", "test-neutral-nh",
                                   n_frames=0))

    threshold_before = model.threshold_
    result = evaluate_heldout(model, train_trials, test_trials)
    assert model.threshold_ == threshold_before  # evaluation never retunes

    trial = result["trial_level"]
    frame = result["frame_level"]
    assert result["runtime_score_floor"] == RUNTIME_SCORE_FLOOR
    assert trial["total"] == len(test_trials)
    assert trial["no_hand_trials"] == 1
    assert frame["total"] == 8 * 10

    # Clean synthetic clusters: every raw label is right before the gate.
    assert trial["label_correct"] == trial["total"]
    assert frame["label_correct"] == frame["total"]

    # The gated numbers are whatever the real score floor gives — recomputed
    # independently here, never assumed to be 100%.
    expected_gated = expected_out = 0
    for t in test_trials:
        if t["features"].shape[0] == 0:
            expected_gated += 1  # NEUTRAL no-hand: emitting nothing is right
            continue
        template = np.median(t["features"], axis=0).astype(np.float32)
        raw, score = model.predict(template)
        if raw is not None and score >= RUNTIME_SCORE_FLOOR:
            expected_gated += 1
        elif raw == t["label"]:
            expected_out += 1
    assert trial["gated_correct"] == expected_gated
    assert trial["gated_out"] == expected_out
    assert trial["gated_correct"] <= trial["label_correct"]
    assert frame["gated_correct"] <= frame["label_correct"]


class _StubModel:
    """predict() double so the gate itself is tested, not the templates."""

    def __init__(self, label, score):
        self._label = label
        self._score = score

    def predict(self, _feature):
        return self._label, self._score


def test_evaluate_gates_sub_floor_scores_like_the_runtime():
    train_trials, bases = _make_training_set()
    test_trials = [_make_trial(bases["THUMB_UP"], "THUMB_UP", "t-1", "t-1")]
    model = _StubModel("THUMB_UP", RUNTIME_SCORE_FLOOR - 0.01)

    result = evaluate_heldout(model, train_trials, test_trials)
    trial = result["trial_level"]
    assert trial["label_correct"] == 1   # the label itself was right
    assert trial["gated_correct"] == 0   # but the runtime would reject it
    assert trial["gated_out"] == 1
    assert result["frame_level"]["gated_correct"] == 0
    assert result["frame_level"]["label_correct"] == 10


def test_evaluate_accepts_score_exactly_at_the_floor():
    train_trials, bases = _make_training_set()
    test_trials = [_make_trial(bases["THUMB_UP"], "THUMB_UP", "t-1", "t-1")]
    model = _StubModel("THUMB_UP", RUNTIME_SCORE_FLOOR)

    result = evaluate_heldout(model, train_trials, test_trials)
    assert result["trial_level"]["gated_correct"] == 1
    assert result["trial_level"]["gated_out"] == 0


def test_evaluate_zero_frame_positive_is_a_miss_not_a_success():
    train_trials, _ = _make_training_set()
    model = PersonalGestureModel().fit(train_trials)
    test_trials = [
        _make_trial(np.zeros(FEATURE_DIM), "THUMB_UP", "t-empty-pos",
                    "t-empty-pos", n_frames=0),
        _make_trial(np.zeros(FEATURE_DIM), NEUTRAL_LABEL, "t-empty-neu",
                    "t-empty-neu", n_frames=0),
    ]

    result = evaluate_heldout(model, train_trials, test_trials)
    trial = result["trial_level"]
    assert trial["total"] == 2
    assert trial["no_hand_trials"] == 2
    # Only the NEUTRAL no-hand trial is correct; the zero-frame positive
    # gesture is a miss in both counts.
    assert trial["label_correct"] == 1
    assert trial["gated_correct"] == 1
    assert trial["per_label"]["THUMB_UP"]["trials"] == 1
    assert trial["per_label"]["THUMB_UP"]["label_correct"] == 0
    assert trial["per_label"][NEUTRAL_LABEL]["gated_correct"] == 1


# ------------------------------------------------------------------------ CLI
def _write_trial_npz(directory, trial: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / f"{trial['trial_id']}.npz",
        features=trial["features"],
        label=np.asarray(trial["label"]),
        session_id=np.asarray(trial["session_id"]),
        trial_id=np.asarray(trial["trial_id"]),
    )


def _populate_data_root(root):
    train_trials, bases = _make_training_set()
    for t in train_trials:
        _write_trial_npz(root / "train", t)
    for label in POSITIVE_LABELS:
        for i in range(2):
            _write_trial_npz(root / "test", _make_trial(
                bases[label], label, f"held-{label}-{i}", f"held-{label}-{i}",
                seed=3000 + ALL_LABELS.index(label) * 100 + i))
    return train_trials


def test_cli_trains_reports_and_refuses_overwrite(tmp_path, capsys):
    data_root = tmp_path / "gestures"
    output = tmp_path / "model.npz"
    _populate_data_root(data_root)

    assert train_cli.main(["--data", str(data_root),
                           "--output", str(output)]) == 0
    assert output.exists()

    report = output.with_suffix(".report.txt")
    text = report.read_text(encoding="utf-8")
    assert "Runtime score floor" in text
    assert "label only, before the runtime gate" in text
    assert "runtime-gated" in text
    # Honest framing stays attached: static gate results are not live events.
    assert "does NOT mean real-time events pass" in text
    assert "Limitations" in text  # small-sample caveats stay attached
    assert "not a probability" in text

    # Existing model is not overwritten without the explicit flag.
    assert train_cli.main(["--data", str(data_root),
                           "--output", str(output)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_refuses_session_leak_between_train_and_test(tmp_path, capsys):
    data_root = tmp_path / "gestures"
    train_trials = _populate_data_root(data_root)
    _write_trial_npz(data_root / "test", dict(train_trials[0]))  # leaked session

    assert train_cli.main(["--data", str(data_root),
                           "--output", str(tmp_path / "m.npz")]) == 2
    assert "overlap" in capsys.readouterr().err

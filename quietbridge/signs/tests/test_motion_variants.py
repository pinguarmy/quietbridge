"""Variant search for the personal gesture trainer (synthetic data).

These tests check software contracts: the final-check split is never read
before selection, selection follows its stated rule, the report is complete,
augmentation cannot leak across splits or labels and never manufactures REST,
and runs are reproducible. Synthetic separability says nothing about how
well any variant works on real hands.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from quietbridge.signs import motion_variants, train_motion
from quietbridge.signs.motion import (
    MotionModel,
    MotionModelError,
    REST_LABEL,
    augment_trial,
    estimate_landmark_jitter,
    masked_distance,
    preprocess_trial,
)
from quietbridge.signs.motion_variants import (
    BASELINE,
    VARIANTS,
    crossfit_folds,
    merge_evaluations,
    required_gain,
    select_variant,
    validation_sufficiency,
)
from quietbridge.signs.tests.test_motion import make_trial
from quietbridge.signs.tests.test_train_motion import _save_npz

LABELS = ["HELLO", "WELCOME", REST_LABEL]


def _write_dataset(root, per_val=4, per_test=2, labels=LABELS):
    for split in ("train", "validation", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
    for label in labels:
        for s in range(4):
            _save_npz(root / "train" / f"{label}-{s}.npz",
                      make_trial(label, seed=s, session="sess-train", trial_id=f"train-{label}-{s}"))
        for s in range(per_val):
            _save_npz(root / "validation" / f"{label}-{s}.npz",
                      make_trial(label, seed=100 + s, session="sess-val", trial_id=f"val-{label}-{s}"))
        for s in range(per_test):
            _save_npz(root / "test" / f"{label}-{s}.npz",
                      make_trial(label, seed=200 + s, session="sess-test", trial_id=f"test-{label}-{s}"))


def _scores(positive_correct, positive_trials=20, rest_fe=0, mislabels=0, rest_trials=5):
    return {"positive_correct": positive_correct, "positive_trials": positive_trials,
            "rest_false_emissions": rest_fe, "rest_trials": rest_trials, "mislabels": mislabels}


def _results(*scores):
    names = [v["name"] for v in VARIANTS]
    return [{"name": names[i], "crossfit_validation": s} for i, s in enumerate(scores)]


# -- (a) the final-check split is never read before selection ----------------------------------

def test_test_split_is_not_read_until_selection_and_calibration_are_frozen(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dataset(data)
    real_load, real_search = train_motion.load_trials, getattr(train_motion, "search", None)
    state = {"selected": False, "calibrated": False, "loads": []}

    def guarded_load(data_dir, split, prefix=""):
        state["loads"].append(split)
        if split == "test" and not (state["selected"] and state["calibrated"]):
            raise AssertionError("test split read before variant selection and calibration")
        return real_load(data_dir, split, prefix)

    def recording_search(*args, **kwargs):
        result = real_search(*args, **kwargs)
        state["selected"] = True
        model = result[1]
        original = model.calibrate

        def calibrate(trials, *a, **k):
            assert trials, "final calibration must use the validation session"
            assert all(t["session_id"] == "sess-val" for t in trials)
            out = original(trials, *a, **k)
            state["calibrated"] = True
            return out
        model.calibrate = calibrate
        return result

    monkeypatch.setattr(train_motion, "load_trials", guarded_load)
    monkeypatch.setattr(train_motion, "search", recording_search, raising=False)
    output = tmp_path / "m.npz"
    assert train_motion.main(["--data", str(data), "--output", str(output)]) == 0
    assert state["loads"] == ["train", "validation", "test"]
    report = json.loads((tmp_path / "m.report.json").read_text())
    assert report["variant_selection"]["test_split_read"] == "after selection and calibration were frozen"


def test_search_itself_never_touches_test_trials(tmp_path):
    train = [make_trial(l, seed=s, trial_id=f"tr-{l}-{s}") for l in LABELS for s in range(4)]
    val = [make_trial(l, seed=50 + s, session="v", trial_id=f"va-{l}-{s}") for l in LABELS for s in range(4)]
    seen = []

    def evaluate(model, trials, score_gate=None):
        seen.extend(t["trial_id"] for t in trials)
        return train_motion.evaluate(model, trials, score_gate)

    motion_variants.search(train, val, evaluate)
    assert seen and set(seen) <= {t["trial_id"] for t in val}


# -- (b) selection rule --------------------------------------------------------------------------

def test_tie_keeps_the_baseline():
    results = _results(*[_scores(15) for _ in VARIANTS])
    assert select_variant(results)[0] == BASELINE


def test_gain_below_noise_threshold_keeps_baseline_and_clear_gain_is_chosen():
    gain = required_gain(20)
    assert gain == 2
    results = _results(_scores(15), _scores(15 + gain - 1), _scores(15))
    assert select_variant(results)[0] == BASELINE
    results = _results(_scores(15), _scores(15), _scores(15 + gain))
    assert select_variant(results)[0] == VARIANTS[2]["name"]


def test_rest_silence_beats_recall():
    # A variant that finds every gesture but fires once on resting hands never wins.
    results = _results(_scores(10), _scores(20, rest_fe=1), _scores(12))
    assert select_variant(results)[0] == VARIANTS[2]["name"]
    results = _results(_scores(10), _scores(20, rest_fe=1))
    assert select_variant(results)[0] == BASELINE
    # Strictly safer at no recall cost is chosen even without a recall gain.
    results = _results(_scores(10, rest_fe=1), _scores(10, rest_fe=0))
    assert select_variant(results)[0] == VARIANTS[1]["name"]
    # Safer but losing gestures is not chosen.
    results = _results(_scores(10, rest_fe=1), _scores(9, rest_fe=0))
    assert select_variant(results)[0] == BASELINE


def test_more_wrong_gestures_disqualifies_even_with_more_recall():
    results = _results(_scores(10), _scores(20, mislabels=1))
    assert select_variant(results)[0] == BASELINE


def test_equal_best_variants_tie_break_on_list_order():
    results = _results(_scores(10), _scores(15), _scores(15), _scores(15))
    assert select_variant(results)[0] == VARIANTS[1]["name"]


def test_too_small_validation_keeps_baseline_without_comparing(tmp_path):
    data = tmp_path / "data"
    _write_dataset(data, per_val=3)
    output = tmp_path / "m.npz"
    assert train_motion.main(["--data", str(data), "--output", str(output)]) == 0
    selection = json.loads((tmp_path / "m.report.json").read_text())["variant_selection"]
    assert selection["status"] == "baseline_fallback_validation_too_small"
    assert selection["chosen"] == BASELINE
    assert [v["name"] for v in selection["variants"]] == [BASELINE]
    assert set(selection["not_tried"]) == {v["name"] for v in VARIANTS[1:]}
    assert "too small" in selection["reason"]
    assert MotionModel.load(output).augmentation_["copies_per_trial"] == 0


def test_sufficiency_needs_rest_and_every_gesture():
    train = [{"label": l} for l in ("A", "B", REST_LABEL)]
    val = [{"label": "A"}] * 4 + [{"label": "B"}] * 4
    out = validation_sufficiency(train, val)
    assert not out["sufficient"] and "REST" in out["reasons"][0]
    out = validation_sufficiency(train, val + [{"label": REST_LABEL}] * 4)
    assert out["sufficient"]
    out = validation_sufficiency(train, [{"label": "A"}] * 4 + [{"label": REST_LABEL}] * 4)
    assert not out["sufficient"] and "B (0)" in out["reasons"][0]


def test_crossfit_scores_every_validation_try_once_with_balanced_folds():
    val = [{"label": l, "trial_id": f"{l}-{i}"} for l in ("A", "B", REST_LABEL) for i in range(5)]
    folds = crossfit_folds(val)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(len(val)))
    assert abs(len(folds[0]) - len(folds[1])) <= 1
    for label in ("A", "B", REST_LABEL):
        sizes = [sum(val[i]["label"] == label for i in f) for f in folds]
        assert max(sizes) - min(sizes) <= 1


def test_merge_evaluations_keeps_denominators():
    a = {"trials": 3, "correct": 2, "false_positives": 1, "misses": 0, "positive_trials": 2, "positive_correct": 1,
         "mislabels": 1, "rest_trials": 1, "rest_correctly_silent": 1,
         "per_label": {"A": {"trials": 2, "correct": 1, "miss": 0, "wrong_label": 1}}, "confusion": {"A -> B": 1}}
    b = {**a, "rest_trials": 0, "rest_correctly_silent": 0, "trials": 2, "correct": 1}
    merged = merge_evaluations([a, b])
    assert merged["positive_trials"] == 4 and merged["positive_recall"] == pytest.approx(0.5)
    assert merged["rest_trials"] == 1 and merged["rest_false_emissions"] == 0
    assert merged["confusion"]["A -> B"] == 2
    assert merge_evaluations([b])["rest_silent_rate"] is None  # not measured, not 0


# -- (c) report completeness and determinism ----------------------------------------------------

def test_report_lists_every_variant_with_validation_numbers_and_the_choice(tmp_path):
    data = tmp_path / "data"
    _write_dataset(data)
    output = tmp_path / "m.npz"
    assert train_motion.main(["--data", str(data), "--output", str(output)]) == 0
    report = json.loads((tmp_path / "m.report.json").read_text())
    selection = report["variant_selection"]
    assert [v["name"] for v in selection["search_space"]] == [v["name"] for v in VARIANTS]
    assert [v["name"] for v in selection["variants"]] == [v["name"] for v in VARIANTS]
    for v in selection["variants"]:
        cf = v["crossfit_validation"]
        for key in ("positive_correct", "positive_trials", "rest_false_emissions", "rest_trials", "mislabels"):
            assert isinstance(cf[key], int)
        assert cf["trials"] == report["validation_trials"]
        assert v["seconds"]["fit"] >= 0 and "qualifies" in v and v["description"]
    assert selection["chosen"] in {v["name"] for v in VARIANTS}
    assert selection["status"] in ("selected", "baseline_kept")
    assert selection["reason"] and selection["selection_metric"] and selection["seed"] == 0
    assert "mirroring" in selection["excluded_on_purpose"]
    assert report["model_config"]["variant"] == selection["chosen"]
    assert report["validation_crossfit"]["per_label"] and "confusion" in report["validation_crossfit"]
    assert len(report["model_sha256"]) == 64
    # The recorded choice is exactly what the rule gives for the listed numbers.
    rescored = [{"name": v["name"], "crossfit_validation": v["crossfit_validation"]} for v in selection["variants"]]
    assert select_variant(rescored)[0] == selection["chosen"]
    assert report["test_ungated"]["trials"] == 6


def test_same_seed_reproduces_the_candidate(tmp_path):
    data = tmp_path / "data"
    _write_dataset(data)
    reports, models = [], []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.npz"
        assert train_motion.main(["--data", str(data), "--output", str(out), "--seed", "7"]) == 0
        reports.append(json.loads((tmp_path / f"{name}.report.json").read_text()))
        models.append(MotionModel.load(out))
    strip = lambda r: [{k: v for k, v in x.items() if k != "seconds"} for x in r["variant_selection"]["variants"]]
    assert strip(reports[0]) == strip(reports[1])
    assert reports[0]["variant_selection"]["chosen"] == reports[1]["variant_selection"]["chosen"]
    np.testing.assert_array_equal(models[0].templates_, models[1].templates_)
    assert models[0].reject_threshold_ == models[1].reject_threshold_


def test_augmented_fit_is_deterministic_and_seed_dependent():
    train = [make_trial(l, seed=s, trial_id=f"t-{l}-{s}") for l in LABELS for s in range(3)]
    a = MotionModel().fit(train, augment=3, seed=1)
    b = MotionModel().fit(list(reversed(train)), augment=3, seed=1)
    c = MotionModel().fit(train, augment=3, seed=2)
    # Copies are seeded per trial id, so input order does not change them.
    rows = lambda m: sorted(r.tobytes() for r in m.templates_.reshape(len(m.labels_), -1))
    assert rows(a) == rows(b)
    assert not np.allclose(a.templates_[a.augmented_], c.templates_[c.augmented_])


# -- (d) augmentation leakage guards --------------------------------------------------------------

def test_augmentation_keeps_label_session_and_never_manufactures_rest():
    train = [make_trial(l, seed=s, trial_id=f"t-{l}-{s}") for l in LABELS for s in range(3)]
    model = MotionModel().fit(train, augment=4, seed=0)
    n_rest = sum(t["label"] == REST_LABEL for t in train)
    assert int(np.sum(model.labels_ == REST_LABEL)) == n_rest  # REST templates are only real tries
    assert not model.augmented_[model.labels_ == REST_LABEL].any()
    for i in np.flatnonzero(model.augmented_):
        source = train[model.groups_[i]]
        assert model.labels_[i] == source["label"] != REST_LABEL
    # Every template comes from a training trial: nothing from other splits can enter.
    assert set(model.groups_.tolist()) == set(range(len(train)))
    assert len(model.labels_) == len(train) + 4 * (len(train) - n_rest)
    with pytest.raises(MotionModelError):
        augment_trial(make_trial(REST_LABEL), 0, 0.01)


def test_augmented_copy_is_a_plausible_variation_of_its_source():
    t = make_trial("HELLO", seed=3, trial_id="src")
    jitter = estimate_landmark_jitter([t, make_trial(REST_LABEL, seed=4)])
    copy = augment_trial(t, 0, jitter, seed=0)
    assert copy["label"] == "HELLO" and copy["session_id"] == t["session_id"]
    assert copy["augment_source"] == "src" and copy["trial_id"] != "src"
    assert np.all(np.diff(copy["timestamps_ms"]) > 0)
    assert copy["hand_present"].sum() >= 0.8 * t["hand_present"].sum()
    # Missing hands stay missing.
    assert np.isnan(copy["landmarks"][~copy["hand_present"]]).all()
    # Same seed and id: identical; different copy index: different.
    np.testing.assert_array_equal(copy["landmarks"], augment_trial(t, 0, jitter, seed=0)["landmarks"])
    other = augment_trial(t, 1, jitter, seed=0)
    assert other["landmarks"].shape != copy["landmarks"].shape or \
        not np.allclose(other["landmarks"][:, 0, 0, 0], copy["landmarks"][:, 0, 0, 0])


def test_jitter_is_measured_from_resting_tries():
    rest = [make_trial(REST_LABEL, seed=s) for s in range(3)]  # per-coordinate noise sd 0.005
    jitter = estimate_landmark_jitter(rest + [make_trial("HELLO")])
    # median |N(0, 0.005) - N(0, 0.005)| = 0.6745 * 0.005 * sqrt(2) ~= 0.0048
    assert 0.0035 < jitter < 0.006
    assert estimate_landmark_jitter([make_trial("HELLO")]) > 0


def test_leave_one_out_never_matches_a_trial_against_its_own_copies():
    # ONLY has one recording: without group exclusion its own copies would make LOO "correct".
    train = [make_trial("HELLO", seed=s, trial_id=f"h{s}") for s in range(3)]
    train += [make_trial("WELCOME", seed=9, trial_id="only")]
    model = MotionModel().fit(train, augment=4, seed=0)
    assert model.loo_report_["n_trials"] == len(train)  # copies are never LOO queries
    stats = model._leave_one_out_stats()
    only = [s for s in stats if s["true"] == "WELCOME"]
    assert len(only) == 1 and only[0]["pred"] != "WELCOME"


# -- shared mechanics ------------------------------------------------------------------------------

def test_vectorised_distances_match_reference_masked_distance():
    train = [make_trial(l, seed=s) for l in LABELS for s in range(2)]
    train[0]["hand_present"][5:20, 0] = False
    model = MotionModel().fit(train)
    query = make_trial("HELLO", seed=77)
    query["hand_present"][:12, 0] = False
    feat, mask = preprocess_trial(query)
    fast = model._distances_to_templates(feat, mask)
    slow = [masked_distance(feat, mask, model.templates_[i], model.masks_[i]) for i in range(len(model.labels_))]
    for a, b in zip(fast, slow):
        assert (math.isinf(a) and math.isinf(b)) or a == pytest.approx(b, rel=1e-9)


def test_knn_aggregation_round_trips_and_differs_from_nearest(tmp_path):
    train = [make_trial(l, seed=s) for l in LABELS for s in range(3)]
    nearest = MotionModel().fit(train)
    knn = MotionModel(aggregation="knn", k=2).fit(train)
    query = make_trial("HELLO", seed=40)
    assert knn.diagnose(query)["d1"] >= nearest.diagnose(query)["d1"]
    path = tmp_path / "knn.npz"
    knn.save(path)
    loaded = MotionModel.load(path)
    assert (loaded.aggregation, loaded.k) == ("knn", 2)
    assert loaded.diagnose(query) == knn.diagnose(query)
    with pytest.raises(MotionModelError):
        MotionModel(aggregation="nearest", k=2)

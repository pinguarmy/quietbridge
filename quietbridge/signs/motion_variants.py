"""Choose a personal-gesture model variant on the independent validation session.

Few-shot template models are sensitive to preprocessing choices, so the
trainer tries a small, fixed list of principled variants (``VARIANTS``) and
keeps one only when the "Check on another occasion" session shows it is
safer, or clearly better at no cost in safety. Nothing here reads the
final-check (test) split; ``train_motion`` loads that split only after the
choice is frozen.

How a variant is scored (``crossfit_validation``): thresholds are themselves
calibrated on validation, so scoring a variant on the same tries it was
calibrated on would flatter flexible variants. Each variant is therefore
scored by 2-fold cross-fitting inside validation: calibrate on one half,
score the other half, swap, add the counts. Every validation try is scored
exactly once, by thresholds that never saw it.

Selection rule (``select_variant``), REST silence first:
1. Fewer REST false emissions (resting hands that produced a gesture) beats
   everything; then fewer wrong-gesture emissions; then more gestures found.
2. A variant may replace the baseline only if it is no worse on either
   safety count AND either strictly safer (at no loss of gestures found) or
   finds at least ``required_gain`` more gestures. ``required_gain`` is
   max(2 tries, 10 % of validation gesture tries): a one-try difference on a
   few dozen tries is within noise.
3. Ties keep the earlier variant in ``VARIANTS``; the baseline is first.

If validation is too small to tell variants apart (fewer than
``MIN_VALIDATION_TRIES_PER_GESTURE`` tries of any trained gesture, or fewer
than ``MIN_VALIDATION_REST_TRIES`` REST tries), no variant is compared and the
plain baseline is kept; the report says so.

All of this is a software contract. Synthetic tests show the selection and
leakage guards behave as specified; they say nothing about real hands.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

from .motion import REST_LABEL, MotionModel

#: Every variant tried, in tie-break order. Keep this list short and explicit.
VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline", "augment": 0, "aggregation": "nearest", "k": 1, "per_label": True,
     "description": "Recorded tries only; nearest recording decides; thresholds per gesture."},
    {"name": "global_thresholds", "augment": 0, "aggregation": "nearest", "k": 1, "per_label": False,
     "description": "As baseline, but one acceptance threshold shared by all gestures."},
    {"name": "knn2", "augment": 0, "aggregation": "knn", "k": 2, "per_label": True,
     "description": "A gesture's distance is the mean of its 2 closest recordings, so one odd try counts less."},
    {"name": "augment4", "augment": 4, "aggregation": "nearest", "k": 1, "per_label": True,
     "description": "Adds 4 slightly varied copies of each practice gesture try (timing, position, small "
                    "rotation, tracking noise). Resting tries are never copied."},
    {"name": "augment4_knn2", "augment": 4, "aggregation": "knn", "k": 2, "per_label": True,
     "description": "Varied copies plus the mean of the 2 closest distinct recordings."},
    {"name": "augment4_global", "augment": 4, "aggregation": "nearest", "k": 1, "per_label": False,
     "description": "Varied copies with one shared acceptance threshold."},
]
BASELINE = "baseline"

#: Recommendations, not guarantees: below these, differences between variants
#: are mostly noise, so the baseline is kept without comparing.
MIN_VALIDATION_TRIES_PER_GESTURE = 4
MIN_VALIDATION_REST_TRIES = 4
MIN_GAIN_TRIES = 2
MIN_GAIN_FRACTION = 0.10
CROSSFIT_FOLDS = 2

SELECTION_METRIC = (
    "Cross-fitted on the validation session (2 folds; thresholds calibrated on one half, scored on "
    "the other). Ordered by: fewest REST false emissions, then fewest wrong-gesture emissions, then "
    "most gestures found. A variant replaces the baseline only if it is no worse on both safety "
    "counts and is strictly safer at no recall cost, or finds at least max(2, 10% of validation "
    "gesture tries) more gestures. Ties keep the earlier variant (baseline first). The final-check "
    "session is never read."
)


def validation_sufficiency(train_trials, validation_trials) -> dict[str, Any]:
    """Whether validation can distinguish variants; the reasons when it cannot."""
    gestures = sorted({str(t["label"]) for t in train_trials} - {REST_LABEL})
    counts = {label: sum(1 for t in validation_trials if t["label"] == label) for label in gestures}
    rest = sum(1 for t in validation_trials if t["label"] == REST_LABEL)
    reasons = []
    short = [label for label, n in counts.items() if n < MIN_VALIDATION_TRIES_PER_GESTURE]
    if short:
        reasons.append(f"fewer than {MIN_VALIDATION_TRIES_PER_GESTURE} validation tries for: "
                       + ", ".join(f"{label} ({counts[label]})" for label in short))
    if rest < MIN_VALIDATION_REST_TRIES:
        reasons.append(f"{rest} validation REST tries (need {MIN_VALIDATION_REST_TRIES})")
    return {"sufficient": not reasons, "reasons": reasons, "gesture_tries": counts, "rest_tries": rest,
            "minimums": {"tries_per_gesture": MIN_VALIDATION_TRIES_PER_GESTURE,
                         "rest_tries": MIN_VALIDATION_REST_TRIES}}


def fit_variant(variant: dict[str, Any], train_trials, seed: int = 0) -> MotionModel:
    """Fit templates (and train-only LOO thresholds) for one variant."""
    model = MotionModel(aggregation=variant["aggregation"], k=variant["k"])
    return model.fit(train_trials, augment=variant["augment"], seed=seed)


def crossfit_folds(validation_trials, folds: int = CROSSFIT_FOLDS) -> list[list[int]]:
    """Deterministic label-stratified folds: per label, by trial_id, round robin."""
    by_label: dict[str, list[int]] = {}
    for i, t in enumerate(validation_trials):
        by_label.setdefault(str(t["label"]), []).append(i)
    out: list[list[int]] = [[] for _ in range(folds)]
    offset = 0
    for label in sorted(by_label):
        members = sorted(by_label[label], key=lambda i: str(validation_trials[i].get("trial_id", i)))
        for j, i in enumerate(members):
            out[(j + offset) % folds].append(i)
        offset += len(members)  # keep fold sizes balanced across labels
    return [sorted(f) for f in out]


def merge_evaluations(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Add the counts of several ``train_motion.evaluate`` results; recompute rates."""
    keys = ("trials", "correct", "false_positives", "misses", "positive_trials", "positive_correct",
            "mislabels", "rest_trials", "rest_correctly_silent")
    total = {k: sum(p[k] for p in parts) for k in keys}
    per_label: dict[str, dict[str, int]] = {}
    confusion: dict[str, int] = {}
    for p in parts:
        for label, stats in p["per_label"].items():
            row = per_label.setdefault(label, {"trials": 0, "correct": 0, "miss": 0, "wrong_label": 0})
            for k in row:
                row[k] += stats[k]
        for key, n in p["confusion"].items():
            confusion[key] = confusion.get(key, 0) + n
    n, n_rest, n_pos = total["trials"], total["rest_trials"], total["positive_trials"]
    return {
        **total,
        "accuracy": total["correct"] / n if n else None,
        "positive_recall": total["positive_correct"] / n_pos if n_pos else None,
        "rest_silent_rate": total["rest_correctly_silent"] / n_rest if n_rest else None,
        "rest_false_emissions": n_rest - total["rest_correctly_silent"],
        "per_label": per_label,
        "confusion": dict(sorted(confusion.items())),
    }


def crossfit_validation(model: MotionModel, validation_trials, evaluate_fn: Callable,
                        per_label: bool = True, folds: int = CROSSFIT_FOLDS) -> dict[str, Any]:
    """Score every validation try once, with thresholds calibrated on the other fold(s)."""
    parts = []
    for held in crossfit_folds(validation_trials, folds):
        if not held:
            continue
        held_set = set(held)
        calibration = [t for i, t in enumerate(validation_trials) if i not in held_set]
        scored = [validation_trials[i] for i in held]
        fold_model = copy.copy(model)  # calibrate() reassigns thresholds; templates are shared read-only
        fold_model.calibrate(calibration, per_label=per_label)
        parts.append(evaluate_fn(fold_model, scored))
    merged = merge_evaluations(parts)
    merged["folds"] = len(parts)
    return merged


def _safety(m: dict[str, Any]) -> tuple[int, int]:
    return (m["rest_false_emissions"], m["mislabels"])


def selection_key(m: dict[str, Any]) -> tuple[int, int, int]:
    """Smaller is better: REST false emissions, wrong gestures, then gestures missed."""
    return (m["rest_false_emissions"], m["mislabels"], -m["positive_correct"])


def required_gain(positive_trials: int) -> int:
    return max(MIN_GAIN_TRIES, math.ceil(MIN_GAIN_FRACTION * positive_trials))


def select_variant(results: list[dict[str, Any]]) -> tuple[str, str]:
    """Pick one variant from scored results (list order = tie-break order).

    Each result has ``name`` and ``crossfit_validation`` (merged counts). The
    first result must be the baseline.
    """
    if not results or results[0]["name"] != BASELINE:
        raise ValueError("the baseline must be the first scored variant")
    base = results[0]["crossfit_validation"]
    gain = required_gain(base["positive_trials"])
    qualifying = [(0, results[0])]
    for order, result in enumerate(results[1:], start=1):
        m = result["crossfit_validation"]
        no_worse = all(a <= b for a, b in zip(_safety(m), _safety(base)))
        safer = _safety(m) < _safety(base) and m["positive_correct"] >= base["positive_correct"]
        clearly_more = m["positive_correct"] >= base["positive_correct"] + gain
        result["qualifies"] = bool(no_worse and (safer or clearly_more))
        if result["qualifies"]:
            qualifying.append((order, result))
    results[0]["qualifies"] = True
    order, best = min(qualifying, key=lambda pair: (selection_key(pair[1]["crossfit_validation"]), pair[0]))
    if best["name"] == BASELINE:
        return BASELINE, ("kept the baseline: no variant was safer on validation, or found at least "
                          f"{gain} more gestures without more REST or wrong-gesture emissions")
    m = best["crossfit_validation"]
    return best["name"], (
        f"chose {best['name']}: on validation (cross-fitted) it had {m['rest_false_emissions']} REST false "
        f"emissions (baseline {base['rest_false_emissions']}), {m['mislabels']} wrong gestures (baseline "
        f"{base['mislabels']}) and found {m['positive_correct']} of {m['positive_trials']} gestures "
        f"(baseline {base['positive_correct']}); required gain was {gain}")


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("positive_correct", "positive_trials", "positive_recall", "rest_false_emissions",
            "rest_trials", "rest_silent_rate", "mislabels", "misses", "trials", "folds")
    return {k: m.get(k) for k in keys}


def search(train_trials, validation_trials, evaluate_fn: Callable, seed: int = 0,
           variants: list[dict[str, Any]] | None = None,
           clock: Callable[[], float] = time.perf_counter) -> tuple[dict[str, Any], MotionModel, dict[str, Any]]:
    """Fit, cross-fit and select. Returns (report, fitted chosen model, chosen variant).

    The chosen model is fitted on train only and not yet calibrated; the
    caller calibrates it on the whole validation session afterwards.
    """
    variants = list(variants or VARIANTS)
    if variants[0]["name"] != BASELINE:
        raise ValueError("the baseline must be the first variant")
    started = clock()
    sufficiency = validation_sufficiency(train_trials, validation_trials) if validation_trials else {
        "sufficient": False, "reasons": ["no validation session"], "gesture_tries": {}, "rest_tries": 0,
        "minimums": {"tries_per_gesture": MIN_VALIDATION_TRIES_PER_GESTURE,
                     "rest_tries": MIN_VALIDATION_REST_TRIES}}
    to_try = variants if sufficiency["sufficient"] else variants[:1]
    results, models = [], {}
    for variant in to_try:
        t0 = clock()
        model = fit_variant(variant, train_trials, seed=seed)
        t1 = clock()
        scored = (crossfit_validation(model, validation_trials, evaluate_fn, per_label=variant["per_label"])
                  if validation_trials else None)
        t2 = clock()
        models[variant["name"]] = model
        results.append({"name": variant["name"], "description": variant["description"],
                        "config": {k: variant[k] for k in ("augment", "aggregation", "k", "per_label")},
                        "templates": int(len(model.labels_)),
                        "crossfit_validation": scored,
                        "seconds": {"fit": round(t1 - t0, 3), "crossfit": round(t2 - t1, 3)}})
    if sufficiency["sufficient"]:
        chosen, reason = select_variant(results)
        status = "selected" if chosen != BASELINE else "baseline_kept"
    else:
        chosen = BASELINE
        status = "baseline_fallback_validation_too_small"
        reason = ("kept the baseline without comparing variants: validation is too small to tell them "
                  "apart (" + "; ".join(sufficiency["reasons"]) + ")")
    full = {r["name"]: r["crossfit_validation"] for r in results}
    for r in results:
        if r["crossfit_validation"] is not None:
            r["crossfit_validation"] = _summary(r["crossfit_validation"])
    report = {
        "status": status,
        "chosen": chosen,
        "reason": reason,
        "selection_metric": SELECTION_METRIC,
        "seed": int(seed),
        "validation_sufficiency": sufficiency,
        "search_space": [{"name": v["name"], "description": v["description"],
                          **{k: v[k] for k in ("augment", "aggregation", "k", "per_label")}} for v in variants],
        "not_tried": [v["name"] for v in variants if v not in to_try],
        "required_gain_tries": (required_gain(results[0]["crossfit_validation"]["positive_trials"])
                                if results and results[0]["crossfit_validation"] else None),
        "variants": results,
        "test_split_read": False,
        "excluded_on_purpose": {
            "mirroring": "Left/right mirroring is never used: hand identity is part of a personal gesture, "
                         "and a mirrored movement can look like a different gesture.",
            "uniform_scale_and_speed": "Uniform scaling and uniform speed changes are exact no-ops after "
                                       "hand-size normalisation and per-try resampling.",
            "rest_augmentation": "REST tries are never copied or synthesised; rest evidence stays real.",
        },
        "seconds_total": round(clock() - started, 3),
        "caveat": "Synthetic tests check this selection logic only; real benefit needs real validation sessions.",
        # Full cross-fitted counts (per gesture, confusions) of the chosen variant.
        "chosen_crossfit_full": full[chosen],
    }
    chosen_variant = next(v for v in variants if v["name"] == chosen)
    return report, models[chosen], chosen_variant

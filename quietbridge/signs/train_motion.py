#!/usr/bin/env python
"""Train and evaluate the personal motion-v2 dynamic gesture model.

Fits MotionModel on the train split of a sequences directory (by default every
recorded session; --session-prefix narrows it). A small, fixed list of
variants (training-only augmentation, aggregation over recordings, per-gesture
vs shared thresholds; see motion_variants.VARIANTS) is compared on the
independent validation session only, with REST silence first; the baseline is
kept when validation is too small to tell them apart. Thresholds are then
calibrated on validation (train leave-one-out fallback). Only after that is
the held-out test split read, and evaluated exactly once.

Outputs (default under ~/.cache/quietbridge/, outside the repo):
  motion_v2.npz          model file
  motion_v2.report.json  metrics report (paths and counts, no raw data)

Usage:
  .venv/bin/python quietbridge/signs/train_motion.py [--data DIR]
      [--output PATH] [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quietbridge.signs.capture_quality import gesture_quality  # noqa: E402
from quietbridge.signs.motion import (  # noqa: E402
    REST_LABEL,
    check_session_leakage,
    load_trials,
)
from quietbridge.signs.motion_variants import VARIANTS, search  # noqa: E402

DEFAULT_DATA = "~/.local/share/quietbridge/sequences"
DEFAULT_OUTPUT = "~/.cache/quietbridge/motion_v2.npz"
SESSION_PREFIX = ""
SCORE_GATE = 0.75


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sessions_per_label(trials):
    sessions = {}
    for t in trials:
        sessions.setdefault(t["label"], set()).add(t["session_id"])
    return {label: len(s) for label, s in sorted(sessions.items())}


def evaluate(model, trials, score_gate=None):
    """Evaluate once; no threshold changes here. If score_gate is set, only
    predictions with score >= gate are counted as emitted (diagnostic only)."""
    per_label = {}
    confusion = {}
    n_correct = n_false_positive = n_miss = n_rest_ok = n_mislabels = 0
    t0 = time.perf_counter()
    for t in trials:
        true = t["label"]
        label, score = model.predict(t)
        if score_gate is not None and label is not None and score < score_gate:
            label, score = None, 0.0
        stats = per_label.setdefault(
            true, {"trials": 0, "correct": 0, "miss": 0, "wrong_label": 0})
        stats["trials"] += 1
        key = (true, label if label is not None else "<none>")
        confusion[key] = confusion.get(key, 0) + 1
        if true == REST_LABEL:
            if label is None:
                n_rest_ok += 1
                stats["correct"] += 1
            else:
                n_false_positive += 1  # REST produced output: false alarm
                stats["wrong_label"] += 1
        else:
            if label is None:
                n_miss += 1
                stats["miss"] += 1
            elif label == true:
                n_correct += 1
                stats["correct"] += 1
            else:
                n_false_positive += 1
                n_mislabels += 1
                stats["wrong_label"] += 1
    elapsed = time.perf_counter() - t0
    n = len(trials)
    n_rest = per_label.get(REST_LABEL, {}).get("trials", 0)
    return {
        "trials": n,
        "correct": n_correct + n_rest_ok,
        # No trials is "not measured", not 0 % (or 100 %).
        "accuracy": (n_correct + n_rest_ok) / n if n else None,
        "false_positives": n_false_positive,
        "misses": n_miss,
        "positive_trials": n - n_rest,
        "positive_correct": n_correct,
        "positive_recall": n_correct / (n - n_rest) if n > n_rest else None,
        "mislabels": n_mislabels,
        "rest_trials": n_rest,
        "rest_correctly_silent": n_rest_ok,
        "rest_silent_rate": n_rest_ok / n_rest if n_rest else None,
        "per_label": per_label,
        "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(confusion.items())},
        "mean_predict_ms": round(1000.0 * elapsed / n, 2) if n else None,
    }


def capture_quality_summary(trials):
    """Report (never exclude) tries whose capture looks poor.

    Uses the review stored at save time when present; older recordings are
    measured now from their landmarks with the same thresholds.
    """
    flagged, stored = [], 0
    for t in trials:
        review = t.get("quality")
        source = "stored"
        if review is None:
            source = "measured_now"
            try:
                review = gesture_quality(t["landmarks"], t["hand_present"], t["timestamps_ms"], t["label"])
            except ValueError:
                review = None
        else:
            stored += 1
        codes = [w.get("code") for w in (review or {}).get("warnings", [])
                 if isinstance(w, dict) and w.get("severity", "warn") == "warn"]
        if codes:
            flagged.append({"trial_id": t["trial_id"], "label": t["label"], "warnings": codes,
                            "source": source, "decision": (review or {}).get("decision")})
    return {"checked": len(trials), "with_stored_review": stored, "flagged": flagged,
            "used_in_training_or_scoring": True,
            "note": "Flagged tries are reported, not excluded. Re-record them if the warning is right."}


def _label_counts(trials):
    return {l: sum(1 for t in trials if t["label"] == l) for l in sorted({t["label"] for t in trials})}


def _diagnostics(model, trials):
    return [{"trial_id": trial["trial_id"], "true": trial["label"], **model.diagnose(trial)} for trial in trials]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="sequences root containing train/ and test/")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="candidate NPZ path (report goes to <name>.report.json)")
    parser.add_argument("--session-prefix", default=SESSION_PREFIX,
                        help="only sessions whose id starts with this prefix; default uses every session")
    parser.add_argument("--allow-legacy-no-validation", action="store_true",
                        help="reproduce old data without an independent validation session")
    parser.add_argument("--no-variant-search", action="store_true",
                        help="train only the plain baseline (no augmentation/aggregation variants)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for training-only augmentation; the same seed reproduces the candidate")
    parser.add_argument("--overwrite", action="store_true",
                        help="required to replace an existing model file")
    args = parser.parse_args(argv)

    data_dir = Path(args.data).expanduser()
    output = Path(args.output).expanduser()
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")
    report_path = output.with_suffix("").with_name(output.stem + ".report.json")
    if report_path.exists() and not args.overwrite:
        parser.error(f"{report_path} exists; pass --overwrite to replace it")

    load = load_trials  # resolved at call time (tests replace it to guard the test split)
    started = time.perf_counter()
    # The final-check (test) split is NOT loaded here: it is read only after
    # the variant and thresholds are frozen, so it cannot influence them.
    train_trials = load(data_dir, "train", args.session_prefix)
    validation_trials = load(data_dir, "validation", args.session_prefix)
    if not train_trials:
        parser.error(f"no train trials with session prefix {args.session_prefix!r} in {data_dir}")
    if not validation_trials and not args.allow_legacy_no_validation:
        parser.error("record an independent validation session, or explicitly use --allow-legacy-no-validation")
    if validation_trials:
        check_session_leakage(train_trials, validation_trials)

    # 1. Variant search on validation only (baseline alone when it is too small).
    variants = VARIANTS[:1] if args.no_variant_search else VARIANTS
    selection, model, chosen = search(train_trials, validation_trials, evaluate, seed=args.seed,
                                      variants=variants)
    if args.no_variant_search:
        selection["status"] = "baseline_only_requested"
        selection["reason"] = "variant search disabled with --no-variant-search"

    # 2. Validation before calibration (train leave-one-out thresholds).
    val_pre = evaluate(model, validation_trials) if validation_trials else None
    val_pre_diag = _diagnostics(model, validation_trials)

    # 3. Calibrate thresholds on the whole validation session (never sees test).
    model.calibrate(validation_trials, per_label=chosen["per_label"])

    # 4. Validation after calibration: in-sample for thresholds, so optimistic.
    val_post = evaluate(model, validation_trials) if validation_trials else None
    val_post_diag = _diagnostics(model, validation_trials)
    chosen_crossfit = selection.pop("chosen_crossfit_full")
    selection_seconds = time.perf_counter() - started

    # 5. Selection is frozen. Only now is the final-check split read, once.
    test_trials = load(data_dir, "test", args.session_prefix)
    if validation_trials:
        check_session_leakage(validation_trials, test_trials)
    check_session_leakage(train_trials, test_trials)
    selection["test_split_read"] = "after selection and calibration were frozen"
    test_ungated = evaluate(model, test_trials)
    test_score_gated = evaluate(model, test_trials, score_gate=SCORE_GATE)
    test_diag = _diagnostics(model, test_trials)

    model.save(output)

    report = {
        "model_path": str(output),
        "model_sha256": _sha256(output),
        "train_dir": str(data_dir / "train"),
        "validation_dir": str(data_dir / "validation") if validation_trials else None,
        "test_dir": str(data_dir / "test"),
        "session_prefix": args.session_prefix,
        "train_trials": len(train_trials),
        "validation_trials": len(validation_trials),
        "test_trials": len(test_trials),
        "train_labels": _label_counts(train_trials),
        "validation_labels": _label_counts(validation_trials) if validation_trials else {},
        "test_labels": _label_counts(test_trials),
        "train_sessions": _sessions_per_label(train_trials),
        "validation_sessions": _sessions_per_label(validation_trials),
        "thresholds_calibrated_on": "validation split (with train LOO fallback)" if validation_trials else "train leave-one-trial-out only",
        "variant_selection": selection,
        "model_config": {"variant": chosen["name"], "aggregation": model.aggregation, "k": model.k,
                         "per_label_thresholds": chosen["per_label"], "augmentation": model.augmentation_},
        "loo": model.loo_report_,
        "calibration": model.calibration_report_,
        # Honest validation numbers for the chosen variant (per gesture and confusions):
        # every try scored by thresholds calibrated without it. Guidance reads this.
        "validation_crossfit": chosen_crossfit,
        "validation_pre_calibration": val_pre,
        "validation_pre_calibration_diagnostics": val_pre_diag,
        "validation_post_calibration": val_post,
        "validation_post_calibration_diagnostics": val_post_diag,
        "validation_ungated": val_post,
        "validation_diagnostics": val_post_diag,
        "test_ungated": test_ungated,
        "test_score_gated_0.75": test_score_gated,
        "test_diagnostics": test_diag,
        "capture_quality": {"train": capture_quality_summary(train_trials),
                            "validation": capture_quality_summary(validation_trials),
                            "test": capture_quality_summary(test_trials)},
        "timing_seconds": {"selection_and_calibration": round(selection_seconds, 3),
                           "total": round(time.perf_counter() - started, 3)},
        "notes": [
            "Test split was read only after the variant and thresholds were frozen, evaluated once, "
            "and never used for selection, tuning or calibration.",
            "validation_crossfit scores each validation try with thresholds calibrated on the other half; "
            "validation_post_calibration reuses the same tries for calibration and scoring, so it is optimistic.",
            "score is a heuristic matching score, not a probability.",
            "The 0.75 score gate is diagnostic only; runtime uses model acceptance.",
            "Model classifies pre-segmented trials only; no continuous recognition.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def print_metrics(name, metrics):
        if not metrics:
            return
        recall = metrics["positive_recall"]
        silent = metrics["rest_silent_rate"]
        recall_text = f"{recall:.2f}" if recall is not None else "n/a"
        silent_text = f"{silent:.2f}" if silent is not None else "n/a"
        accuracy = metrics["accuracy"]
        accuracy_text = f"{accuracy:.2f}" if accuracy is not None else "n/a (no trials)"
        print(f"{name}: acc={accuracy_text} fp={metrics['false_positives']} "
              f"misses={metrics['misses']} mislabels={metrics['mislabels']} "
              f"positive_recall={metrics['positive_correct']}/{metrics['positive_trials']} "
              f"({recall_text}) rest_silent={metrics['rest_correctly_silent']}/"
              f"{metrics['rest_trials']} ({silent_text})")

    print(f"train: {report['train_trials']} trials {report['train_labels']}")
    if validation_trials:
        print(f"val:   {report['validation_trials']} trials {report['validation_labels']}")
    print(f"test:  {report['test_trials']} trials {report['test_labels']}")
    print(f"LOO:   acc={model.loo_report_['raw_accuracy']:.2f} "
          f"reject={model.loo_report_['reject_threshold']:.4f} margin={model.loo_report_['margin_threshold']:.4f}")
    if validation_trials:
        print_metrics("validation (pre-cal)", val_pre)
        cal = model.calibration_report_ or {}
        cov = cal.get("coverage_rate", 0.0)
        cal_lbls = cal.get("calibrated_labels", [])
        fb_lbls = cal.get("fallback_labels", [])
        print(f"calibration: coverage={cov:.1%} calibrated={cal_lbls} fallback={fb_lbls}")
        print_metrics("validation (post-cal)", val_post)
    print(f"variant: {selection['chosen']} ({selection['status']}); {selection['reason']}")
    print_metrics("test", test_ungated)
    print_metrics("gated@0.75", test_score_gated)
    print(f"wrote {output} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

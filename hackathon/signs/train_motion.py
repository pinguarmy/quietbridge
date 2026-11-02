#!/usr/bin/env python
"""Train and evaluate the personal motion-v2 dynamic gesture model.

Fits MotionModel on the train split of ~/.local/share/quietbridge/sequences
(session prefix peter-motion-v2-), calibrates thresholds with
leave-one-trial-out on train only, then evaluates once on the held-out
test split. The test split is never used for tuning.

Outputs (default under ~/.cache/quietbridge/, outside the repo):
  motion_v2.npz          model file
  motion_v2.report.json  metrics report (paths and counts, no raw data)

Usage:
  .venv/bin/python hackathon/signs/train_motion.py [--data DIR]
      [--output PATH] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackathon.signs.motion import (  # noqa: E402
    MotionModel,
    REST_LABEL,
    check_session_leakage,
    load_trials,
)

DEFAULT_DATA = "~/.local/share/quietbridge/sequences"
DEFAULT_OUTPUT = "~/.cache/quietbridge/motion_v2.npz"
SESSION_PREFIX = "peter-motion-v2-"
SCORE_GATE = 0.75


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
        "accuracy": (n_correct + n_rest_ok) / n if n else 0.0,
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
        "mean_predict_ms": round(1000.0 * elapsed / n, 2) if n else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="sequences root containing train/ and test/")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="candidate NPZ path (report goes to <name>.report.json)")
    parser.add_argument("--session-prefix", default=SESSION_PREFIX,
                        help="only this session prefix; pass '' for a profile's custom sessions")
    parser.add_argument("--allow-legacy-no-validation", action="store_true",
                        help="reproduce old data without an independent validation session")
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

    train_trials = load_trials(data_dir, "train", args.session_prefix)
    validation_trials = load_trials(data_dir, "validation", args.session_prefix)
    test_trials = load_trials(data_dir, "test", args.session_prefix)
    if not train_trials:
        parser.error(f"no train trials with session prefix {args.session_prefix!r} in {data_dir}")
    if not validation_trials and not args.allow_legacy_no_validation:
        parser.error("record an independent validation session, or explicitly use --allow-legacy-no-validation")
    if validation_trials:
        check_session_leakage(train_trials, validation_trials)
        check_session_leakage(validation_trials, test_trials)
    check_session_leakage(train_trials, test_trials)

    model = MotionModel().fit(train_trials)
    model.save(output)

    report = {
        "model_path": str(output),
        "train_dir": str(data_dir / "train"),
        "test_dir": str(data_dir / "test"),
        "session_prefix": args.session_prefix,
        "train_trials": len(train_trials),
        "validation_trials": len(validation_trials),
        "test_trials": len(test_trials),
        "train_labels": {l: sum(1 for t in train_trials if t["label"] == l)
                         for l in sorted({t["label"] for t in train_trials})},
        "test_labels": {l: sum(1 for t in test_trials if t["label"] == l)
                        for l in sorted({t["label"] for t in test_trials})},
        "thresholds_calibrated_on": "train leave-one-trial-out only",
        "loo": model.loo_report_,
        "validation_ungated": evaluate(model, validation_trials) if validation_trials else None,
        "validation_diagnostics": [
            {"trial_id": trial["trial_id"], "true": trial["label"], **model.diagnose(trial)}
            for trial in validation_trials
        ],
        "test_ungated": evaluate(model, test_trials),
        "test_score_gated_0.75": evaluate(model, test_trials, score_gate=SCORE_GATE),
        "test_diagnostics": [
            {"trial_id": trial["trial_id"], "true": trial["label"], **model.diagnose(trial)}
            for trial in test_trials
        ],
        "notes": [
            "Test split was never used for threshold tuning.",
            "score is a heuristic matching score, not a probability.",
            "The 0.75 score gate is diagnostic only; runtime uses model acceptance.",
            "Model classifies pre-segmented trials only; no continuous recognition.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def print_metrics(name, metrics):
        recall = metrics["positive_recall"]
        silent = metrics["rest_silent_rate"]
        recall_text = f"{recall:.2f}" if recall is not None else "n/a"
        silent_text = f"{silent:.2f}" if silent is not None else "n/a"
        print(f"{name}: acc={metrics['accuracy']:.2f} fp={metrics['false_positives']} "
              f"misses={metrics['misses']} mislabels={metrics['mislabels']} "
              f"positive_recall={metrics['positive_correct']}/{metrics['positive_trials']} "
              f"({recall_text}) rest_silent={metrics['rest_correctly_silent']}/"
              f"{metrics['rest_trials']} ({silent_text})")

    if validation_trials:
        print_metrics("validation", report["validation_ungated"])
    g = report["test_ungated"]
    print(f"train: {report['train_trials']} trials {report['train_labels']}")
    print(f"test:  {report['test_trials']} trials {report['test_labels']}")
    print(f"LOO:   acc={model.loo_report_['raw_accuracy']:.2f} "
          f"reject={model.reject_threshold_:.4f} margin={model.margin_threshold_:.4f}")
    print_metrics("test", g)
    print_metrics("gated@0.75", report["test_score_gated_0.75"])
    print(f"wrote {output} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

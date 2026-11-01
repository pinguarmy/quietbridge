"""Train the personal gesture template model from recorded trials.

Usage:

    .venv/bin/python -m hackathon.signs.train [--data PATH] [--output PATH] [--overwrite]

Data layout (written by the collection step; one .npz per independent trial):

    <data>/train/*.npz   fields: features (N,63) float32, label, session_id, trial_id
    <data>/test/*.npz    same fields; sessions must not overlap with train

The model reads train/ only for fitting and threshold estimation. test/ is
used once, afterwards, for a held-out report. An existing model file is never
overwritten unless --overwrite is passed explicitly. The report is written
next to the model (default under ~/.cache/quietbridge, outside Git).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from hackathon.signs.personal import (
    ALL_LABELS,
    RUNTIME_SCORE_FLOOR,
    PersonalGestureModel,
    assert_disjoint_sessions,
    evaluate_heldout,
)

logger = logging.getLogger("hackathon.signs.train")

DEFAULT_DATA_DIR = Path("~/.local/share/quietbridge/gestures").expanduser()
DEFAULT_MODEL_PATH = Path("~/.cache/quietbridge/personal_gestures.npz").expanduser()


def _scalar_str(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def load_trials(directory: Path) -> list[dict]:
    """Load every trial .npz in one split directory (no pickling)."""
    trials = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            missing = {"features", "label", "session_id", "trial_id"} - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            trials.append(
                {
                    "features": np.asarray(data["features"], dtype=np.float32),
                    "label": _scalar_str(data["label"]),
                    "session_id": _scalar_str(data["session_id"]),
                    "trial_id": _scalar_str(data["trial_id"]),
                }
            )
    return trials


def write_report(
    report_path: Path,
    model: PersonalGestureModel,
    train_trials: list[dict],
    test_trials: list[dict],
    evaluation: dict | None,
) -> None:
    lines = [
        "Personal gesture model — training report",
        "=" * 46,
        "",
        f"Templates: {model.templates_.shape[0]} "
        f"(one median template per independent trial)",
        f"Distance rejection threshold (leave-one-trial-out, train only): "
        f"{model.threshold_:.4f}",
        f"Runtime score floor (same gate as SignRecognizer.MIN_SCORE): "
        f"{RUNTIME_SCORE_FLOOR:.2f}",
        f"No-hand NEUTRAL trials excluded from templates: "
        f"{len(model.no_hand_trial_ids_)} {model.no_hand_trial_ids_}",
        "",
        "Training trials per label:",
    ]
    counts = {label: 0 for label in ALL_LABELS}
    for t in train_trials:
        if t["features"].shape[0] > 0:
            counts[t["label"]] += 1
    for label in ALL_LABELS:
        lines.append(f"  {label}: {counts[label]}")

    lines.append("")
    if evaluation is None:
        lines.append("No held-out test trials found; no evaluation performed.")
    else:
        trial = evaluation["trial_level"]
        frame = evaluation["frame_level"]
        floor = evaluation["runtime_score_floor"]
        lines.append(
            f"Held-out trial level (label only, before the runtime gate): "
            f"{trial['label_correct']}/{trial['total']} correct"
        )
        lines.append(
            f"Held-out trial level (runtime-gated, score >= {floor:.2f}): "
            f"{trial['gated_correct']}/{trial['total']} correct; "
            f"{trial['gated_out']} correct labels were rejected by the score "
            "floor, exactly as the live recognizer would reject them"
        )
        lines.append(
            f"  ({trial['no_hand_trials']} trials had no detected hand at "
            "all; only NEUTRAL ones count as correct — a positive gesture "
            "with zero usable frames is a miss)"
        )
        for label in ALL_LABELS:
            per = trial["per_label"][label]
            if per["trials"]:
                lines.append(
                    f"  {label}: {per['gated_correct']}/{per['trials']} "
                    f"runtime-gated, {per['label_correct']}/{per['trials']} "
                    "label-only"
                )
        lines.append(
            f"Held-out frame level: {frame['gated_correct']}/{frame['total']} "
            f"runtime-gated ({frame['label_correct']}/{frame['total']} "
            "label-only)"
        )
        lines.append("")
        lines.append(
            "Trial-level and frame-level numbers are reported separately on "
            "purpose: frames within one held-out trial are highly correlated, "
            "so the frame rate is the weaker evidence."
        )
        lines.append("")
        lines.append(
            "The runtime gate above applies the same score floor the live "
            "recognizer uses before a label may become an event. Passing it "
            "in this static evaluation does NOT mean real-time events pass: "
            "a live gesture must also survive the temporal stabilizer "
            "(debounce) on camera input. Live acceptance still requires "
            "running the recognizer on real footage and is not claimed here."
        )

    lines += [
        "",
        "Limitations (read before trusting this model):",
        "- Small-sample personal model: templates come from a handful of one",
        "  person's recorded trials. Counts above are not calibrated accuracy",
        "  and say little about other people, cameras, lighting or clothing.",
        "- The rejection score is a heuristic similarity, not a probability.",
        "- The threshold was estimated on training trials only "
        "(leave-one-trial-out);",
        "  with this few trials it is a coarse boundary, not a tuned operating",
        "  point.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"trial root containing train/ and test/ "
                             f"(default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"model output path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing model file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.output.exists() and not args.overwrite:
        print(
            f"refusing to overwrite existing model {args.output}; "
            "pass --overwrite to replace it",
            file=sys.stderr,
        )
        return 2

    train_dir = args.data / "train"
    if not train_dir.is_dir():
        print(f"no training directory at {train_dir}", file=sys.stderr)
        return 2
    train_trials = load_trials(train_dir)
    if not train_trials:
        print(f"no trial .npz files under {train_dir}", file=sys.stderr)
        return 2

    test_dir = args.data / "test"
    test_trials = load_trials(test_dir) if test_dir.is_dir() else []

    try:
        assert_disjoint_sessions(train_trials, test_trials)
        model = PersonalGestureModel().fit(train_trials)
        evaluation = (
            evaluate_heldout(model, train_trials, test_trials) if test_trials else None
        )
    except ValueError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2

    model.save(args.output)
    report_path = args.output.with_suffix(".report.txt")
    write_report(report_path, model, train_trials, test_trials, evaluation)

    print(f"model saved to {args.output}")
    print(f"report saved to {report_path}")
    if evaluation is not None:
        trial = evaluation["trial_level"]
        print(
            f"held-out trial level: {trial['gated_correct']}/{trial['total']} "
            f"runtime-gated ({trial['label_correct']}/{trial['total']} "
            "label-only)"
        )
    print(
        "note: small-sample personal model; the report lists why these "
        "numbers are not calibrated accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fine-tune lip reading on captured mouth trials without changing global models.

The final test split is decoded only after validation has selected stock or the
candidate.  An attempted candidate is always reported and saved, but rejection
never creates a claimed improvement or enables deployment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from lipflow import paths as _lipflow_paths

# One models-folder rule for the lip reader, the downloader and this trainer.
TOKENIZER = Path(_lipflow_paths.MODELS) / "lm" / "unigram5000.model"

#: Refusal limits (engineering defaults, not validated): below these the
#: trainer stops before loading the model and says what to record. With fewer
#: practice clips a 6-epoch adaptation mostly memorises; with fewer check clips
#: a one-word WER difference decides selection.
MIN_FIT_CLIPS = 10
MIN_VALIDATION_CLIPS = 5


class InsufficientData(ValueError):
    """Too little recorded data to train or to judge a candidate."""


def check_minimums(fit: list[dict], val: list[dict]) -> dict:
    """Raise InsufficientData with a concrete next step; return the counts otherwise."""
    counts = {"fit_clips": len(fit), "validation_clips": len(val),
              "fit_phrases": len({x["text"] for x in fit}),
              "validation_words": sum(len(normalise(x["text"]).split()) for x in val),
              "minimums": {"fit_clips": MIN_FIT_CLIPS, "validation_clips": MIN_VALIDATION_CLIPS}}
    problems = []
    if len(fit) < MIN_FIT_CLIPS:
        problems.append(f"record {MIN_FIT_CLIPS - len(fit)} more lip practice tries "
                        f"({len(fit)} of at least {MIN_FIT_CLIPS})")
    if len(val) < MIN_VALIDATION_CLIPS:
        problems.append(f"record {MIN_VALIDATION_CLIPS - len(val)} more lip check tries on another occasion "
                        f"({len(val)} of at least {MIN_VALIDATION_CLIPS})")
    if problems:
        raise InsufficientData("Not enough lip recordings to train a candidate: " + "; ".join(problems) + ".")
    return counts


def _sha256(path) -> str | None:
    import hashlib
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_quality_summary(items: list[dict]) -> dict:
    """Report (never exclude) lip tries whose capture review asked for a decision."""
    flagged = [{"trial_id": x["trial_id"], "text": x["text"], "warnings": x["quality_warnings"],
                "decision": x.get("quality_decision")} for x in items if x.get("quality_warnings")]
    return {"checked": len(items), "flagged": flagged, "used_in_training_or_scoring": True}


def _scalar(z, name: str) -> str:
    value = z[name]
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar metadata")
    return str(value.item())


def load_dataset(root: str | os.PathLike) -> dict[str, list[dict]]:
    """Load train/validation/test NPZs and reject every cross-split ID leak."""
    root = Path(root).expanduser()
    data: dict[str, list[dict]] = {}
    trial_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        items = []
        for path in sorted((root / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                required = {"rois", "text", "session_id", "trial_id"}
                missing = required.difference(z.files)
                if missing:
                    raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
                rois = z["rois"]
                if rois.dtype != np.uint8 or rois.ndim != 3 or rois.shape[1:] != (96, 96) or not len(rois):
                    raise ValueError(f"{path}: rois must be nonempty uint8 (T,96,96)")
                item = {
                    "path": str(path), "rois": rois.copy(), "text": _scalar(z, "text").strip(),
                    "session_id": _scalar(z, "session_id"), "trial_id": _scalar(z, "trial_id"),
                    "split": split,
                }
                if "timestamps_ms" in z.files:
                    timestamps = np.asarray(z["timestamps_ms"], dtype=np.int64)
                    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
                    item.update(
                        raw_frames=len(timestamps),
                        capture_fps=round((len(timestamps) - 1) * 1000 / duration, 3) if duration > 0 else None,
                        max_frame_gap_ms=int(np.diff(timestamps).max()) if len(timestamps) > 1 else 0,
                    )
                from .capture_quality import from_npz, lip_quality
                review = from_npz(z)
                if review is None and "timestamps_ms" in z.files and "face_present" in z.files \
                        and len(z["timestamps_ms"]) == len(z["face_present"]):
                    review = lip_quality(z["timestamps_ms"], z["face_present"])
                item["quality_warnings"] = [w.get("code") for w in (review or {}).get("warnings", [])
                                            if isinstance(w, dict) and w.get("severity", "warn") == "warn"]
                item["quality_decision"] = (review or {}).get("decision")
                if "split" in z.files and _scalar(z, "split") != split:
                    raise ValueError(f"{path}: split metadata does not match its directory")
            if not item["text"] or not item["session_id"] or not item["trial_id"]:
                raise ValueError(f"{path}: text/session_id/trial_id must be nonempty")
            if item["trial_id"] in trial_splits and trial_splits[item["trial_id"]] == split:
                # A copied file: one recording would count twice in fitting or scoring.
                raise ValueError(f"{path}: trial_id {item['trial_id']!r} appears twice in {split}")
            for key, seen in (("trial_id", trial_splits), ("session_id", session_splits)):
                value = item[key]
                if value in seen and seen[value] != split:
                    raise ValueError(f"{key} {value!r} appears in both {seen[value]} and {split}")
                seen[value] = split
            items.append(item)
        data[split] = items
    if not data["train"]:
        raise ValueError(f"no training NPZs found under {root / 'train'}")
    return data


def partition_train(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Legacy same-session split, retained only to reproduce old experiments.

    New training uses the independent validation directory instead.
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["text"], []).append(item)
    fit, val = [], []
    for text in sorted(groups):
        group = sorted(groups[text], key=lambda x: x["path"])
        if len(group) < 2:
            raise ValueError(f"prompt {text!r} needs at least two train trials")
        fit.extend(group[:-1])
        val.append(group[-1])
    return fit, val


def independent_partition(data: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    if not data.get("validation"):
        raise ValueError(
            "Independent validation recordings are required under validation/. "
            "Record a new session instead of splitting validation from training."
        )
    return data["train"], data["validation"]


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^A-Za-z' ]+", " ", text).upper().split())


def target_prompt(items: list[dict]) -> str:
    """The protected target prompt is the longest recorded prompt.

    Report keys keep their historical ``*_welcome`` names for compatibility:
    ``contrast_false_welcome`` is the rate at which a different prompt was
    decoded as this target prompt.
    """
    return max((x["text"] for x in items), key=lambda s: (len(normalise(s).split()), len(s)))


def metrics(results: list[dict], target: str) -> dict:
    from lipflow.bench import wer
    target_norm = normalise(target)
    errors = words = target_errors = target_words = 0
    target_hits = target_n = contrast_false = contrast_n = 0
    for row in results:
        e, n = wer(row["hypothesis"], row["text"])
        errors += e
        words += n
        if normalise(row["text"]) == target_norm:
            target_n += 1
            target_errors += e
            target_words += n
            target_hits += normalise(row["hypothesis"]) == target_norm
        else:
            contrast_n += 1
            contrast_false += normalise(row["hypothesis"]) == target_norm
    # A rate with an empty denominator was not measured: report null, never 0.
    return {
        "count": len(results), "wer_errors": errors, "wer_words": words,
        "wer": errors / words if words else None,
        "fullphrase_exact": target_hits / target_n if target_n else None,
        "fullphrase_exact_counts": [target_hits, target_n],
        "target_wer": target_errors / target_words if target_words else None,
        "contrast_false_welcome": contrast_false / contrast_n if contrast_n else None,
        "contrast_false_welcome_counts": [contrast_false, contrast_n],
        "mean_capture_fps": round(float(np.mean([row["capture_fps"] for row in results
                                if row.get("capture_fps") is not None])), 3)
                            if any(row.get("capture_fps") is not None for row in results) else None,
        # Older recordings have no timestamps: their gap is unknown (None), not 0.
        "max_frame_gap_ms": max((row["max_frame_gap_ms"] for row in results
                                 if row.get("max_frame_gap_ms") is not None), default=None),
    }


def select_candidate(stock: dict, candidate: dict) -> tuple[bool, str]:
    """Selection is frozen before final test and protects the hardest target prompt."""
    if stock.get("wer") is None or candidate.get("wer") is None:
        return False, "rejected: validation has no scorable words"
    stock_contrast = stock.get("contrast_false_welcome")
    candidate_contrast = candidate.get("contrast_false_welcome")
    if stock_contrast is not None and candidate_contrast is not None \
            and candidate_contrast > stock_contrast:
        return False, "rejected: contrast false-welcome rate increased"
    if candidate.get("target_wer") is not None and stock.get("target_wer") is not None \
            and candidate["target_wer"] > stock["target_wer"] + 1e-12:
        return False, "rejected: target-prompt WER regressed"
    if candidate["wer"] >= stock["wer"]:
        return False, "rejected: validation WER did not improve"
    return True, "selected: validation WER improved without target or contrast regression"


def evaluate(reader, items: list[dict], cache: dict[str, dict] | None = None) -> list[dict]:
    results = []
    for item in items:
        if cache is not None and item["path"] in cache:
            results.append(cache[item["path"]])
            continue
        hypothesis, seconds = reader.read(item["rois"], fast=False)
        row = {
            "path": item["path"], "trial_id": item["trial_id"], "text": item["text"],
            "hypothesis": hypothesis, "decode_seconds": float(seconds),
            "capture_fps": item.get("capture_fps"), "raw_frames": item.get("raw_frames"),
            "max_frame_gap_ms": item.get("max_frame_gap_ms"),
        }
        if cache is not None:
            cache[item["path"]] = row
        results.append(row)
    return results


def save_candidate(reader, output: Path, scope: str) -> None:
    import torch

    prefixes = ["encoder.frontend."]
    if scope in ("frontend+embed", "frontend+encoder1"):
        prefixes.append("encoder.embed.")
    if scope == "frontend+encoder1":
        prefixes.append("encoder.encoders.0.")
    state = {k: v.detach().cpu() for k, v in reader.model.state_dict().items()
             if k.startswith(tuple(prefixes))}
    if not state:
        raise RuntimeError(f"no tensors found for scope {scope}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:  # exclusive creation: never overwrite an experiment
        torch.save(state, f)


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(".report.json")
    if output.suffix != ".pth":
        raise ValueError("--output must name a new .pth file")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if not TOKENIZER.is_file():
        raise FileNotFoundError(f"required training tokenizer is missing: {TOKENIZER}")

    started = time.perf_counter()
    data = load_dataset(args.data)
    fit, val = independent_partition(data)
    target = target_prompt(data["train"])
    seed = getattr(args, "seed", 0)
    final_test = not getattr(args, "no_final_test", False)
    report = {
        "data": str(Path(args.data).expanduser().resolve()), "output": str(output),
        "config": {"epochs": args.epochs, "beam": args.beam, "lr": args.lr,
                   "scope": args.scope, "device": args.device, "seed": seed,
                   "augmentation": "train clips only: random 88px crop, horizontal flip, brightness/contrast "
                                   "jitter, 2 time masks (lipflow.train_vsr._augment)"},
        "counts": {"train": len(data["train"]), "fit": len(fit),
                   "validation": len(val), "val": len(val), "test": len(data["test"])},
        "target_prompt": target, "candidate_attempted": False,
        "selection": {"selected": "stock", "deployment": False, "reason": "candidate not completed"},
        "capture_quality": {split: capture_quality_summary(data[split]) for split in ("train", "validation", "test")},
        "tokenizer_sha256": _sha256(TOKENIZER),
        "failures": [],
    }
    try:
        # Before loading the ~1 GB model: too little data fails with what to record.
        report["data_requirements"] = check_minimums(fit, val)
    except InsufficientData as exc:
        report["failures"].append(str(exc))
        report["data_requirements"] = {"fit_clips": len(fit), "validation_clips": len(val),
                                       "minimums": {"fit_clips": MIN_FIT_CLIPS,
                                                    "validation_clips": MIN_VALIDATION_CLIPS}}
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(report_path, report)
        raise
    try:
        from lipflow.train_vsr import finetune
        from lipflow.vsr import LipReader
        reader = LipReader(device=args.device, beam_size=args.beam, personal=False)
        stock_cache: dict[str, dict] = {}
        # Validation alone selects the model. Decoding all fit clips first adds
        # substantial search latency without affecting the frozen selection rule.
        stock_val = evaluate(reader, val, stock_cache)
        report["baseline"] = {
            "val": {"metrics": metrics(stock_val, target), "transcripts": stock_val},
        }
        report["candidate_attempted"] = True
        candidate = finetune(fit, epochs=args.epochs, lr=args.lr, scope=args.scope,
                             reader=reader, seed=seed)
        candidate_val_rows = evaluate(candidate, val)
        candidate_val = metrics(candidate_val_rows, target)
        report["candidate"] = {"val": {"metrics": candidate_val, "transcripts": candidate_val_rows}}
        chosen, reason = select_candidate(report["baseline"]["val"]["metrics"], candidate_val)
        report["selection"] = {"selected": "candidate" if chosen else "stock",
                               "deployment": False, "reason": reason}
        save_candidate(candidate, output, args.scope)
        # Binds this report to the exact weights it describes; promotion checks it.
        report["candidate_sha256"] = _sha256(output)

        if final_test:
            # Frozen selection above: this is the sole final-test evaluation pass.
            # finetune() mutates its reader, so construct a fresh stock reader for
            # the untouched baseline rather than accidentally reusing candidate weights.
            stock_reader = LipReader(device=args.device, beam_size=args.beam, personal=False)
            stock_test_rows = evaluate(stock_reader, data["test"], stock_cache)
            candidate_test_rows = evaluate(candidate, data["test"])
            report["baseline"]["test"] = {"metrics": metrics(stock_test_rows, target),
                                          "transcripts": stock_test_rows}
            report["candidate"]["test"] = {"metrics": metrics(candidate_test_rows, target),
                                           "transcripts": candidate_test_rows}
        else:
            # Recipe experiments compare configurations on validation only; the
            # final check is decoded once, for the chosen configuration, later.
            report["final_test"] = "not evaluated (--no-final-test experiment run; not promotable)"
            report["selection"]["experiment_only"] = True
        report["elapsed_seconds"] = time.perf_counter() - started
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(report_path, report)
        raise
    _write_report(report_path, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="mouth dataset root with train/, validation/ and test/")
    parser.add_argument("--output", required=True, help="new external .pth candidate path (never overwritten)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scope", default="frontend",
                        choices=("frontend", "frontend+embed", "frontend+encoder1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds Python/torch RNGs for augmentation and shuffling (CPU runs reproduce)")
    parser.add_argument("--no-final-test", action="store_true",
                        help="recipe experiments only: skip the final-test decode; the candidate is not promotable")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.beam < 1 or args.lr <= 0:
        parser.error("--epochs/--beam must be positive and --lr must be > 0")
    report = run(args)
    print(json.dumps({"output": report["output"], "report": str(Path(args.output).with_suffix(".report.json")),
                      "selection": report["selection"], "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()

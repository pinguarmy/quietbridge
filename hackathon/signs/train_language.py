"""Train a personal language model candidate for a QuietBridge profile.

CLI:
    python -m hackathon.signs.train_language --data <profile mouths root> --output <new lm.candidate.pth>

Extracts text transcripts from the profile's mouth NPZ recordings (without loading
ROI arrays into memory), validates cross-split session/trial isolation, fine-tunes
the language model on the user's phrases via `lipflow.train_lm.train()`, and writes
a sidecar JSON report alongside the output candidate.

Safety rules:
- Candidate output files and sidecar reports are never overwritten.
- Trial and session IDs must not leak across train and validation splits.
- Candidates are saved only when held-out validation perplexity improves and general English
  text perplexity does not regress beyond the allowed threshold.
- Candidates are NOT automatically deployed. Explicit promotion is required.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _scalar(z: np.lib.npyio.NpzFile, name: str, path: Path) -> str:
    if name not in z.files:
        raise ValueError(f"{path}: missing {name}")
    val = z[name]
    if val.ndim != 0:
        raise ValueError(f"{path}: {name} must be scalar metadata")
    return str(val.item())


def load_text_metadata(root: str | os.PathLike) -> dict[str, list[dict[str, str]]]:
    """Safely load text metadata from train/validation NPZs without reading ROIs into memory."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root_path}")

    data: dict[str, list[dict[str, str]]] = {}
    trial_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}

    splits = ["train", "validation"]
    test_dir = root_path / "test"
    if test_dir.is_dir() and any(test_dir.glob("*.npz")):
        splits.append("test")

    for split in splits:
        folder = root_path / split
        items: list[dict[str, str]] = []
        if folder.is_dir():
            for npz_path in sorted(folder.glob("*.npz")):
                with np.load(npz_path, allow_pickle=False) as z:
                    for req in ("text", "session_id", "trial_id"):
                        if req not in z.files:
                            raise ValueError(f"{npz_path}: missing required metadata '{req}'")
                    text = _scalar(z, "text", npz_path).strip()
                    session_id = _scalar(z, "session_id", npz_path).strip()
                    trial_id = _scalar(z, "trial_id", npz_path).strip()
                    if "split" in z.files:
                        z_split = _scalar(z, "split", npz_path).strip()
                        if z_split != split:
                            raise ValueError(f"{npz_path}: split metadata {z_split!r} does not match directory {split!r}")

                if not text or not session_id or not trial_id:
                    raise ValueError(f"{npz_path}: text/session_id/trial_id must be nonempty")

                for key, val, seen in (
                    ("trial_id", trial_id, trial_splits),
                    ("session_id", session_id, session_splits),
                ):
                    if val in seen and seen[val] != split:
                        raise ValueError(f"{key} {val!r} appears in both {seen[val]} and {split}")
                    seen[val] = split

                items.append({
                    "path": str(npz_path),
                    "text": text,
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "split": split,
                })
        data[split] = items

    if not data.get("train"):
        raise ValueError(f"no training NPZs found under {root_path / 'train'}")
    if not data.get("validation"):
        raise ValueError(f"no validation NPZs found under {root_path / 'validation'}")

    return data


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(".report.json")

    if output.suffix != ".pth":
        raise ValueError("--output must name a new .pth file")

    for p in (output, report_path):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {p}")

    data_root = Path(args.data).expanduser().resolve()
    data = load_text_metadata(data_root)

    train_phrases = [item["text"] for item in data["train"]]
    val_phrases = [item["text"] for item in data["validation"]]

    started = time.perf_counter()
    report: dict[str, Any] = {
        "data": str(data_root),
        "output": str(output),
        "report": str(report_path),
        "counts": {
            "train": len(data["train"]),
            "validation": len(data["validation"]),
        },
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "bs": args.bs,
            "seed": args.seed,
            "max_general_loss": args.max_general_loss,
            "device": getattr(args, "device", None),
        },
        "deployment": False,
        "auto_deploy": False,
        "saved": False,
        "failures": [],
    }
    if "test" in data:
        report["counts"]["test"] = len(data["test"])

    try:
        from lipflow import train_lm

        train_result = train_lm.train(
            epochs=args.epochs,
            lr=args.lr,
            bs=args.bs,
            max_general_loss=args.max_general_loss,
            seed=args.seed,
            phrases=train_phrases,
            validation_phrases=val_phrases,
            output_path=output,
            device=getattr(args, "device", None),
        )

        report["counts"]["train_sentences"] = train_result.get("train", len(data["train"]))
        report["counts"]["validation_sentences"] = train_result.get("val", len(data["validation"]))
        report["perplexity"] = {
            "before": train_result["before"],
            "after": train_result["after"],
        }
        report["before"] = train_result["before"]
        report["after"] = train_result["after"]
        report["saved"] = train_result["saved"]
        report["selection"] = {
            "selected": "candidate" if train_result["saved"] else "stock",
            "deployment": False,
            "reason": (
                "candidate improved validation perplexity without general regression"
                if train_result["saved"]
                else "rejected: validation perplexity did not improve or general text regressed"
            ),
        }
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_report(report_path, report)
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report["elapsed_seconds"] = time.perf_counter() - started
        if not report_path.exists():
            try:
                _write_report(report_path, report)
            except Exception:
                pass
        raise

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="mouth dataset root with train/ and validation/")
    parser.add_argument("--output", required=True, help="new external .pth candidate path (never overwritten)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max-general-loss", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.lr <= 0 or args.bs < 1:
        parser.error("--epochs and --bs must be positive and --lr must be > 0")

    report = run(args)
    print(json.dumps({
        "output": report["output"],
        "report": report["report"],
        "saved": report["saved"],
        "deployment": report["deployment"],
        "auto_deploy": report["auto_deploy"],
        "counts": report["counts"],
        "perplexity": report.get("perplexity", {}),
        "selection": report.get("selection", {}),
    }, indent=2))


if __name__ == "__main__":
    main()

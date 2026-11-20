"""Compare lip fine-tune recipes on the validation session only, then run the final check once.

Hyper-parameters of the lip adaptation (epochs, scope, learning rate) cannot be
judged without the base weights and real recordings, so this module makes the
experiment executable instead of changing the numbers:

1. ``grid``: for each configuration, run ``train_lips`` with ``--no-final-test``
   into a scratch folder. Every run decodes validation only; candidates are
   marked experiment-only and are never promotable.
2. Pick the configuration with the lowest validation WER that also passes the
   trainer's own gate (no target-prompt or contrast regression vs stock).
   Ties keep the earlier (current default) configuration.
3. ``final``: print, or run, the single normal ``train_lips`` command for the
   chosen configuration; only that run decodes the final-check session.

Usage::

    python -m quietbridge.signs.lip_recipe_experiment grid --data PROFILE/mouths --out /tmp/lipexp
    python -m quietbridge.signs.lip_recipe_experiment final --out /tmp/lipexp [--run]

Needs the base VSR weights and >= train_lips.MIN_FIT_CLIPS practice and
MIN_VALIDATION_CLIPS check clips. With few validation words, differences of a
word or two are noise; the summary prints the word counts next to every WER.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Current product default first (train_lips CLI defaults), then the settings the
#: train_vsr docstring reports from held-out speaker-adaptation runs.
CONFIGS = [
    {"name": "current_default", "epochs": 3, "scope": "frontend", "lr": 1e-4},
    {"name": "frontend_6ep", "epochs": 6, "scope": "frontend", "lr": 1e-4},
    {"name": "frontend_encoder1_6ep", "epochs": 6, "scope": "frontend+encoder1", "lr": 1e-4},
    {"name": "frontend_encoder1_3ep", "epochs": 3, "scope": "frontend+encoder1", "lr": 1e-4},
    {"name": "frontend_6ep_lr5e-5", "epochs": 6, "scope": "frontend", "lr": 5e-5},
]


def _command(data, output, config, seed, final):
    cmd = [sys.executable, "-m", "quietbridge.signs.train_lips", "--data", str(data), "--output", str(output),
           "--epochs", str(config["epochs"]), "--scope", config["scope"], "--lr", str(config["lr"]),
           "--seed", str(seed)]
    return cmd if final else cmd + ["--no-final-test"]


def summarize(out: Path) -> dict:
    """Read every grid report; choose on validation only (test is never present)."""
    rows = []
    for config in CONFIGS:
        path = out / config["name"] / "lip.report.json"
        if not path.is_file():
            rows.append({**config, "status": "missing"})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if "test" in (report.get("candidate") or {}):
            raise ValueError(f"{path} decoded the final check; grid runs must use --no-final-test")
        cand = ((report.get("candidate") or {}).get("val") or {}).get("metrics") or {}
        stock = ((report.get("baseline") or {}).get("val") or {}).get("metrics") or {}
        rows.append({**config, "status": "failed" if report.get("failures") else "ok",
                     "validation_wer": cand.get("wer"), "validation_words": cand.get("wer_words"),
                     "stock_validation_wer": stock.get("wer"),
                     "passes_gate": (report.get("selection") or {}).get("selected") == "candidate",
                     "reason": (report.get("selection") or {}).get("reason")})
    eligible = [(i, r) for i, r in enumerate(rows) if r.get("status") == "ok" and r.get("passes_gate")]
    chosen = min(eligible, key=lambda p: (p[1]["validation_wer"], p[0]))[1]["name"] if eligible else None
    return {"rows": rows, "chosen": chosen,
            "rule": "lowest validation WER among configurations that pass train_lips' gate; ties keep the earlier one",
            "note": "Final-check results are not read here. Run `final` once for the chosen configuration."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--data", required=True)
    grid.add_argument("--out", required=True)
    grid.add_argument("--seed", type=int, default=0)
    grid.add_argument("--dry-run", action="store_true", help="print the commands only")
    final = sub.add_parser("final")
    final.add_argument("--out", required=True)
    final.add_argument("--data")
    final.add_argument("--seed", type=int, default=0)
    final.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser()
    if args.action == "grid":
        for config in CONFIGS:
            cmd = _command(args.data, out / config["name"] / "lip.pth", config, args.seed, final=False)
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False)
        if not args.dry_run:
            print(json.dumps(summarize(out), indent=2))
        return 0
    summary = summarize(out)
    print(json.dumps(summary, indent=2))
    if summary["chosen"] is None:
        print("No configuration passed the validation gate; keep the current default.")
        return 1
    config = next(c for c in CONFIGS if c["name"] == summary["chosen"])
    data = args.data or json.loads((out / config["name"] / "lip.report.json").read_text())["data"]
    cmd = _command(data, out / "final" / "lip.pth", config, args.seed, final=True)
    print(" ".join(cmd))
    if args.run:
        return subprocess.run(cmd, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

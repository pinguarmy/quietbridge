"""Run synthetic text through the configured CPA cleanup, without camera or paste.

This is an integration check, not a personal lip-reading accuracy benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from lipflow.cleanup import Cleaner

CASES = [
    {"id": "filler", "candidates": ["I THINK UM WE SHOULD WE SHOULD SEND THE EMAIL TOMORROW"],
     "words": [], "style": "polish"},
    {"id": "negation-number", "candidates": ["I DO NOT WANT TO CHANGE THE MEETING AT FOUR THIRTY"],
     "words": [], "style": "polish"},
    {"id": "proper-noun", "candidates": ["CAN YOU COMPARE JANE STREET AND OPTIVER FOR ME",
                                        "CAN YOU COMPARE JANE STREE AND OPTIVER FOR ME"],
     "words": ["Jane Street", "Optiver"], "style": "faithful"},
    {"id": "dictated-instruction", "candidates": ["CAN YOU SUMMARIZE THE REPORT AND SEND ME AN EMAIL"],
     "words": [], "style": "polish"},
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for case in CASES:
        cleaner = Cleaner("gemini", style=case["style"])
        start = time.perf_counter()
        text = cleaner(case["candidates"], words=case["words"])
        row = {**case, "text": text, "elapsed_ms": round((time.perf_counter() - start) * 1000),
               "result": getattr(cleaner, "last_result", {})}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"kind": "synthetic-text-integration-not-personal-WER", "cases": rows},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

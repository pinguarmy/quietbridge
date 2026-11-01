"""Run a compact, local-crop personalization replay after one practice/training round.

Only candidate TEXT reaches the selected cloud cleanup. Crops stay local.
Results from existing practice clips are a tuning check, not an independent final benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from lipflow.dictation import load_settings
from lipflow.paths import HOME
from lipflow.practice import saved_clips
from lipflow.tune import generate_markdown_report, replay_evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=6, help="Start with six clips; expand only if needed")
    p.add_argument("--output", default=str(Path(HOME) / "tuning"))
    args = p.parse_args()
    clips = saved_clips()
    if not clips:
        raise SystemExit("Record your practice round in Lipflow first; no labeled clips yet.")
    random.Random(1).shuffle(clips)
    clips = clips[:max(1, args.limit)]
    from lipflow.vsr import LipReader
    print(f"Local-only VSR replay: {len(clips)} labeled practice clips", flush=True)
    reader = LipReader(beam_size=4)
    samples = []
    for clip in clips:
        t = time.perf_counter()
        candidates = reader.beam_search(reader.encode(clip["rois"]), nbest=5)
        if isinstance(candidates, str):
            candidates = [candidates]
        samples.append({"id": Path(clip["path"]).stem, "reference": clip["text"],
                        "candidates": candidates, "vsr_ms": round((time.perf_counter() - t) * 1000, 1)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "labeled-candidates.jsonl"
    dataset.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    for backend, style in (("basic", "faithful"), ("gemini", "faithful"), ("gemini", "polish")):
        print(f"Replay {backend}/{style}…", flush=True)
        report = replay_evaluation(samples, backend=backend, style=style)
        report["scope_note"] = "Existing labeled practice clips; not fresh independent held-out user evaluation."
        name = f"{backend}-{style}"
        (output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.md").write_text(generate_markdown_report(report), encoding="utf-8")
        print(f"{name}: WER={report.get('wer')}, risks={report['semantic_risks']['any_risk_rate']}, "
              f"cleanup P50={report['cleanup_latency_ms']['p50']}ms", flush=True)
    print(f"Finished. Reports: {output}")
    print("Now try five NEW short utterances in your target apps; keep negations/numbers in two of them.")


if __name__ == "__main__":
    main()

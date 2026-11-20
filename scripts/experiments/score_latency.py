"""Latency of `LipReader.score_texts` (teacher-forced phrase scoring) vs list size and threads.

For each B segment (cached stock encoder output) the decoder's top text plus N saved phrases
(sentences of session A, 9-25 words: longer than a communication aid's phrases, so a worst case
for length) are scored. "cold" = empty LM cache (first utterance after start); "warm" = the saved
phrases' LM scores are cached (every later utterance: an LM score does not depend on the video).
As in the product, the top text's LM score comes from the beam search that just ran.

    .venv/bin/python scripts/experiments/score_latency.py --threads 1 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 8, 20, 50])
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--short", action="store_true",
                    help="phrases = 3-6 word pieces of the A sentences (closer to a communication aid's phrases)")
    args = ap.parse_args()
    import torch
    S = args.scratch
    reader = C.make_reader(4)
    phrases = sorted({s["reference"] for s in C.load_manifest(os.path.join(S, "manifest", "A.jsonl"))})
    if args.short:
        pieces = []
        for k, p in enumerate(phrases):
            w = p.split()
            n = 3 + k % 4
            pieces += [" ".join(w[i:i + n]) for i in range(0, len(w) - n + 1, n)]
        phrases = pieces
    rows = [json.loads(line) for line in open(os.path.join(S, "results", "decode", "stock", "B__base__beam4.jsonl"))]
    rows = rows[:args.segments]
    report = {"git": C.git_state(), "cpu_count": os.cpu_count(), "segments": len(rows), "short": args.short,
              "phrase_words_mean": float(np.mean([len(p.split()) for p in phrases])), "runs": {}}
    for threads in args.threads:
        torch.set_num_threads(threads)
        timing = {size: {"cold": [], "warm": [], "frames": []} for size in args.sizes}
        for row in rows:
            enc = torch.load(os.path.join(S, "enc", f"stock__{row['id']}__base.pt"))["enc"]
            # As in the product: the beam search runs first and leaves its hypotheses' LM scores
            # in the cache, so the decoder's top text is scored without re-running the LM.
            reader._lm_score_cache.clear()
            top = reader.beam_search_details(enc, nbest=3)[0]["text"]
            seeded = dict(reader._lm_score_cache)
            for size in args.sizes:
                lst = (phrases * 2)[:size]
                reader._lm_score_cache = dict(seeded)  # saved phrases not yet seen: LM runs for them
                t0 = time.perf_counter()
                reader.score_texts(enc, [top] + lst)
                timing[size]["cold"].append(time.perf_counter() - t0)
                t0 = time.perf_counter()  # every later utterance: saved phrases' LM scores cached
                reader.score_texts(enc, [top] + lst)
                timing[size]["warm"].append(time.perf_counter() - t0)
                timing[size]["frames"].append(int(enc.shape[0]))
        for size in args.sizes:
            cold, warm, frames = timing[size]["cold"], timing[size]["warm"], timing[size]["frames"]
            key = f"threads{threads}__n{size}"
            report["runs"][key] = {"threads": threads, "phrases": size, "texts": size + 1,
                                   "cold_ms_median": 1000 * float(np.median(cold)), "cold_ms_max": 1000 * float(max(cold)),
                                   "warm_ms_median": 1000 * float(np.median(warm)), "warm_ms_max": 1000 * float(max(warm)),
                                   "warm_ms_per_phrase": 1000 * float(np.median(warm)) / size,
                                   "encoder_frames_median": float(np.median(frames)), "encoder_frames_max": max(frames)}
            r = report["runs"][key]
            print(f"{key}: cold {r['cold_ms_median']:.0f} ms (max {r['cold_ms_max']:.0f}), warm {r['warm_ms_median']:.0f} ms "
                  f"(max {r['warm_ms_max']:.0f}), {r['warm_ms_per_phrase']:.1f} ms/phrase warm", flush=True)
    C.write_json(args.out, report)


if __name__ == "__main__":
    main()

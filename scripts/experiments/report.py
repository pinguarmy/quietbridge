"""Print the markdown tables used in docs/experiments/real-clip-experiments.md from scratch results.

    .venv/bin/python scripts/experiments/report.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from decode import summarise  # noqa: E402


def rows(path):
    return [json.loads(l) for l in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--pair", nargs=2, metavar=("BASE", "OTHER"), action="append", default=[],
                    help="extra paired comparison of two runs, e.g. stock/B__fps20__beam4 stock/B__fps20_resamplenearest__beam4")
    args = ap.parse_args()
    S = args.scratch
    print("| run | n segs | words | WER % [95% CI] | exact % | oracle@3 exact % | oracle@all exact % | oracle@all WER % | s per s video |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(os.path.join(S, "results", "decode", "*", "*.jsonl"))):
        r = rows(p)
        if not r:
            continue
        s = summarise(r)
        tag = os.path.basename(os.path.dirname(p))
        print(f"| {tag}/{os.path.basename(p)[:-6]} | {s['n']} | {s['words']} | {C.fmt(s['wer'])} | {C.fmt(s['exact'])} | "
              f"{C.fmt(s['oracle3_exact'])} | {C.fmt(s['oracleall_exact'])} | {C.fmt(s['oracleall_wer'])} | "
              f"{s['sec_per_sec']['value']:.2f} |")
    # paired difference vs stock base, same segments
    base = {}
    for p in glob.glob(os.path.join(S, "results", "decode", "stock", "*__base__beam4.jsonl")):
        for r in rows(p):
            base[r["id"]] = r
    print("\nPaired WER difference vs stock base beam 4 (same segments; bootstrap over segments)\n")
    print("| run | n | delta WER points [95% CI] |")
    print("|---|---|---|")
    import numpy as np
    for p in sorted(glob.glob(os.path.join(S, "results", "decode", "*", "*.jsonl"))):
        r = [x for x in rows(p) if x["id"] in base]
        if not r or p.endswith("__base__beam4.jsonl") and "/stock/" in p:
            continue
        d = np.array([(C.errors(x["reference"], x["hyps"][0]["text"] if x["hyps"] else "")[0]
                       - C.errors(x["reference"], base[x["id"]]["hyps"][0]["text"] if base[x["id"]]["hyps"] else "")[0],
                       C.errors(x["reference"], "")[1]) for x in r], dtype=float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(d), (2000, len(d)))
        st = d[:, 0][idx].sum(1) / d[:, 1][idx].sum(1)
        lo, hi = np.percentile(st, [2.5, 97.5])
        tag = os.path.basename(os.path.dirname(p))
        print(f"| {tag}/{os.path.basename(p)[:-6]} | {len(d)} | {100 * d[:, 0].sum() / d[:, 1].sum():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    for a, b in args.pair:
        ra = {x["id"]: x for x in rows(os.path.join(S, "results", "decode", a + ".jsonl"))}
        rb = [x for x in rows(os.path.join(S, "results", "decode", b + ".jsonl")) if x["id"] in ra]
        top = lambda x: x["hyps"][0]["text"] if x["hyps"] else ""
        d = np.array([(C.errors(x["reference"], top(x))[0] - C.errors(x["reference"], top(ra[x["id"]]))[0],
                       len(x["reference"].split())) for x in rb], dtype=float)
        idx = np.random.default_rng(0).integers(0, len(d), (2000, len(d)))
        st = d[:, 0][idx].sum(1) / d[:, 1][idx].sum(1)
        lo, hi = np.percentile(st, [2.5, 97.5])
        print(f"| {b} vs {a} | {len(d)} | {100 * d[:, 0].sum() / d[:, 1].sum():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    # length buckets on base
    print("\nStock beam 4 WER by utterance length (words), sessions pooled\n")
    print("| bucket | n | WER % [95% CI] |")
    print("|---|---|---|")
    for lo_, hi_ in ((0, 12), (12, 18), (18, 99)):
        b = [C.errors(x["reference"], x["hyps"][0]["text"] if x["hyps"] else "") for x in base.values()
             if lo_ <= len(x["reference"].split()) < hi_]
        print(f"| {lo_}-{hi_ - 1} words | {len(b)} | {C.fmt(C.bootstrap(b))} |")


if __name__ == "__main__":
    main()

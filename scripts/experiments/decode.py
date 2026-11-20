"""Decode manifest segments with the real VSR model and summarise WER / exact / n-best oracle.

Product settings (quietbridge/lips_adapter.py at the recorded commit): LipReader(beam_size=4,
personal=False) -> defaults ctc_weight 0.1, LM weight 0.3, length bonus 0; the UI keeps the first 3
distinct non-repetitive candidates. Encoder outputs are cached per (model tag, segment, condition), so
beam / LM sweeps do not redo face tracking or encoding.

    .venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/A.jsonl --beam 4 20
    .venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --preset robustness
    .venv/bin/python scripts/experiments/decode.py --manifest ... --vsr-delta d.pth --tag adaptA   # exp 4
    .venv/bin/python scripts/experiments/decode.py --manifest ... --lm personal_lm.pth --tag plm   # exp 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

PRESETS = {
    "base": [None],
    # ordered by priority: the run is time-boxed and whatever completes is reported
    "robustness": [{"fps": 12}, {"fps": 8}, {"fps": 15}, {"gap": 0.4}, {"trim_start": 0.3}, {"trim_end": 0.3},
                   {"shift": (0, 6)}, {"scale": 1.1}, {"gap": 0.8}, {"fps": 20}, {"gap": 0.2}, {"shift": (6, 0)},
                   {"scale": 0.9}],
    "lowfps": [{"fps": 12}],
    "fps20": [{"fps": 20}],
    # product resampling (searchsorted-left: the first frame at/after each 25 fps tick) vs nearest frame
    "resample": [{"resample": "nearest"}, {"fps": 20, "resample": "nearest"}, {"fps": 12, "resample": "nearest"}],
    # follow-up: cadences the product sees (browser uploading every frame at ~30 fps with clock jitter; the
    # old irregular ~20 fps delivery), each with the product resampler and with nearest-frame resampling
    "cadence": [{"jitter": 4}, {"jitter": 4, "resample": "nearest"}, {"keep": 0.67, "jitter": 4},
                {"keep": 0.67, "jitter": 4, "resample": "nearest"}],
}


def repetitive(text: str) -> bool:  # same guard as lips_adapter.finish()
    w = text.upper().split()
    grams = [tuple(w[i:i + 6]) for i in range(max(0, len(w) - 5))]
    return len(grams) != len(set(grams))


def summarise(rows: list[dict], product_n: int = 3) -> dict:
    wer = [C.errors(r["reference"], r["hyps"][0]["text"] if r["hyps"] else "") for r in rows]
    exact = [(float(C.norm(r["reference"]) == C.norm(r["hyps"][0]["text"] if r["hyps"] else "")), 1) for r in rows]
    def oracle(n):
        best = []
        for r in rows:
            hs = [h["text"] for h in r["hyps"][:n]] or [""]
            best.append(min((C.errors(r["reference"], h) for h in hs), key=lambda e: e[0]))
        hit = [(float(e[0] == 0), 1) for e in best]
        return C.bootstrap(best), C.bootstrap(hit)
    o3w, o3e = oracle(product_n)
    oaw, oae = oracle(10 ** 6)
    rtf = [(r["encode_s"] + r["search_s"], r["duration"]) for r in rows]
    return {"n": len(rows), "words": sum(e[1] for e in wer), "wer": C.bootstrap(wer), "exact": C.bootstrap(exact),
            "oracle3_wer": o3w, "oracle3_exact": o3e, "oracleall_wer": oaw, "oracleall_exact": oae,
            "mean_hyps": sum(len(r["hyps"]) for r in rows) / max(len(rows), 1),
            "sec_per_sec": C.bootstrap(rtf), "search_sec_per_sec": C.bootstrap([(r["search_s"], r["duration"]) for r in rows])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--beam", type=int, nargs="+", default=[4])
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--tag", default="stock", help="names the encoder cache and results (change with --vsr-delta)")
    ap.add_argument("--enc-tag", help="encoder cache tag (default: --tag); share it when only the LM changes")
    ap.add_argument("--vsr-delta", help="partial state dict (personal lip model) loaded over the stock model")
    ap.add_argument("--lm", help="personal LM checkpoint, scored beside the stock LM (LipReader)")
    ap.add_argument("--lm-mode", choices=["shared", "additive"],
                    help="personal LM weighting (lipflow.vsr.lm_weights); Experiment 5 used 'additive' "
                         "(then the only behaviour); default: LipReader's default")
    ap.add_argument("--personal-weight", type=float,
                    help="additive: extra weight (0.2); shared: the personal share of the LM weight (0.5)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    segs = C.load_manifest(args.manifest)[: args.limit or None]
    session = segs[0]["session"]
    clip = C.ClipCache(os.path.join(args.scratch, "cache", session))
    conds = PRESETS[args.preset]
    out_dir = os.path.join(args.scratch, "results", "decode", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    summary_all = {"git": C.git_state(), "manifest": args.manifest, "tag": args.tag, "vsr_delta": args.vsr_delta,
                   "lm": args.lm, "lm_mode": args.lm_mode, "personal_weight": args.personal_weight, "runs": {}}
    for beam in args.beam:
        reader = C.make_reader(beam, args.lm, args.lm_mode, args.personal_weight)
        if args.vsr_delta:
            import torch
            res = reader.model.load_state_dict(torch.load(args.vsr_delta, map_location="cpu", weights_only=True),
                                               strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
        for cond in conds:
            key = f"{session}__{C.cond_key(cond)}__beam{beam}"
            path = os.path.join(out_dir, key + ".jsonl")
            rows = []
            if os.path.exists(path):
                rows = [json.loads(l) for l in open(path)]
            done = {r["id"] for r in rows}
            with open(path, "a") as f:
                for seg in segs:
                    if seg["id"] in done:
                        continue
                    enc, raw, enc_s = C.encode_cached(reader, clip, seg, cond, os.path.join(args.scratch, "enc"),
                                                      args.enc_tag or args.tag)
                    if enc is None:
                        hyps, search_s = [], 0.0
                    else:
                        t0 = time.time()
                        hyps = reader.beam_search_details(enc, nbest=10 ** 6)
                        search_s = time.time() - t0
                    hyps = [h for h in hyps if not repetitive(h["text"])]
                    row = {"id": seg["id"], "reference": seg["reference"], "duration": seg["end"] - seg["start"],
                           "raw_frames": raw, "encode_s": enc_s, "search_s": search_s, "hyps": hyps,
                           "cond": C.cond_key(cond), "beam": beam}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            s = summarise(rows)
            summary_all["runs"][key] = s
            print(f"{key}: n={s['n']} words={s['words']} WER {C.fmt(s['wer'])} exact {C.fmt(s['exact'])} "
                  f"oracle@3 exact {C.fmt(s['oracle3_exact'])} oracle@all WER {C.fmt(s['oracleall_wer'])} "
                  f"exact {C.fmt(s['oracleall_exact'])} hyps {s['mean_hyps']:.1f} "
                  f"s/s {s['sec_per_sec']['value']:.2f} (search {s['search_sec_per_sec']['value']:.2f})", flush=True)
    C.write_json(os.path.join(out_dir, f"summary__{session}__{args.preset}__beam{'-'.join(map(str, args.beam))}.json"),
                 summary_all)


if __name__ == "__main__":
    main()

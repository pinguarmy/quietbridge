"""Follow-up to Experiment 2: the PRODUCT's model-scored phrase offers vs the shipped text matcher.

Simulates exactly what the runtime does for one lip result, with the product's own functions:
`phrase_match.phrases_to_score` (prefilter / score-all), `LipReader.score_texts` (normalised
attention + CTC + LM score, also for the decoder's top text) and `phrase_match.model_offers`
(offer iff score >= top score - margin, best first, at most 2). Baseline: `phrase_match.suggest`
(the text-only behaviour that shipped before). Candidates: the product's 3 lip candidates (beam 4).

Saved-phrase scenarios per segment (sentences are the sessions' references; variants are
constructed one-word edits of the segment's reference: 2 substitutions, 2 deletions, 2 inserted NOT):
  open      a random half of the segment's session + every sentence of the other session (10 splits)
  var1_out  ONE variant (not the reference) + the other session's sentences (6 lists: each variant)
  var_in    the reference + all 6 variants + the other session's sentences
  var_out   all 6 variants (not the reference) + the other session's sentences

Scores are computed once per (segment, text) and cached in --out-dir, so a margin sweep needs no model.
The margin is chosen on --select (A) with a rule fixed before looking (see SELECTION_RULE); the test
session is then evaluated once at that frozen margin.

    .venv/bin/python scripts/experiments/model_offers.py --score A B            # model, cached
    .venv/bin/python scripts/experiments/model_offers.py --select A             # sweep, choose
    .venv/bin/python scripts/experiments/model_offers.py --evaluate B --margin M  # once
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

MARGINS = [-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
SPLITS = 10
SELECTION_RULE = ("largest open-set rescue rate on the selection session subject to: (i) for every scenario, "
                  "results whose top-1 is already exactly right get an offer in <= 10% of rows, and (ii) open-set "
                  "rows whose sentence is not saved get any offer in <= 5% of rows; ties -> the smallest margin")


def variants(seg: dict, vocab: list[str]) -> list[tuple[str, str]]:
    w = seg["reference"].split()
    out = []
    for i, kind in enumerate(("substitute", "substitute", "delete", "delete", "insert_not", "insert_not")):
        rng = random.Random(f"{seg['id']}-{i}")
        k = rng.randrange(len(w))
        v = (w[:k] + [rng.choice(vocab)] + w[k + 1:] if kind == "substitute" else
             w[:k] + w[k + 1:] if kind == "delete" else w[:k] + ["NOT"] + w[k:])
        text = " ".join(v)
        if text != seg["reference"] and text not in {t for _, t in out}:
            out.append((kind, text))
    return out


def load(S, cond):
    mani = {k: C.load_manifest(os.path.join(S, "manifest", f"{k}.jsonl")) for k in ("A", "B")}
    dec = {k: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(
        S, "results", "decode", "stock", f"{k}__{cond}__beam4.jsonl"))} for k in ("A", "B")}
    vocab = sorted({w for k in mani for s in mani[k] for w in s["reference"].split() if len(w) > 3})
    return mani, dec, vocab


def score(args):
    import torch
    S = args.scratch
    mani, dec, vocab = load(S, args.cond)
    reader = C.make_reader(4)
    sentences = sorted({s["reference"] for k in mani for s in mani[k]})
    for sess in args.score:
        path = os.path.join(args.out_dir, f"scores__{sess}__{args.cond}.json")
        out = json.load(open(path)) if os.path.exists(path) else {}
        for seg in mani[sess]:
            if seg["id"] in out:
                continue
            row = dec[sess][seg["id"]]
            cands = [h["text"] for h in row["hyps"][:3]]
            enc = torch.load(os.path.join(S, "enc", f"stock__{seg['id']}__{args.cond}.pt"))["enc"]
            texts = list(dict.fromkeys([*cands[:1], *sentences, *(t for _, t in variants(seg, vocab))]))
            res = reader.score_texts(enc, texts)
            out[seg["id"]] = {t: (r["normalized"] if r else None) for t, r in zip(texts, res)}
            print(sess, seg["id"], len(texts), flush=True)
            C.write_json(path, out)


def simulate(cands, plist, scores, margins):
    """Product path for one result and phrase list; the prefilter does not depend on the margin."""
    import quietbridge.phrase_match as pm
    idx = pm.phrases_to_score(cands, plist)
    sc = {i: scores[plist[i]] for i in idx}
    return {m: [o.phrase for o in pm.model_offers(cands, plist, sc, scores[cands[0]], margin=m)]
            for m in margins}, len(idx)


def text_offers(cands, plist):
    import quietbridge.phrase_match as pm
    return [m.phrase for m in pm.suggest(cands, plist)]


def evaluate(sess, other, mani, dec, vocab, scores, margins):
    """Per margin (and 'text' for the shipped matcher): per-segment (num, den) pairs per metric."""
    own = sorted({s["reference"] for s in mani[sess]})
    others = sorted({s["reference"] for s in mani[other]})
    res = {m: {} for m in [*margins, "text"]}
    scored_counts = []

    def add(m, metric, seg_id, num, den=1):
        d = res[m].setdefault(metric, {}).setdefault(seg_id, [0, 0])
        d[0] += num
        d[1] += den

    for seg in mani[sess]:
        row = dec[sess][seg["id"]]
        cands = [h["text"] for h in row["hyps"][:3]]
        if not cands:
            continue
        ref = seg["reference"]
        top1 = C.norm(cands[0]) == C.norm(ref)
        in_nbest = any(C.norm(c) == C.norm(ref) for c in cands)
        sc = scores[seg["id"]]
        var = variants(seg, vocab)
        lists = []
        for s in range(SPLITS):
            rng = random.Random(1000 + s)
            saved = set(rng.sample(own, len(own) // 2))
            plist = sorted(saved) + others
            rng.shuffle(plist)
            lists.append(("open", plist, ref in saved))
        for kind, v in var:
            lists.append(("var1_out", [v] + others, False))
        lists.append(("var_in", [ref] + [v for _, v in var] + others, True))
        lists.append(("var_out", [v for _, v in var] + others, False))
        for scen, plist, in_list in lists:
            by_margin, n_scored = simulate(cands, plist, sc, margins)
            scored_counts.append(n_scored)
            by_margin["text"] = text_offers(cands, plist)
            for m, offered in by_margin.items():
                hit = any(C.norm(o) == C.norm(ref) for o in offered)
                first = bool(offered) and C.norm(offered[0]) == C.norm(ref)
                wrong = [o for o in offered if C.norm(o) != C.norm(ref)]
                key = "top1_right" if top1 else "top1_wrong"
                add(m, f"{scen}__{key}__any_offer", seg["id"], bool(offered))
                add(m, f"{scen}__{key}__wrong_offer", seg["id"], bool(wrong))
                if in_list and not top1:
                    add(m, f"{scen}__rescue", seg["id"], hit)
                    add(m, f"{scen}__rescue_first", seg["id"], first)
                    if not in_nbest:  # the reference is not already one of the 3 lip candidates
                        add(m, f"{scen}__rescue_possible", seg["id"], hit)
                if not in_list:
                    add(m, f"{scen}__false_offer", seg["id"], bool(offered))
                add(m, f"{scen}__all__any_offer", seg["id"], bool(offered))
    out = {}
    for m, metrics in res.items():
        out[str(m)] = {k: C.bootstrap([tuple(v) for v in segs.values()]) | {"rows": sum(v[1] for v in segs.values())}
                       for k, segs in metrics.items()}
    return out, scored_counts


def choose(summary, margins):
    feasible = []
    for m in margins:
        s = summary[str(m)]
        right = [v["value"] for k, v in s.items() if "__top1_right__any_offer" in k and v["value"] is not None]
        false = s.get("open__false_offer", {}).get("value")
        rescue = s.get("open__rescue", {}).get("value")
        if right and max(right) <= 0.10 and false is not None and false <= 0.05 and rescue is not None:
            feasible.append((-rescue, m))
    return min(feasible)[1] if feasible else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cond", default="base")
    ap.add_argument("--score", nargs="*", default=[])
    ap.add_argument("--select")
    ap.add_argument("--evaluate")
    ap.add_argument("--margin", type=float, help="default: phrase_match.MODEL_OFFER_MARGIN (the frozen product value)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.score:
        score(args)
    mani, dec, vocab = load(args.scratch, args.cond)
    import quietbridge.phrase_match as pm
    if args.margin is None:
        args.margin = pm.MODEL_OFFER_MARGIN
    for sess, margins, tag in ((args.select, MARGINS, "select"), (args.evaluate, [args.margin], "evaluate")):
        if not sess:
            continue
        other = "B" if sess == "A" else "A"
        scores = json.load(open(os.path.join(args.out_dir, f"scores__{sess}__{args.cond}.json")))
        summary, scored = evaluate(sess, other, mani, dec, vocab, scores, margins)
        report = {"git": C.git_state(), "session": sess, "cond": args.cond, "role": tag, "summary": summary,
                  "phrase_match_sha256": C.sha256(os.path.join(C.ROOT, "quietbridge", "phrase_match.py")),
                  "constants": {"SCORE_ALL_PHRASES": pm.SCORE_ALL_PHRASES, "SCORE_MAX_PHRASES": pm.SCORE_MAX_PHRASES,
                                "PREFILTER_MAX_RATIO": pm.PREFILTER_MAX_RATIO},
                  "phrases_scored_per_result": {"min": min(scored), "max": max(scored),
                                                "mean": sum(scored) / len(scored)} if scored else None}
        if tag == "select":
            report["selection_rule"] = SELECTION_RULE
            report["chosen_margin"] = choose(summary, margins)
            print("chosen margin:", report["chosen_margin"])
        C.write_json(os.path.join(args.out_dir, f"offers__{tag}__{sess}__{args.cond}.json"), report)
        keys = ["open__rescue", "open__rescue_possible", "open__rescue_first", "open__top1_wrong__wrong_offer", "open__top1_right__any_offer",
                "open__false_offer", "var1_out__top1_right__any_offer", "var1_out__top1_wrong__any_offer",
                "var_in__rescue_first", "var_in__top1_wrong__wrong_offer", "var_in__top1_right__any_offer",
                "var_out__top1_right__any_offer", "var_out__top1_wrong__any_offer"]
        for m in [*margins, "text"]:
            s = summary[str(m)]
            print(f"[{sess} {args.cond}] margin {m}: " + " | ".join(
                f"{k} {C.fmt(s[k])} (n={s[k]['rows']})" for k in keys if k in s), flush=True)


if __name__ == "__main__":
    main()

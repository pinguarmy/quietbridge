"""Experiment 2: using the person's known phrases at decode time.

(a) Text level: the product's `quietbridge.phrase_match.suggest` (if it imports; else a plain
    normalised word-edit-distance matcher) over the product's 3 lip candidates. Saved-phrase list =
    a random half of the TEST session's sentences + all DEV-session sentences as distractors; repeated
    over --splits random halves. Threshold sweep on MAX_DISTANCE_RATIO; the operating point is picked on
    the DEV session (roles swapped) and only reported on TEST.
(b) Model level: every phrase scored under the VSR model for the segment, combined like the beam
    search: 0.9*attention + 0.1*CTC (+0.3*LM). Variant chosen on DEV. Closed-set accuracy vs list size,
    and whether best-vs-second margin separates "intended sentence in list" from "not in list".

    .venv/bin/python scripts/experiments/phrase_experiments.py --dev A --test B
    .venv/bin/python scripts/experiments/phrase_experiments.py --dev A --test B --cond fps12   # 2c
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

WEIGHTS = {"decoder": 0.9, "ctc": 0.1, "lm": 0.3}  # LipReader defaults used by the product


# ------------------------------------------------------------------ (a) text level
def plain_suggest(cands, phrases, ratio):
    out = []
    for p in phrases:
        pk = C.norm(p)
        best = min((C.edit_distance(pk, C.norm(c)) / max(len(pk), len(C.norm(c)), 1) for c in cands if c.strip()),
                   default=9)
        if best <= ratio and not any(C.norm(c) == pk for c in cands):
            out.append((best, p))
    return [p for _, p in sorted(out)[:2]]


def text_level(rows_test, phrases_dev, ratios, splits, use_product, no_lookalike=False, rescue_rows=None):
    import importlib
    pm = importlib.import_module("quietbridge.phrase_match") if use_product else None
    if pm is not None:
        # Plain word edit distance = no look-alike discount. Patch lookalike() rather than setting
        # LOOKALIKE_COST=1.0: _align counts `cost == LOOKALIKE_COST` as a look-alike word, so 1.0 would
        # make every substitution a "look-alike" and loosen the short-phrase rule.
        pm.__dict__.setdefault("_orig_lookalike", pm.lookalike)
        pm.lookalike = (lambda a, b: False) if no_lookalike else pm._orig_lookalike
        shipped = pm.MAX_DISTANCE_RATIO if not hasattr(pm, "_shipped_ratio") else pm._shipped_ratio
        pm._shipped_ratio = shipped
    test_sents = sorted({r["reference"] for r in rows_test})
    out = {}
    for ratio in ratios:
        per_seg = {r["id"]: [] for r in rows_test}
        for s in range(splits):
            rng = random.Random(1000 + s)
            saved = set(rng.sample(test_sents, len(test_sents) // 2))
            plist = sorted(saved) + sorted(phrases_dev)
            rng.shuffle(plist)
            for r in rows_test:
                cands = [h["text"] for h in r["hyps"][:3]]
                in_list = r["reference"] in saved
                top1 = bool(cands) and C.norm(cands[0]) == C.norm(r["reference"])
                if pm is not None:
                    pm.MAX_DISTANCE_RATIO = ratio
                    offered = [m.phrase for m in pm.suggest(cands, plist)]
                else:
                    offered = plain_suggest(cands, plist, ratio)
                hit = any(C.norm(o) == C.norm(r["reference"]) for o in offered)
                if pm is not None and rescue_rows is not None and ratio == pm._shipped_ratio:
                    rescue_rows.append(pm.rescue_outcome(cands, r["reference"], plist))
                per_seg[r["id"]].append({"in": in_list, "top1": top1, "offered": len(offered), "hit": hit,
                                         "nbest": any(C.norm(c) == C.norm(r["reference"]) for c in cands)})
        out[ratio] = per_seg
    return out


def summarise_text(per_seg):
    segs = list(per_seg.values())
    def pairs(f, cond):
        return [(sum(f(o) for o in s if cond(o)), sum(1 for o in s if cond(o))) for s in segs]
    return {
        "top1_exact": C.bootstrap(pairs(lambda o: o["top1"], lambda o: True)),
        "in_list_top1_or_offer": C.bootstrap(pairs(lambda o: o["top1"] or o["hit"], lambda o: o["in"])),
        "rescue_when_top1_wrong": C.bootstrap(pairs(lambda o: o["hit"], lambda o: o["in"] and not o["top1"])),
        "in_list_wrong_offer": C.bootstrap(pairs(lambda o: o["offered"] > 0 and not o["hit"], lambda o: o["in"])),
        "false_offer_not_in_list": C.bootstrap(pairs(lambda o: o["offered"] > 0, lambda o: not o["in"])),
        "n_in": sum(1 for s in segs for o in s if o["in"]), "n_out": sum(1 for s in segs for o in s if not o["in"]),
    }


def near_miss(rows_test, phrases_dev, splits):
    """Hard negatives (constructed): the intended sentence is NOT saved, but a one-word variant of it is
    (substitution, deletion, or an inserted NOT). Counts how often phrase_match offers that wrong variant,
    and whether it flags sensitive_difference. Constructed text, not recorded speech."""
    import quietbridge.phrase_match as pm
    vocab = sorted({w for p in phrases_dev for w in p.split() if len(w) > 3})
    out = {}
    for kind in ("substitute", "delete", "insert_not"):
        per = []
        for r in rows_test:
            hits, flagged, n = 0, 0, 0
            for s in range(splits):
                rng = random.Random(hash((r["id"], kind, s)) & 0xffffffff)
                w = r["reference"].split()
                k = rng.randrange(len(w))
                if kind == "substitute":
                    v = w[:k] + [rng.choice(vocab)] + w[k + 1:]
                elif kind == "delete":
                    v = w[:k] + w[k + 1:]
                else:
                    v = w[:k] + ["NOT"] + w[k:]
                variant = " ".join(v)
                plist = [variant] + sorted(phrases_dev)
                rng.shuffle(plist)
                offered = pm.suggest([h["text"] for h in r["hyps"][:3]], plist)
                bad = [m for m in offered if C.norm(m.phrase) == C.norm(variant)]
                hits += bool(bad)
                flagged += bool(bad) and bad[0].sensitive_difference
                n += 1
            per.append((hits, n, flagged))
        out[kind] = {"wrong_variant_offered": C.bootstrap([(h, n) for h, n, _ in per]),
                     "flagged_sensitive_among_offered": sum(f for _, _, f in per) / max(sum(h for h, _, _ in per), 1)}
    return out


# ------------------------------------------------------------------ (b) model level
class PhraseScorer:
    def __init__(self, reader):
        import sentencepiece as spm
        import torch
        from lipflow.train_lm import SPM
        self.torch = torch
        self.reader = reader
        self.sp = spm.SentencePieceProcessor(model_file=SPM)
        self.ids = {t: i for i, t in enumerate(reader.token_list)}
        self.eos = len(reader.token_list) - 1
        self.lm = reader.beam.full_scorers.get("lm")
        self._lm_cache = {}

    def tok(self, text):
        return [self.ids.get(p, self.ids["<unk>"]) for p in self.sp.encode(" ".join(C.norm(text)), out_type=str)]

    def lm_score(self, text):
        torch = self.torch
        if text not in self._lm_cache:
            ys = self.tok(text)
            x = torch.tensor([[self.eos] + ys]); t = torch.tensor([ys + [self.eos]])
            with torch.inference_mode():
                _, nll, _ = self.lm(x, t)
            self._lm_cache[text] = -float(nll)
        return self._lm_cache[text]

    def score(self, enc, phrases, batch=64):
        """Per phrase: dict(att, ctc, lm, ntok) log-probs (sums incl. <eos>)."""
        torch = self.torch
        import torch.nn.functional as F
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
        m = self.reader.model
        res = []
        with torch.inference_mode():
            logp_ctc = m.ctc.ctc_lo(enc.unsqueeze(0)).log_softmax(-1).transpose(0, 1)  # (T,1,V)
            for k in range(0, len(phrases), batch):
                chunk = phrases[k:k + batch]
                toks = [self.tok(p) for p in chunk]
                L = max(len(t) for t in toks) + 1
                ys_in = torch.full((len(chunk), L), self.eos, dtype=torch.long)
                for i, t in enumerate(toks):
                    ys_in[i, 1:len(t) + 1] = torch.tensor(t)
                mask = subsequent_mask(L).unsqueeze(0)
                mem = enc.unsqueeze(0).expand(len(chunk), -1, -1)
                pred, _ = m.decoder(ys_in, mask, mem, None)
                lp = pred.log_softmax(-1)
                tgt = torch.tensor([t + [self.eos] + [0] * (L - len(t) - 1) for t in toks])
                g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                valid = torch.tensor([[j <= len(t) for j in range(L)] for t in toks])
                att = (g * valid).sum(1)
                T = logp_ctc.shape[0]
                flat = torch.tensor([x for t in toks for x in t])
                ctc = -F.ctc_loss(logp_ctc.expand(T, len(chunk), -1), flat, torch.full((len(chunk),), T),
                                  torch.tensor([len(t) for t in toks]), blank=0, reduction="none",
                                  zero_infinity=False)
                for i, p in enumerate(chunk):
                    res.append({"att": float(att[i]), "ctc": float(ctc[i]), "lm": self.lm_score(p),
                                "ntok": len(toks[i]) + 1})
        return res


VARIANTS = {
    "att+ctc": lambda s: WEIGHTS["decoder"] * s["att"] + WEIGHTS["ctc"] * s["ctc"],
    "att+ctc+lm": lambda s: WEIGHTS["decoder"] * s["att"] + WEIGHTS["ctc"] * s["ctc"] + WEIGHTS["lm"] * s["lm"],
    "att+ctc/len": lambda s: (WEIGHTS["decoder"] * s["att"] + WEIGHTS["ctc"] * s["ctc"]) / s["ntok"],
    "att+ctc+lm/len": lambda s: (WEIGHTS["decoder"] * s["att"] + WEIGHTS["ctc"] * s["ctc"]
                                 + WEIGHTS["lm"] * s["lm"]) / s["ntok"],
}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def model_level(segs, encs, all_sents, scorer, sizes, splits, test_sents):
    """Score all sentences once per segment, then evaluate lists by subsetting the cached scores."""
    t_cost = {}
    scores = {}
    for seg in segs:
        enc = encs[seg["id"]]
        t0 = time.time()
        sc = scorer.score(enc, all_sents)
        t_cost.setdefault(len(all_sents), []).append(time.time() - t0)
        scores[seg["id"]] = dict(zip(all_sents, sc))
    for size in (10, 50):  # measured cost of scoring a smaller list
        if size < len(all_sents):
            seg = segs[0]
            t0 = time.time()
            scorer.score(encs[seg["id"]], all_sents[:size])
            t_cost.setdefault(size, []).append(time.time() - t0)
    results = {}
    for vname, f in VARIANTS.items():
        closed = {}
        for size in sizes:
            per = []
            for seg in segs:
                ok = 0
                for s in range(splits):
                    rng = random.Random(hash((seg["id"], size, s)) & 0xffffffff)
                    others = [x for x in all_sents if x != seg["reference"]]
                    lst = [seg["reference"]] + (rng.sample(others, min(size - 1, len(others))))
                    best = max(lst, key=lambda p: f(scores[seg["id"]][p]))
                    ok += best == seg["reference"]
                per.append((ok, splits))
            closed[size if size < len(all_sents) else "all"] = C.bootstrap(per)
        # open set: list = random half of test sentences + all dev sentences
        pos_m, neg_m, correct_in = [], [], []
        for s in range(splits):
            rng = random.Random(1000 + s)
            saved = set(rng.sample(test_sents, len(test_sents) // 2))
            lst = sorted(saved) + [x for x in all_sents if x not in set(test_sents)]
            for seg in segs:
                vals = sorted(((f(scores[seg["id"]][p]), p) for p in lst), reverse=True)
                margin = vals[0][0] - vals[1][0]
                if seg["reference"] in saved:
                    pos_m.append(margin)
                    correct_in.append((vals[0][1] == seg["reference"], margin))
                else:
                    neg_m.append(margin)
        results[vname] = {"closed": closed, "open_auroc_margin": auroc(pos_m, neg_m),
                          "open_top1_correct_in_list": sum(c for c, _ in correct_in) / max(len(correct_in), 1),
                          "pos_margins": pos_m, "neg_margins": neg_m, "correct_in": correct_in}
    cost = {k: float(np.median(v)) for k, v in t_cost.items()}
    return results, cost, scores


def model_near_miss(segs, encs, scorer, vocab, n_variants=6, seed=0):
    """Closed set {reference + one-word variants} and open set {variants only}: can the VSR score tell the
    intended sentence from a one-word neighbour (constructed variants of real references)?"""
    rng = random.Random(seed)
    out = {}
    for vname, f in VARIANTS.items():
        out[vname] = {"closed": [], "margin_in": [], "margin_out": [], "by_kind": {}}
    for seg in segs:
        w = seg["reference"].split()
        variants = []
        for i in range(n_variants):
            kind = ("substitute", "delete", "insert_not")[i % 3]
            k = rng.randrange(len(w))
            v = (w[:k] + [rng.choice(vocab)] + w[k + 1:] if kind == "substitute" else
                 w[:k] + w[k + 1:] if kind == "delete" else w[:k] + ["NOT"] + w[k:])
            if " ".join(v) != seg["reference"]:
                variants.append((kind, " ".join(v)))
        texts = [seg["reference"]] + [v for _, v in variants]
        sc = scorer.score(encs[seg["id"]], texts)
        for vname, f in VARIANTS.items():
            vals = [f(x) for x in sc]
            o = out[vname]
            o["closed"].append((float(vals[0] > max(vals[1:])), 1))
            for (kind, _), val in zip(variants, vals[1:]):
                o["by_kind"].setdefault(kind, []).append((float(vals[0] > val), 1))
            srt = sorted(vals, reverse=True)
            o["margin_in"].append(srt[0] - srt[1])
            vo = sorted(vals[1:], reverse=True)
            o["margin_out"].append(vo[0] - vo[1] if len(vo) > 1 else 0.0)
    res = {}
    for vname, o in out.items():
        res[vname] = {"closed_ref_beats_all_variants": C.bootstrap(o["closed"]),
                      "pairwise_by_kind": {k: C.bootstrap(v) for k, v in o["by_kind"].items()},
                      "open_auroc_margin": auroc(o["margin_in"], o["margin_out"]), "n": len(o["closed"])}
    return res


def operating_points(res, thresholds):
    """Accept the best phrase iff margin >= t. In-list: accepted-and-correct; not-in-list: accepted (false)."""
    out = []
    for t in thresholds:
        acc_ok = sum(1 for c, m in res["correct_in"] if m >= t and c)
        acc_wrong_in = sum(1 for c, m in res["correct_in"] if m >= t and not c)
        false_out = sum(1 for m in res["neg_margins"] if m >= t)
        out.append({"t": t, "in_list_correct_accept": acc_ok / max(len(res["correct_in"]), 1),
                    "in_list_wrong_accept": acc_wrong_in / max(len(res["correct_in"]), 1),
                    "not_in_list_false_accept": false_out / max(len(res["neg_margins"]), 1),
                    "n_in": len(res["correct_in"]), "n_out": len(res["neg_margins"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="A")
    ap.add_argument("--test", default="B")
    ap.add_argument("--cond", default="base", help="encoder/decode condition key from decode.py (base, fps12)")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--no-lookalike", action="store_true", help="phrase_match without the viseme look-alike discount")
    ap.add_argument("--text-only", action="store_true", help="skip the model-level part (b)")
    ap.add_argument("--plain", action="store_true", help="use the plain matcher even if phrase_match imports")
    args = ap.parse_args()
    S = args.scratch
    mani = {k: C.load_manifest(os.path.join(S, "manifest", f"{k}.jsonl")) for k in (args.dev, args.test)}
    dec = {}
    for k in (args.dev, args.test):
        p = os.path.join(S, "results", "decode", "stock", f"{k}__{args.cond}__beam{args.beam}.jsonl")
        dec[k] = [json.loads(l) for l in open(p)]
    use_product = False
    if not args.plain:
        try:
            import quietbridge.phrase_match  # noqa: F401
            use_product = True
        except Exception as e:  # pragma: no cover
            print("phrase_match does not import:", e)
    report = {"git": C.git_state(), "cond": args.cond, "beam": args.beam, "matcher":
              "quietbridge.phrase_match.suggest (MAX_DISTANCE_RATIO swept, other defaults)" if use_product
              else "plain normalised word edit distance"}
    report["no_lookalike"] = args.no_lookalike
    if use_product:
        report["phrase_match_commit_blob"] = C.git_state()["head"]
        report["phrase_match_sha256"] = C.sha256(os.path.join(C.ROOT, "quietbridge", "phrase_match.py"))
    ratios = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    report["text"] = {}
    for role, (t, d) in {"dev": (args.dev, args.test), "test": (args.test, args.dev)}.items():
        phrases_other = sorted({r["reference"] for r in mani[d]})
        rescue_rows = []
        per = text_level(dec[t], phrases_other, ratios, args.splits, use_product, args.no_lookalike, rescue_rows)
        report["text"][role] = {str(r): summarise_text(v) for r, v in per.items()}
        if rescue_rows:
            import quietbridge.phrase_match as pm
            report["text"][role]["shipped_summarize_rescue"] = pm.summarize_rescue(rescue_rows)
        for r in ratios:
            s = report["text"][role][str(r)]
            print(f"[text {role}={t}] ratio {r}: top1 {C.fmt(s['top1_exact'])} rescue {C.fmt(s['rescue_when_top1_wrong'])} "
                  f"in-list ok {C.fmt(s['in_list_top1_or_offer'])} wrong-offer(in) {C.fmt(s['in_list_wrong_offer'])} "
                  f"false-offer(out) {C.fmt(s['false_offer_not_in_list'])}", flush=True)

    if use_product:
        report["near_miss_test"] = near_miss(dec[args.test], sorted({r["reference"] for r in mani[args.dev]}), args.splits)
        for k, v in report["near_miss_test"].items():
            print(f"[near-miss {args.test}] {k}: wrong variant offered {C.fmt(v['wrong_variant_offered'])}, "
                  f"flagged sensitive {v['flagged_sensitive_among_offered']:.2f}")
    tagx = args.cond + ("__nolookalike" if args.no_lookalike else "")
    if args.text_only:
        C.write_json(os.path.join(S, "results", f"phrases_text__{tagx}.json"), report)
        return
    # (b)
    reader = C.make_reader(args.beam)
    scorer = PhraseScorer(reader)
    import torch
    all_sents = sorted({s["reference"] for k in mani for s in mani[k]})
    report["model"] = {}
    for role, t in {"dev": args.dev, "test": args.test}.items():
        encs = {}
        for seg in mani[t]:
            p = os.path.join(S, "enc", f"stock__{seg['id']}__{args.cond}.pt")
            encs[seg["id"]] = torch.load(p)["enc"]
        test_sents = sorted({s["reference"] for s in mani[t]})
        res, cost, _ = model_level(mani[t], encs, all_sents, scorer, [10, 25, 50, len(all_sents)], args.splits,
                                   test_sents)
        report["model"][role] = {"cost_sec_per_segment_by_list_size": cost, "variants": {}}
        for v, r in res.items():
            r_small = {k: r[k] for k in ("closed", "open_auroc_margin", "open_top1_correct_in_list")}
            r_small["operating"] = operating_points(r, [0, 1, 2, 3, 5, 8, 12])
            report["model"][role]["variants"][v] = r_small
            print(f"[model {role}={t}] {v}: closed " + ", ".join(f"{k}:{C.fmt(c)}" for k, c in r["closed"].items())
                  + f" | open AUROC(margin) {r['open_auroc_margin']:.3f} top1-in-list {r['open_top1_correct_in_list']:.3f}",
                  flush=True)
        print(f"[model {role}] cost s/segment by list size: {cost}")
        vocab = sorted({w for k in mani for x in mani[k] for w in x["reference"].split() if len(w) > 3})
        nm = model_near_miss(mani[t], encs, scorer, vocab)
        report["model"][role]["near_miss"] = nm
        for v, r in nm.items():
            print(f"[model near-miss {role}={t}] {v}: ref beats all variants {C.fmt(r['closed_ref_beats_all_variants'])} "
                  + " ".join(f"{k}:{C.fmt(c)}" for k, c in r["pairwise_by_kind"].items())
                  + f" | open AUROC {r['open_auroc_margin']:.3f}", flush=True)
    C.write_json(os.path.join(S, "results", f"phrases__{tagx}.json"), report)


if __name__ == "__main__":
    main()

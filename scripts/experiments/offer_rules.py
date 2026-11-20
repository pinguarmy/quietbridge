"""Second follow-up: re-select the model-gated offer rule with a cost-matched criterion.

In the product a lip-only result always needs confirmation (the top text is preselected; one tap),
so an extra phrase row beside a correct result costs clutter, and an extra tap only when it creates a
CONFLICT (negation/number/safety difference). A missed rescue costs retyping. Selection on --select
(A): maximise open-set rescue subject to (a) unsaved distinct sentences get an offer in <= 5 % of rows
and (b) in every scenario, conflict offers beside an already-correct top-1 in <= 10 % of rows; ties ->
fewer offers beside correct results, then the earlier (simpler) rule in RULES order. (b) is pooled over
the constructed-variant scenarios (see choose()).

Rules (offers always ordered by model score, at most 2, never a phrase already shown):
  top:m          score >= score(top-1 text) - m                      (the 0.2 design)
  best:m         score >= max score over the displayed lip candidates - m
  union-<r>:m:M  rule <r>:m  OR  (shipped text rule AND score >= ref - M): the model only vetoes
  text           the shipped text matcher (before), for reference

Scores come from model_offers.py's cache (run `model_offers.py --score A B` first; it now also scores
the 2nd/3rd lip candidates).

    .venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --select A [--cond fps12 --rules ...]
    .venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --evaluate B --rules RULE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from model_offers import SPLITS, load, variants  # noqa: E402

MARGINS = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
UNION_M = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
UNION_BIG = [0.5, 0.6, 0.7, 0.8, 1.0, 2.0, math.inf]


def all_rules():
    rules = ["text"]
    for ref in ("top", "best"):
        rules += [f"{ref}:{m}" for m in MARGINS]
    for ref in ("top", "best"):
        rules += [f"union-{ref}:{m}:{M}" for m in UNION_M for M in UNION_BIG if M > m]
    return rules


def offers(rule, cands, plist, sc, text_offered, idx=None):
    """One result: the phrases a rule offers (best model score first, at most 2). ``idx``: the product
    prefilter's choice for this list (computed once per list)."""
    import quietbridge.phrase_match as pm
    if rule == "text":
        return list(text_offered)
    parts = rule.split(":")
    kind, m = parts[0], float(parts[1])
    big = float(parts[2]) if len(parts) > 2 else None
    ref_kind = kind.split("-")[-1]
    shown = [c for c in cands if sc.get(c) is not None]
    ref = sc[cands[0]] if ref_kind == "top" else max(sc[c] for c in shown)
    listed = {pm.normalized(c) for c in cands}
    if idx is None:
        idx = pm.phrases_to_score(cands, plist)
    chosen = {}
    for i in idx:
        p = plist[i]
        if sc.get(p) is not None and sc[p] >= ref - m and pm.normalized(p) not in listed:
            chosen[p] = sc[p]
    if big is not None:
        for p in text_offered:
            if sc.get(p) is not None and sc[p] >= ref - big:
                chosen[p] = sc[p]
    return [p for p, _ in sorted(chosen.items(), key=lambda kv: -kv[1])][:2]


def evaluate(sess, other, mani, dec, vocab, scores, rules):
    import quietbridge.phrase_match as pm
    own = sorted({s["reference"] for s in mani[sess]})
    others = sorted({s["reference"] for s in mani[other]})
    res = {r: {} for r in rules}

    def add(r, metric, seg_id, num):
        d = res[r].setdefault(metric, {}).setdefault(seg_id, [0, 0])
        d[0] += num
        d[1] += 1

    for seg in mani[sess]:
        cands = [h["text"] for h in dec[sess][seg["id"]]["hyps"][:3]]
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
        for _, v in var:
            lists.append(("var1_out", [v] + others, False))
        lists.append(("var_in", [ref] + [v for _, v in var] + others, True))
        lists.append(("var_out", [v for _, v in var] + others, False))
        top_sensitive = pm.sensitive_words(cands[0])
        for scen, plist, in_list in lists:
            text_offered = [m.phrase for m in pm.suggest(cands, plist)]
            idx = pm.phrases_to_score(cands, plist)
            for r in rules:
                got = offers(r, cands, plist, sc, text_offered, idx)
                hit = any(C.norm(o) == C.norm(ref) for o in got)
                wrong = [o for o in got if C.norm(o) != C.norm(ref)]
                conflict = any(pm.sensitive_words(o) != top_sensitive for o in got)
                if top1:
                    add(r, f"{scen}__beside_correct__any", seg["id"], bool(got))
                    add(r, f"{scen}__beside_correct__conflict", seg["id"], conflict)
                    add(r, "all__beside_correct__any", seg["id"], bool(got))
                    add(r, "all__beside_correct__conflict", seg["id"], conflict)
                    if scen != "open":
                        add(r, "variants__beside_correct__any", seg["id"], bool(got))
                        add(r, "variants__beside_correct__conflict", seg["id"], conflict)
                else:
                    add(r, f"{scen}__beside_wrong__wrong_offer", seg["id"], bool(wrong))
                    if scen != "open":
                        add(r, "variants__beside_wrong__wrong_offer", seg["id"], bool(wrong))
                if in_list and not top1:
                    add(r, f"{scen}__rescue", seg["id"], hit)
                    add(r, f"{scen}__rescue_first", seg["id"], bool(got) and C.norm(got[0]) == C.norm(ref))
                    if not in_nbest:
                        add(r, f"{scen}__rescue_possible", seg["id"], hit)
                if not in_list and scen == "open":
                    add(r, "open__false_offer", seg["id"], bool(got))
    out = {}
    for r, metrics in res.items():
        out[r] = {k: C.bootstrap([tuple(v) for v in segs.values()]) | {"rows": sum(v[1] for v in segs.values())}
                  for k, segs in metrics.items()}
    return out


def choose(summary, rules):
    best = None
    for order, r in enumerate(rules):
        if r == "text":
            continue
        s = summary[r]
        # (b) pooled over the constructed-variant scenarios (var1_out, var_in, var_out), where a conflict
        # is possible at all. A per-scenario maximum cannot resolve 10 %: var_in/var_out have one row per
        # top-1-correct segment (9 on A), so one segment is 11.1 %. Both are reported.
        pooled = s.get("variants__beside_correct__conflict", {}).get("value")
        conflicts = [pooled] if pooled is not None else []
        false = s.get("open__false_offer", {}).get("value")
        rescue = s.get("open__rescue", {}).get("value")
        if rescue is None or false is None or false > 0.05 or (conflicts and max(conflicts) > 0.10):
            continue
        clutter = s.get("all__beside_correct__any", {}).get("value") or 0.0
        key = (-rescue, clutter, order)
        if best is None or key < best[0]:
            best = (key, r)
    return best[1] if best else None


KEYS = ["open__rescue", "open__rescue_possible", "open__false_offer", "all__beside_correct__any",
        "variants__beside_correct__any", "variants__beside_correct__conflict",
        "all__beside_correct__conflict", "var1_out__beside_correct__conflict", "var_out__beside_correct__conflict",
        "variants__beside_wrong__wrong_offer", "var_in__rescue_first"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cond", default="base")
    ap.add_argument("--select")
    ap.add_argument("--evaluate")
    ap.add_argument("--rules", nargs="*", help="default: every rule (selection) ")
    args = ap.parse_args()
    mani, dec, vocab = load(args.scratch, args.cond)
    rules = args.rules or all_rules()
    for sess, tag in ((args.select, "select"), (args.evaluate, "evaluate")):
        if not sess:
            continue
        other = "B" if sess == "A" else "A"
        scores = json.load(open(os.path.join(args.out_dir, f"scores__{sess}__{args.cond}.json")))
        summary = evaluate(sess, other, mani, dec, vocab, scores, rules)
        report = {"git": C.git_state(), "session": sess, "cond": args.cond, "role": tag, "rules": rules,
                  "selection_rule": __doc__.split("Rules")[0].strip(), "summary": summary}
        if tag == "select":
            report["chosen"] = choose(summary, rules)
            print("chosen:", report["chosen"])
        C.write_json(os.path.join(args.out_dir, f"rules__{tag}__{sess}__{args.cond}.json"), report)
        for r in rules:
            s = summary[r]
            print(f"[{sess} {args.cond}] {r}: " + " | ".join(
                f"{k} {C.fmt(s[k])} ({s[k]['rows']})" for k in KEYS if k in s), flush=True)


if __name__ == "__main__":
    main()

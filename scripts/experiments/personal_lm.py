"""Experiment 5: personal language model from the DEV session's text (committed train_lm recipe).

Trains with the COMMITTED `lipflow/train_lm.py` train() (copied from `git show HEAD:`), on the DEV
session's fit sentences, validating on the same held-out tail used by adapt_vsr.py. The recipe's
built-in "general text" forgetting check (GENERAL_SAMPLE) contains sentences transcribed from BOTH
sample clips; the lines that come from the TEST session are removed here so test text cannot influence
whether the candidate is saved. Then decode the test session with the same visual model and beam:

    .venv/bin/python scripts/experiments/personal_lm.py --session A --val-last 6
    .venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --beam 4 \
        --lm $SCRATCH/models/lm_A.pth --tag plmA --enc-tag stock
    # control: same extra scorer at weight 0.2, but the unchanged stock LM
    .venv/bin/python scripts/experiments/personal_lm.py --control-copy
    .venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --beam 4 \
        --lm $SCRATCH/models/lm_control.pth --tag lmcontrol --enc-tag stock
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from adapt_vsr import committed_module  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="A")
    ap.add_argument("--test", default="B")
    ap.add_argument("--val-last", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    ap.add_argument("--control-copy", action="store_true",
                    help="only write the UNCHANGED stock LM as a 'personal' checkpoint (control: extra LM weight, "
                         "no personal data) to $SCRATCH/models/lm_control.pth")
    args = ap.parse_args()
    S = args.scratch
    import torch
    if args.control_copy:
        r = C.make_reader(1)
        path = os.path.join(S, "models", "lm_control.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(r.beam.full_scorers["lm"].state_dict(), path)
        print(path)
        return
    torch.set_num_threads(int(os.environ.get("QB_EXP_THREADS", "2")))
    tlm = committed_module("train_lm", S, {"from .personal": "from lipflow.personal", "from .vsr": "from lipflow.vsr",
                                           "from .paths": "from lipflow.paths"})
    segs = C.load_manifest(os.path.join(S, "manifest", f"{args.session}.jsonl"))
    test = C.load_manifest(os.path.join(S, "manifest", f"{args.test}.jsonl"))
    test_words = {w for s in test for w in s["reference"].split()}
    original = list(tlm.GENERAL_SAMPLE)
    # drop general-sample lines that share a 4-word run with any test-session sentence
    def grams(t, n=4):
        w = t.split()
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    test_grams = set().union(*(grams(s["reference"]) for s in test))
    tlm.GENERAL_SAMPLE = [g for g in original if not grams(g) & test_grams]
    fit, val = segs[:-args.val_last], segs[-args.val_last:]
    out_path = os.path.join(S, "models", f"lm_{args.session}.pth")
    if os.path.exists(out_path):
        os.remove(out_path)  # scratch rerun; the recipe refuses to overwrite
    res = tlm.train(epochs=args.epochs, phrases=[s["reference"] for s in fit],
                    validation_phrases=[s["reference"] for s in val], output_path=out_path, device="cpu")
    fit_words = {w for s in fit for w in s["reference"].split()}
    tw = [w for s in test for w in s["reference"].split()]
    res.update({"git": C.git_state(), "recipe": "git HEAD lipflow/train_lm.py train()", "output": out_path,
                "general_sample_kept": len(tlm.GENERAL_SAMPLE), "general_sample_removed": len(original) - len(tlm.GENERAL_SAMPLE),
                "test_token_coverage_by_fit_vocab": sum(w in fit_words for w in tw) / len(tw),
                "test_types_covered": len(test_words & fit_words) / len(test_words)})
    if os.path.exists(out_path):
        res["bytes"] = os.path.getsize(out_path)
    C.write_json(os.path.join(S, "results", f"personal_lm_{args.session}.json"), res)
    print(res)


if __name__ == "__main__":
    main()

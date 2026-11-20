"""Experiment 4: speaker adaptation with session separation (the product's personal lip model recipe).

Uses the COMMITTED `lipflow/train_vsr.py` (`git show HEAD:...`, copied to scratch with its imports
rewritten) because the working-tree file is being edited by another agent. Fits on the first part of
the DEV session, selects the epoch on a held-out contiguous tail of the DEV session (beam-4 WER, ties broken by teacher-forced val loss; epoch 0
= stock is a candidate), saves the selected delta in scratch. The test session is NOT touched here:
decode it once afterwards with decode.py --vsr-delta <delta> --tag adapt<DEV>.

    .venv/bin/python scripts/experiments/adapt_vsr.py --session A --val-last 6 --epochs 6
"""
from __future__ import annotations

import argparse
import copy
import importlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402


def committed_module(name: str, scratch: str, rewrites: dict[str, str], commit: str = "HEAD"):
    src = subprocess.run(["git", "show", f"{commit}:lipflow/{name}.py"], cwd=C.ROOT, check=True,
                         capture_output=True, text=True).stdout
    rewrites = {**rewrites, "os.path.dirname(__file__)": repr(os.path.join(C.ROOT, "lipflow"))}
    for a, b in rewrites.items():
        src = src.replace(a, b)
    d = os.path.join(scratch, "headcopy")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}_head.py"), "w") as f:
        f.write(src)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(f"{name}_head")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="A")
    ap.add_argument("--val-last", type=int, default=6, help="hold out the last N segments of the session")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scope", default=None, help="default: committed train_vsr.DEFAULT_SCOPE")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--scratch", default=C.DEFAULT_SCRATCH)
    args = ap.parse_args()
    S = args.scratch
    tlm = committed_module("train_lm", S, {"from .personal": "from lipflow.personal", "from .vsr": "from lipflow.vsr",
                                           "from .paths": "from lipflow.paths"})
    tv = committed_module("train_vsr", S, {"from .vsr": "from lipflow.vsr", "from .paths": "from lipflow.paths",
                                           "from .train_lm import Tok": "from train_lm_head import Tok"})
    scope = args.scope or tv.DEFAULT_SCOPE
    segs = C.load_manifest(os.path.join(S, "manifest", f"{args.session}.jsonl"))
    fit, val = segs[:-args.val_last], segs[-args.val_last:]
    clip = C.ClipCache(os.path.join(S, "cache", args.session))
    clips = [{"rois": C.rois_for_segment(clip, s)[0], "text": s["reference"]} for s in fit]
    val_rois = [(s, C.rois_for_segment(clip, s)[0]) for s in val]
    reader = C.make_reader(args.beam)

    def validate():
        import torch
        reader.model.eval()
        e, n = 0, 0
        rows = []
        with torch.inference_mode():
            for s, rois in val_rois:
                hyps = reader.beam_search_details(reader.encode(rois), nbest=1)
                text = hyps[0]["text"] if hyps else ""
                ee, nn = C.errors(s["reference"], text)
                rows.append((ee, nn))
                e, n = e + ee, n + nn
        return C.bootstrap(rows)

    def val_loss():
        """Teacher-forced loss of the recipe (0.1 CTC + 0.9 attention), no augmentation: finer than WER."""
        import torch
        from lipflow.vsr import LipReader
        reader.model.eval()
        with torch.no_grad():
            ls = [float(tv.clip_loss(reader, LipReader.to_tensor(r), tv._targets(reader, s["reference"])))
                  for s, r in val_rois]
        return sum(ls) / len(ls)

    curve = [{"epoch": 0, "val_wer": validate(), "val_loss": val_loss(), "seconds": 0.0}]
    print(f"epoch 0 (stock): val WER {C.fmt(curve[0]['val_wer'])} val loss {curve[0]['val_loss']:.3f}", flush=True)
    prefixes = ["encoder.frontend."] + (["encoder.embed."] if scope in ("frontend+embed", "frontend+encoder1") else []) \
        + (["encoder.encoders.0."] if scope == "frontend+encoder1" else [])
    snaps = {}
    t0 = time.time()

    def on_epoch(ep, total):
        sec = time.time() - t0
        v, vl = validate(), val_loss()
        curve.append({"epoch": ep, "val_wer": v, "val_loss": vl, "seconds": round(sec, 1), "steps": ep * len(clips)})
        import torch
        snaps[ep] = os.path.join(S, "models", f"vsr_delta_{args.session}_epoch{ep}.pth")
        os.makedirs(os.path.dirname(snaps[ep]), exist_ok=True)
        torch.save({k: t.detach().cpu().clone() for k, t in reader.model.state_dict().items()
                    if k.startswith(tuple(prefixes))}, snaps[ep])
        print(f"epoch {ep}: val WER {C.fmt(v)} val loss {vl:.3f}  train time {sec:.0f}s ({sec / (ep * len(clips)):.2f} s/step)", flush=True)

    tv.finetune(clips, epochs=args.epochs, lr=args.lr, scope=scope, reader=reader, on_epoch=on_epoch)
    best = min(curve, key=lambda c: (round(c["val_wer"]["value"], 6), c["val_loss"]))  # WER, ties -> val loss
    import torch
    out = {"git": C.git_state(), "recipe": "git HEAD lipflow/train_vsr.py finetune()", "scope": scope, "lr": args.lr,
           "epochs": args.epochs, "fit_segments": [s["id"] for s in fit], "val_segments": [s["id"] for s in val],
           "fit_seconds_video": sum(s["end"] - s["start"] for s in fit), "curve": curve, "selected_epoch": best["epoch"],
           "train_seconds": round(time.time() - t0, 1)}
    if best["epoch"] > 0:
        out["delta_path"], out["delta_bytes"] = snaps[best["epoch"]], os.path.getsize(snaps[best["epoch"]])
    out["last_path"] = snaps[args.epochs]  # diagnostic only: not selected unless it is the best
    C.write_json(os.path.join(S, "results", f"adapt_{args.session}.json"), out)
    print({k: v for k, v in out.items() if k != "curve"})


if __name__ == "__main__":
    main()

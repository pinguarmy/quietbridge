"""Cut a clip into sentence-like segments (about 3-10 s) with reference text from two ASR runs.

The primary ASR (larger model) gives words and timestamps; the secondary ASR (smaller model) is a
cross-check. A segment is kept only if (1) both ASR texts agree after normalisation, or a hand fix
in --fixes says what the reference is, and (2) the face is tracked in >= 90% of its frames.
Every dropped segment is written with its reason.

    .venv/bin/python scripts/experiments/build_manifest.py --session A \
        --primary $SCRATCH/asr/2016-03-12.medium.en.json --secondary $SCRATCH/asr/2016-03-12.small.en.json \
        --cache $SCRATCH/cache/A --fixes scripts/experiments/reference_fixes.json --out $SCRATCH/manifest/A.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import edit_distance, norm  # noqa: E402

PAD = 0.2  # seconds of video kept before the first / after the last word (lips lead the audio)


def words_of(asr: dict) -> list[dict]:
    return [w for s in asr["segments"] for w in s["words"] if w["word"].strip()]


def chunk(words: list[dict], min_s=3.0, max_s=10.0, pause=0.45) -> list[list[dict]]:
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 9.0
        end_sent = w["word"].rstrip('"').endswith((".", "?", "!"))
        dur = w["end"] - cur[0]["start"]
        if (end_sent or gap > pause) and dur >= min_s or nxt is None:
            groups.append(cur)
            cur = []
        elif dur > max_s - 1.0 and (w["word"].endswith((",", ";", ":")) or gap > 0.25):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for g in groups:  # still too long: split at the largest pause that leaves >= 2.5 s on each side
        stack = [g]
        while stack:
            h = stack.pop(0)
            if h[-1]["end"] - h[0]["start"] <= max_s or len(h) < 4:
                out.append(h)
                continue
            ks = [k for k in range(len(h) - 1)
                  if h[k]["end"] - h[0]["start"] >= 2.5 and h[-1]["end"] - h[k + 1]["start"] >= 2.5]
            k = max(ks, key=lambda k: (h[k + 1]["start"] - h[k]["end"], h[k]["word"].endswith((",", ";")))) \
                if ks else len(h) // 2
            stack[:0] = [h[:k + 1], h[k + 1:]]
    merged = True
    while merged:  # too short: merge into the shorter neighbour if the result stays <= 11 s
        merged = False
        for i, h in enumerate(out):
            if h[-1]["end"] - h[0]["start"] >= min_s:
                continue
            opts = [j for j in (i - 1, i + 1) if 0 <= j < len(out)
                    and max(out[j][-1]["end"], h[-1]["end"]) - min(out[j][0]["start"], h[0]["start"]) <= 11.0]
            if opts:
                j = min(opts, key=lambda j: out[j][-1]["end"] - out[j][0]["start"])
                a, b = sorted((i, j))
                out[a:b + 1] = [out[a] + out[b]]
                merged = True
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--cache", required=True, help="track_clips.py output for this clip")
    ap.add_argument("--fixes", help="JSON {segment id: corrected reference text | null to drop}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prim, sec = json.load(open(args.primary)), json.load(open(args.secondary))
    fixes = {k: v for k, v in json.load(open(args.fixes)).items() if not k.startswith("_")} if args.fixes and os.path.exists(args.fixes) else {}
    anchors = np.load(os.path.join(args.cache, "anchors.npy"))
    fps = json.load(open(os.path.join(args.cache, "meta.json")))["fps"]
    pw, sw = words_of(prim), words_of(sec)
    groups = chunk(pw)
    kept, dropped = [], []
    for n, g in enumerate(groups):
        sid = f"{args.session}{n:03d}"
        prev_end = groups[n - 1][-1]["end"] if n else 0.0
        next_start = groups[n + 1][0]["start"] if n + 1 < len(groups) else g[-1]["end"] + PAD
        start = max(g[0]["start"] - PAD, (prev_end + g[0]["start"]) / 2)
        end = min(g[-1]["end"] + PAD, (g[-1]["end"] + next_start) / 2)
        text_p = " ".join(w["word"] for w in g)
        text_s = " ".join(w["word"] for w in sw if g[0]["start"] - 0.05 <= (w["start"] + w["end"]) / 2 <= g[-1]["end"] + 0.05)
        d = edit_distance(norm(text_p), norm(text_s))
        a, b = int(np.ceil(start * fps)), int(np.floor(end * fps))
        face = float(np.isfinite(anchors[a:b + 1, 0, 0]).mean()) if b >= a else 0.0
        rec = {"id": sid, "session": args.session, "clip": prim["clip"], "start": round(start, 3),
               "end": round(end, 3), "asr_primary": text_p, "asr_secondary": text_s, "asr_disagree_words": d,
               "face_fraction": round(face, 3)}
        reason = None
        if sid in fixes:
            if fixes[sid] is None:
                reason = "hand review: reference doubtful"
            else:
                rec["reference"], rec["reference_source"] = fixes[sid], "hand-fixed"
        elif d == 0 and not any(ch.isdigit() for ch in text_p):
            rec["reference"], rec["reference_source"] = " ".join(norm(text_p)), "asr-agree"
        else:
            reason = f"ASR disagreement ({d} words) or digits; needs hand review"
        dur = end - start
        if reason is None and face < 0.9:
            reason = f"face tracked in only {face:.0%} of frames"
        if reason is None and not 2.0 <= dur <= 12.0:
            reason = f"duration {dur:.1f}s outside 2-12 s"
        if reason:
            rec["drop_reason"] = reason
            dropped.append(rec)
            continue
        rec["reference"] = " ".join(norm(rec["reference"]))
        rec["words"] = len(rec["reference"].split())
        kept.append(rec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", ".dropped.jsonl"), "w") as f:
        for r in dropped:
            f.write(json.dumps(r) + "\n")
    secs = sum(r["end"] - r["start"] for r in kept)
    print(f"session {args.session}: {len(groups)} segments, kept {len(kept)} ({secs:.0f}s, "
          f"{sum(r['words'] for r in kept)} words), dropped {len(dropped)}")
    for r in dropped:
        print(f"  DROP {r['id']} {r['start']:.1f}-{r['end']:.1f}: {r['drop_reason']}\n     P: {r['asr_primary']}\n     S: {r['asr_secondary']}")


if __name__ == "__main__":
    main()

"""Face-track a whole clip once and cache grey frames + per-frame anchors (scratch only).

Every later experiment cuts segments from this cache and runs the product's own preprocessing
(`LipReader.resample` -> `lipflow.face.mouth_rois`), so face tracking is never repeated.

    .venv/bin/python scripts/experiments/track_clips.py samples/2016-03-12.mov --out $SCRATCH/cache/A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--out", required=True, help="cache directory (grays.npy, anchors.npy, meta.json)")
    args = ap.parse_args()
    from lipflow.face import FaceTracker

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    grays = np.lib.format.open_memmap(os.path.join(args.out, "grays.npy"), "w+", np.uint8, (n, h, w))
    anchors = np.full((n, 4, 2), np.nan, np.float32)
    mouth_open = np.full(n, np.nan, np.float32)
    tracker = FaceTracker()
    t0 = time.time()
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        obs = tracker.detect(frame, int(i / fps * 1000))
        grays[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if obs is not None:
            anchors[i] = obs.anchors
            mouth_open[i] = obs.mouth_open
        i += 1
    tracker.close()
    grays.flush()
    np.save(os.path.join(args.out, "anchors.npy"), anchors[:i])
    np.save(os.path.join(args.out, "mouth_open.npy"), mouth_open[:i])
    meta = {"clip": args.clip, "fps": fps, "frames": i, "declared_frames": n, "size": [w, h],
            "face_frames": int(np.isfinite(anchors[:i, 0, 0]).sum()), "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(meta)


if __name__ == "__main__":
    main()

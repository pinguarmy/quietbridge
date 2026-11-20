"""Word-timestamped ASR of a clip's audio track, used ONLY to label the video.

Runs in a separate scratch virtualenv (faster-whisper is not a project dependency):

    uv venv $SCRATCH/asr-venv && VIRTUAL_ENV=$SCRATCH/asr-venv uv pip install faster-whisper
    $SCRATCH/asr-venv/bin/python scripts/experiments/asr_transcribe.py samples/2016-03-12.mov \
        --model medium.en --output $SCRATCH/asr/2016-03-12.medium.en.json

Weights come from Hugging Face (Systran/faster-whisper-*); behind this box's proxy set
HF_HUB_DISABLE_XET=1 (the Xet transfer stalled at 0 bytes, plain LFS works). Output: JSON with segments and words
(start, end, text, probability).
"""
from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--output", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    from faster_whisper import WhisperModel

    t0 = time.time()
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    # Decode audio with ffmpeg (16 kHz mono float): the PyAV that installs here rejects an
    # argument faster-whisper passes to av.open().
    import subprocess
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", args.clip, "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    segments, info = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5,
                                      vad_filter=False, condition_on_previous_text=False)
    out = {"clip": args.clip, "model": args.model, "segments": []}
    for s in segments:
        out["segments"].append({"start": s.start, "end": s.end, "text": s.text.strip(),
                                "words": [{"start": w.start, "end": w.end, "word": w.word.strip(),
                                           "p": w.probability} for w in (s.words or [])]})
    out["seconds"] = time.time() - t0
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{args.clip}: {len(out['segments'])} segments in {out['seconds']:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()

"""Whisper mode: read lips *and* a faint whisper with the Auto-AVSR audio-visual model.

Same vocabulary and language model as the lip reader; the encoder fuses a visual stream (the mouth
crops) with an audio stream (16 kHz waveform, 640 samples per 25 fps video frame). Audio is
layer-normalised per clip, so a whisper's low volume doesn't matter, only its shape.
Weights: nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base (downloaded by the app the first time
whisper mode is turned on; see download()).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .vsr import MODELS, LipReader

AV_DIR = os.path.join(MODELS, "av")
SAMPLES_PER_FRAME = 640  # 16 kHz / 25 fps


URL = "https://huggingface.co/nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base/resolve/main/"


def download(progress=None, timeout: float = 60.0):
    """Fetch config + weights (1.8 GB) into models/av. An interrupted download resumes from its .part
    file on the next call; files appear under their final names only once complete."""
    import urllib.request
    os.makedirs(AV_DIR, exist_ok=True)
    headers = {"User-Agent": "lipflow/0.1"}
    config = os.path.join(AV_DIR, "config.json")
    if not os.path.exists(config):
        req = urllib.request.Request(URL + "config.json", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(config + ".part", "wb") as f:
            f.write(r.read())
        os.replace(config + ".part", config)
    dest = os.path.join(AV_DIR, "model.safetensors")
    part = dest + ".part"
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(URL + "model.safetensors",
                                 headers=headers | ({"Range": f"bytes={offset}-"} if offset else {}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resume = offset and r.status == 206 and \
            r.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        done = offset if resume else 0
        total = done + int(r.headers.get("Content-Length") or 0)
        next_report = done
        with open(part, "ab" if resume else "wb") as f:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress and total and done >= next_report:
                    progress(100 * done / total)
                    next_report = done + (50 << 20)
    if total and done != total:
        raise OSError(f"model.safetensors download ended early ({done} of {total} bytes); retry to resume")
    os.replace(part, dest)


def available() -> bool:
    return os.path.exists(os.path.join(AV_DIR, "model.safetensors"))


class AVReader(LipReader):
    """A LipReader whose encoder also listens. Reuses LipReader's beam search, LMs and helpers;
    the AV model's own decoder and CTC head replace the lip-only ones."""

    def __init__(self, beam_size: int = 4, **kw):
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # a dependency of mlx-lm, so only macOS installs have it today
            raise RuntimeError("Whisper mode needs the safetensors package, which this install lacks "
                               "(add it with `uv add safetensors`)") from e
        super().__init__(beam_size=beam_size, **kw)
        from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E as AVE2E
        from espnet.nets.scorers.ctc import CTCPrefixScorer
        with open(os.path.join(AV_DIR, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        args = argparse.Namespace(**cfg, report_cer=False, report_wer=False, char_list=None,
                                  sym_space=" ", sym_blank="<blank>")
        av = AVE2E(cfg["odim"], args)
        sd = {k[len("avsr."):]: v for k, v in load_file(os.path.join(AV_DIR, "model.safetensors")).items()}
        av.load_state_dict(sd)
        av.eval()
        self.av = av
        for m in (av.encoder, av.aux_encoder, av.fusion):
            m.to(self.enc_device)
        av.decoder.to(self.device)
        av.ctc.to(self.device)
        ctc = CTCPrefixScorer(av.ctc, av.eos)
        for d in (self.beam.full_scorers, self.beam.scorers):
            d["decoder"] = av.decoder
        for d in (self.beam.part_scorers, self.beam.scorers):
            d["ctc"] = ctc
        self.model.ctc = av.ctc
        # the lip-only encoder isn't used here (the app keeps a separate lip reader for live words)
        del self.model.encoder
        import gc
        gc.collect()

    @staticmethod
    def audio_tensor(wave: np.ndarray, n_frames: int) -> torch.Tensor:
        """16 kHz mono float → (T_samples, 1), trimmed/padded to the video length, layer-normed."""
        want = n_frames * SAMPLES_PER_FRAME
        w = np.asarray(wave, dtype=np.float32)[:want]
        if len(w) < want:
            w = np.pad(w, (0, want - len(w)))
        x = torch.from_numpy(w)
        x = torch.nn.functional.layer_norm(x, x.shape, eps=1e-5)  # eps: pure silence would be 0/0
        return x.unsqueeze(-1)

    @torch.inference_mode()
    def encode_av(self, rois: np.ndarray, wave: np.ndarray) -> torch.Tensor:
        v = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        a = self.audio_tensor(wave, rois.shape[0]).unsqueeze(0).to(self.enc_device)
        feat, _ = self.av.encoder(v, None)
        aux, _ = self.av.aux_encoder(a, None)
        n = min(feat.shape[1], aux.shape[1])
        fused = self.av.fusion(torch.cat((feat[:, :n], aux[:, :n]), dim=-1))
        return fused.squeeze(0).to(self.device)

    def warmup_av(self):
        self.read_av(np.zeros((25, 96, 96), np.uint8), np.random.default_rng(0).normal(0, 0.01, 16000))

    def read_av(self, rois, wave) -> tuple[str, float]:
        t0 = time.time()
        return self.beam_search(self.encode_av(rois, wave)), time.time() - t0


def load_audio(path: str, start: float, end: float) -> np.ndarray:
    """16 kHz mono float32 audio from a video file (for tests and evaluation)."""
    import av as pyav
    rs = pyav.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks = []
    with pyav.open(path) as c:
        for frame in c.decode(audio=0):
            t = float(frame.pts * frame.time_base)
            if t > end + 0.5:
                break
            for f in rs.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
    wave = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    s = int(start * 16000)
    return wave[s:s + int((end - start) * 16000)]

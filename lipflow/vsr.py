"""Visual speech recognition: mouth crops -> text.

Wraps the Auto-AVSR LRS3 visual-only model (Ma et al., 2023; Apache 2.0) and its
subword RNN language model, decoded with ESPnet's batch beam search.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load  # noqa: E402
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
from espnet.nets.lm_interface import dynamic_import_lm  # noqa: E402
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: E402
from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402

from .paths import MODELS  # noqa: E402  (re-exported: train_lm, train_vsr and av import it from here)

MODEL_FPS = 25
# Legacy: the personal LM used to be ADDED at this weight on top of the stock LM weight (0.3 -> 0.5
# total). Measured on the real clips (docs/experiments/real-clip-experiments.md, Experiment 5): the
# extra weight itself cost +3.3 WER points, and an unchanged copy of the stock LM did the same. Kept
# only for `personal_lm_mode="additive"` (experiments) and as the meaning of an explicit
# `personal_weight` in that mode.
PERSONAL_LM_WEIGHT = 0.2
# Default: a personal LM SHARES the stock LM weight instead of adding to it. The total LM weight stays
# `lm_weight`; the personal model gets this fraction of it (log-linear interpolation of the two LMs).
# 0.5 is an a-priori split, not tuned; see the follow-up section of the experiments document.
PERSONAL_LM_SHARE = 0.5
PERSONAL_LM_MODES = ("shared", "additive")
SPM_MODEL = os.path.join(MODELS, "lm", "unigram5000.model")
# Teacher-forced scoring bounds (score_texts): longer texts are not scored.
MAX_SCORED_TOKENS = 120
_MEAN, _STD = 0.421, 0.165

_DIGIT_WORDS = ("ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN "
                "FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN TWENTY").split()


def model_words(text: str) -> "list[str] | None":
    """Text -> the LRS3-style words the lip model's vocabulary writes (upper case, letters, apostrophes).

    Numbers 0-20 are spelt out (the lip model writes words). Any other digit makes the text
    unscorable (None): guessing how a person says "112" would score a different sentence.
    """
    import re
    import unicodedata
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFKC", text)
    for mark in ("’", "‘", "ʼ", "`", "´"):
        text = text.replace(mark, "'")
    out = []
    for token in re.findall(r"[\w']+", text.replace("-", " ").replace("_", " ")):
        if not token.isascii():
            return None  # letters outside the model's A-Z vocabulary
        if any(ch.isdigit() for ch in token):
            if token.isdigit() and int(token) <= 20:
                out.append(_DIGIT_WORDS[int(token)])
                continue
            return None
        word = token.strip("'").upper()
        if word:
            out.append(word)
    return out


def lm_weights(stock_weight: float, has_personal: bool, mode: str = "shared",
               personal_weight: "float | None" = None) -> dict:
    """Beam-search weights for the stock LM ("lm") and an optional personal LM ("plm").

    "shared" (default): the personal LM takes ``personal_weight`` (default PERSONAL_LM_SHARE) as a
    FRACTION of ``stock_weight`` and the stock LM keeps the rest, so the total LM weight is
    unchanged. "additive" (legacy, for experiments): the personal LM is added at
    ``personal_weight`` (default PERSONAL_LM_WEIGHT) on top of the stock weight.
    """
    if mode not in PERSONAL_LM_MODES:
        raise ValueError(f"personal_lm_mode must be one of {PERSONAL_LM_MODES}")
    if not has_personal or stock_weight <= 0:
        return {"lm": stock_weight}
    if mode == "additive":
        extra = PERSONAL_LM_WEIGHT if personal_weight is None else float(personal_weight)
        return {"lm": stock_weight, "plm": extra} if extra > 0 else {"lm": stock_weight}
    share = PERSONAL_LM_SHARE if personal_weight is None else min(max(float(personal_weight), 0.0), 1.0)
    if share <= 0:
        return {"lm": stock_weight}
    return {"lm": stock_weight * (1.0 - share), "plm": stock_weight * share}


def pick_encoder_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LipReader:
    def __init__(self, device: str = "auto", beam_size: int = 20, lm_weight: float = 0.3,
                 ctc_weight: float = 0.1, use_lm: bool = True, personal_weight: "float | None" = None,
                 personal: bool = True, personal_vsr_path: str | None = None,
                 personal_lm_path: str | None = None, personal_lm_mode: str = "shared"):
        if personal_lm_mode not in PERSONAL_LM_MODES:
            raise ValueError(f"personal_lm_mode must be one of {PERSONAL_LM_MODES}")
        for label, value in (("lip", personal_vsr_path), ("language", personal_lm_path)):
            if value is not None and not os.path.isfile(os.path.expanduser(value)):
                raise FileNotFoundError(f"Personal {label} model not found: {os.path.expanduser(value)}")
        # The 3D-conv + conformer encoder is 17x faster on the Apple GPU; beam search is
        # thousands of tiny ops and runs faster on CPU, so the two are split.
        self.enc_device = pick_encoder_device(device)
        self.device = torch.device("cuda") if self.enc_device.type == "cuda" else torch.device("cpu")
        with open(os.path.join(MODELS, "vsr", "model.json"), "rb") as f:
            confs = json.load(f)
        args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))
        units = os.path.join(os.path.dirname(__file__), "unigram5000_units.txt")
        with open(units, encoding="utf-8") as f:
            self.token_list = ["<blank>"] + [l.split()[0] for l in f.read().splitlines()] + ["<eos>"]
        odim = len(self.token_list)

        self.model = E2E(odim, args)
        state = torch.load(os.path.join(MODELS, "vsr", "model.pth"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        # Explicit paths select an experiment without copying it over the
        # original app's default personal weights. Missing explicit files fail.
        self.personal_vsr = False
        self.personal_vsr_source = None
        from .paths import PERSONAL_LM, PERSONAL_VSR
        pv = os.path.abspath(os.path.expanduser(personal_vsr_path or PERSONAL_VSR))
        if (personal or personal_vsr_path is not None) and os.path.exists(pv):
            self.model.load_state_dict(torch.load(pv, map_location="cpu", weights_only=True), strict=False)
            self.personal_vsr = True
            self.personal_vsr_source = pv
        self.model.to(self.device).eval()
        self.model.encoder.to(self.enc_device)

        scorers = self.model.scorers()
        lm = None
        if use_lm and lm_weight > 0:
            lm_path = os.path.join(MODELS, "lm", "model.pth")
            lm_args = get_model_conf(lm_path, os.path.join(MODELS, "lm", "model.json"))
            lm_class = dynamic_import_lm(getattr(lm_args, "model_module", "default"), lm_args.backend)
            lm = lm_class(odim, lm_args)
            torch_load(lm_path, lm)
            lm.eval()
        scorers["lm"] = lm
        # How you talk: a copy of the LM fine-tuned on your phrases (lipflow train-lm), scored
        # alongside the general one so the search still knows ordinary English. By default the two
        # SHARE the stock LM weight (total unchanged): adding weight on top made recognition worse
        # on the real clips even when the "personal" LM was an unchanged copy of the stock one.
        plm = None
        pl = os.path.abspath(os.path.expanduser(personal_lm_path or PERSONAL_LM))
        if lm is not None and (personal or personal_lm_path is not None) and os.path.exists(pl):
            plm = lm_class(odim, lm_args)
            plm.load_state_dict(torch.load(pl, map_location="cpu", weights_only=True))
            plm.eval()
        self.personal_lm = plm is not None
        self.personal_lm_source = pl if plm is not None else None
        self.personal_lm_mode = personal_lm_mode
        self.lm_weights = lm_weights(lm_weight if lm else 0.0, plm is not None, personal_lm_mode, personal_weight)
        if "plm" in self.lm_weights:
            scorers["plm"] = plm
        scorers["length_bonus"] = LengthBonus(odim)
        self._spm = None
        self._lm_score_cache: dict = {}
        self.beam = BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=odim,
            weights=dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, length_bonus=0.0, **self.lm_weights),
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=self.token_list,
            pre_beam_score_key="decoder",
        ).to(self.device).eval()

    @staticmethod
    def resample(timestamps: list[float], n_frames: int) -> list[int]:
        """Indices that turn a variable-rate capture into a steady 25 fps sequence."""
        if n_frames == 0:
            return []
        t = np.asarray(timestamps, dtype=np.float64)
        grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / MODEL_FPS)
        return np.clip(np.searchsorted(t, grid), 0, n_frames - 1).tolist()

    @staticmethod
    def to_tensor(rois: np.ndarray) -> torch.Tensor:
        """(T, 96, 96) uint8 -> (C=1, T, 88, 88) normalised; encode() adds the batch dim."""
        x = torch.from_numpy(rois).float() / 255.0
        o = (x.shape[-1] - 88) // 2
        x = x[:, o:o + 88, o:o + 88]
        return ((x - _MEAN) / _STD).unsqueeze(0)

    @torch.inference_mode()
    def encode(self, rois: np.ndarray) -> torch.Tensor:
        # Tried and measured on the benchmark, no gain: averaging with the mirrored clip (36.3% ->
        # 37.4% WER), beam 20, CTC weight 0.2/0.3, LM weight 0.2-0.6. Stock decoding stays.
        x = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        enc, _ = self.model.encoder(x, None)
        return enc.squeeze(0).to(self.device)

    @torch.inference_mode()
    def greedy(self, enc: torch.Tensor) -> str:
        """CTC best path: ~1 ms, used for the live preview while you are still talking."""
        ids = self.model.ctc.argmax(enc.unsqueeze(0))[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.token_list[i])
            prev = i
        return self._clean("".join(out))

    def _forget_memory_projections(self) -> None:
        """Drop the decoder's per-utterance cache of projected encoder memory.

        espnet's MultiHeadedAttention caches the source-attention key/value projections keyed by
        the memory tensor's data pointer and length. A later utterance whose encoder output has the
        same length can be allocated at the freed address (measured: it is, on CPU), and would then
        be decoded against the PREVIOUS utterance's projections. Clearing before every search or
        scoring pass makes the cache valid for exactly one utterance.
        """
        model = getattr(self, "model", None)
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "modules"):
            return
        for module in decoder.modules():
            if getattr(module, "_mem_kv", None) is not None:
                module._mem_kv = None

    @torch.inference_mode()
    def beam_search_details(self, enc: torch.Tensor, nbest: int = 1) -> list[dict]:
        """Return text and the decoder's unnormalised score for each candidate."""
        self._forget_memory_projections()
        hyps = self.beam(enc)
        results = []
        seen = set()
        eos = len(self.token_list) - 1
        for h in hyps[:max(nbest, 1)]:
            data = h.asdict()
            # The search already computed each finished hypothesis' unweighted LM log-probability;
            # remember it so score_texts() need not run the 16-layer LM again for the same tokens.
            yseq = [int(t) for t in data.get("yseq", [])]
            if len(yseq) > 2 and yseq[-1] == eos:
                for name in getattr(self, "lm_weights", {}):
                    if name in data.get("scores", {}):
                        self._remember_lm_score((name, tuple(yseq[1:-1])), float(data["scores"][name]))
            text = self._clean(add_results_to_json([data], self.token_list))
            if text in seen:
                continue
            seen.add(text)
            results.append({"text": text, "score": float(data["score"])})
        return results

    @torch.inference_mode()
    def beam_search(self, enc: torch.Tensor, nbest: int = 1) -> "str | list[str]":
        texts = [item["text"] for item in self.beam_search_details(enc, nbest)]
        if nbest > 1:
            return texts
        # ESPnet returns no hypothesis at all when none ended (minlenratio 0): that reads as nothing.
        return texts[0] if texts else ""

    # ------------------------------------------------------------------ teacher-forced text scoring
    def _token_ids(self, text: str) -> "list[int] | None":
        words = model_words(text)
        if not words:
            return None
        if self._spm is None:
            import sentencepiece as spm
            if not os.path.isfile(SPM_MODEL):
                raise FileNotFoundError(f"SentencePiece tokenizer not found at {SPM_MODEL}")
            self._spm = spm.SentencePieceProcessor(model_file=SPM_MODEL)
            self._token_index = {t: i for i, t in enumerate(self.token_list)}
        unk = self._token_index.get("<unk>", 1)
        ids = [self._token_index.get(p, unk) for p in self._spm.encode(" ".join(words), out_type=str)]
        return ids if 0 < len(ids) <= MAX_SCORED_TOKENS else None

    @torch.inference_mode()
    def _lm_logprob(self, name: str, ids_list: "list[list[int]]") -> list[float]:
        """Sum of log p(token) incl. <eos> under the beam's LM scorer `name`; cached per token sequence
        (an LM score does not depend on the video, and saved phrases repeat across utterances)."""
        lm = self.beam.full_scorers.get(name)
        out: list = [None] * len(ids_list)
        todo = []
        for i, ids in enumerate(ids_list):
            key = (name, tuple(ids))
            if key in self._lm_score_cache:
                out[i] = self._lm_score_cache[key]
            else:
                todo.append(i)
        eos = len(self.token_list) - 1
        for k in range(0, len(todo), 32):
            rows = todo[k:k + 32]
            length = max(len(ids_list[i]) for i in rows) + 1
            x = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
            t = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
            for j, i in enumerate(rows):
                ids = ids_list[i]
                x[j, :len(ids) + 1] = torch.tensor([eos] + ids)
                t[j, :len(ids) + 1] = torch.tensor(ids + [eos])
            # Same computation as the LM's forward() (teacher-forced next-token log-probabilities),
            # kept per row instead of summed over the batch.
            h, _ = lm.encoder(lm.embed(x), lm._target_mask(x))
            logp = lm.decoder(h).log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
            logp = (logp * (x != 0)).sum(1)
            for j, i in enumerate(rows):
                out[i] = float(logp[j])
                self._remember_lm_score((name, tuple(ids_list[i])), out[i])
        return out

    def _remember_lm_score(self, key, value: float) -> None:
        # Bounded (oldest first out): LM scores of saved phrases are reused on every utterance.
        if not hasattr(self, "_lm_score_cache"):
            self._lm_score_cache = {}
        if key not in self._lm_score_cache and len(self._lm_score_cache) >= 2048:
            self._lm_score_cache.pop(next(iter(self._lm_score_cache)))
        self._lm_score_cache[key] = value

    @torch.inference_mode()
    def score_texts(self, enc: torch.Tensor, texts: "list[str]", batch: int = 32) -> "list[dict | None]":
        """Score arbitrary sentences against one utterance's encoder output.

        Each text is forced through the attention decoder and the CTC branch, and through the beam's
        LM scorer(s), and combined with the beam search's OWN weights (decoder 1-ctc, ctc, lm[, plm];
        the length bonus is 0 in this product). ``score`` is that weighted sum of log-probabilities
        over the text's tokens plus <eos>; ``normalized`` divides it by that token count. This is the
        combination validated in docs/experiments/real-clip-experiments.md (Experiment 2b). Scoring
        the decoder's own top text the same way gives a comparable reference. Scores are model
        log-probabilities, NOT calibrated probabilities. None for a text the model cannot write
        (digits above 20, non-ASCII letters, empty, more than MAX_SCORED_TOKENS tokens).
        """
        import torch.nn.functional as F
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

        ids_list = [self._token_ids(t) for t in texts]
        valid = [i for i, ids in enumerate(ids_list) if ids]
        results: "list[dict | None]" = [None] * len(texts)
        if not valid:
            return results
        weights = self.beam.weights
        w_att, w_ctc = float(weights.get("decoder", 0.9)), float(weights.get("ctc", 0.1))
        lm_parts = {name: float(w) for name, w in self.lm_weights.items()
                    if w > 0 and self.beam.full_scorers.get(name) is not None}
        self._forget_memory_projections()
        # Similar lengths share a batch (less padding); results keep the caller's order.
        valid.sort(key=lambda i: len(ids_list[i]))
        position = {i: p for p, i in enumerate(valid)}
        lm_scores = {name: self._lm_logprob(name, [ids_list[i] for i in valid]) for name in lm_parts}
        eos = len(self.token_list) - 1
        enc = enc.to(self.device)
        logp_ctc = self.model.ctc.ctc_lo(enc.unsqueeze(0)).log_softmax(-1)[0]  # (T, V)
        n_frames = logp_ctc.shape[0]
        for k in range(0, len(valid), batch):
            rows = valid[k:k + batch]
            toks = [ids_list[i] for i in rows]
            length = max(len(t) for t in toks) + 1
            ys_in = torch.full((len(rows), length), eos, dtype=torch.long, device=self.device)
            target = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
            keep = torch.zeros((len(rows), length), dtype=torch.bool, device=self.device)
            for j, t in enumerate(toks):
                ys_in[j, 1:len(t) + 1] = torch.tensor(t)
                target[j, :len(t) + 1] = torch.tensor(t + [eos])
                keep[j, :len(t) + 1] = True
            mask = subsequent_mask(length, device=self.device).unsqueeze(0)
            memory = enc.unsqueeze(0).expand(len(rows), -1, -1)
            pred, _ = self.model.decoder(ys_in, mask, memory, None)
            att = (pred.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1) * keep).sum(1)
            # CTC over only the vocabulary this batch uses (blank first): identical log-likelihoods,
            # without copying a (T, batch, 5049) tensor.
            vocab = sorted({0, *(x for t in toks for x in t)})
            remap = {v: n for n, v in enumerate(vocab)}
            sub = logp_ctc[:, vocab]
            flat = torch.tensor([remap[x] for t in toks for x in t], dtype=torch.long)
            ctc = -F.ctc_loss(sub.unsqueeze(1).expand(-1, len(rows), -1), flat,
                              torch.full((len(rows),), n_frames, dtype=torch.long),
                              torch.tensor([len(t) for t in toks], dtype=torch.long),
                              blank=0, reduction="none", zero_infinity=False)
            for j, i in enumerate(rows):
                pos = position[i]
                lm_values = {name: lm_scores[name][pos] for name in lm_parts}
                ntok = len(toks[j]) + 1
                c = float(ctc[j])
                if not np.isfinite(c):
                    c = -1e4  # impossible alignment (text too long for the clip): very unlikely, finite
                total = (w_att * float(att[j]) + w_ctc * c
                         + sum(lm_parts[name] * value for name, value in lm_values.items()))
                results[i] = {"att": float(att[j]), "ctc": c, "lm": lm_values, "tokens": ntok,
                              "score": total, "normalized": total / ntok}
        return results

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.replace("▁", " ").replace("<eos>", "").split())

    def read(self, rois: np.ndarray, fast: bool = False) -> tuple[str, float]:
        """Return (UPPERCASE transcript, seconds spent)."""
        t0 = time.time()
        enc = self.encode(rois)
        text = self.greedy(enc) if fast else self.beam_search(enc)
        return text, time.time() - t0

    def warmup(self):
        self.read(np.zeros((25, 96, 96), dtype=np.uint8))

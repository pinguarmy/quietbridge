"""`lipflow train-lm`: fine-tune the decoder's language model on how you talk.

The beam search already uses a 16-layer subword Transformer LM trained on LRS3/TED text. This
makes a personal copy fine-tuned on your imported phrases (phrases.txt). The copy runs next to the
original as an extra scorer, so the decoder still knows general English but leans towards your
phrasing word by word, not just when picking among finished guesses.

Guard against forgetting: perplexity is measured on your held-out phrases *and* on general text
before and after; the model is only saved if yours improves by at least MIN_RELATIVE_GAIN and general
text gets no more than MAX_GENERAL_LOSS worse. Perplexity is not recognition quality: compare a
saved candidate on an independent test session before promoting it.
"""
from __future__ import annotations

import math
import os
import random
import re
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from .personal import PHRASES
from .vsr import MODELS

from .paths import PERSONAL_LM
SPM = os.path.join(MODELS, "lm", "unigram5000.model")
# General English for the forgetting check. Written for this purpose: it must NOT come from the
# sample clips (samples/2016-03-12.mov, samples/2017-01-07.mov) or any other evaluation material,
# because then the text a model is later tested on would decide whether a candidate is saved. An
# earlier version used sentences transcribed from those very clips.
GENERAL_SAMPLE = [
    "THE TRAIN TO THE CITY CENTRE LEAVES EVERY TWENTY MINUTES ON WEEKDAYS",
    "SHE OPENED THE WINDOW TO LET SOME FRESH AIR INTO THE KITCHEN",
    "MOST OF THE SHOPS ON THIS STREET CLOSE EARLY ON SUNDAY",
    "WE SHOULD PLANT THE TOMATOES BEFORE THE WEATHER GETS TOO WARM",
    "HE FORGOT HIS UMBRELLA AND GOT CAUGHT IN THE RAIN ON THE WAY HOME",
    "THE LIBRARY HAS A QUIET ROOM WHERE STUDENTS CAN READ AND WORK",
    "CAN YOU TELL ME HOW LONG IT TAKES TO WALK TO THE STATION",
    "THE DOCTOR SAID I SHOULD DRINK MORE WATER AND SLEEP A LITTLE LONGER",
    "OUR NEIGHBOURS ARE MOVING TO A SMALLER HOUSE NEAR THE COAST",
    "THE MEETING HAS BEEN MOVED TO THURSDAY AFTERNOON",
    "I USUALLY HAVE TEA WITH MILK AND A PIECE OF TOAST FOR BREAKFAST",
    "THE CHILDREN BUILT A SMALL BRIDGE OUT OF STICKS AND STONES",
    "PRICES WENT UP AGAIN THIS MONTH AND MANY FAMILIES ARE WORRIED",
    "PLEASE REMEMBER TO TURN OFF THE LIGHTS WHEN YOU LEAVE THE OFFICE",
    "THE MUSEUM IS FREE ON THE FIRST SATURDAY OF EVERY MONTH",
    "SCIENTISTS HAVE FOUND A NEW SPECIES OF FROG IN THE MOUNTAINS",
]
# Saving rule. Perplexity is NOT recognition quality: on the real clips a candidate whose held-out
# perplexity improved by 1.5 % was saved and then cost 3.3 WER points when decoded (the extra LM
# weight, not the personal text, caused that: docs/experiments/real-clip-experiments.md, Exp. 5).
# A candidate is saved only if held-out perplexity improves by at least this fraction (a small
# validation set moves perplexity by a few percent by chance) ...
MIN_RELATIVE_GAIN = 0.10
# ... and general-text perplexity does not rise by more than this factor (forgetting check).
MAX_GENERAL_LOSS = 1.05
# A saved candidate still needs a recognition-level comparison on an independent test session
# (`quietbridge.evaluate_language`) before anyone promotes it.


def normalise(line: str) -> list[str]:
    """Your text → LRS3-style sentences: upper case, letters and apostrophes, no digits."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", line):
        if re.search(r"\d", sent):
            continue  # LRS3 spells numbers out; skip rather than guess
        s = re.sub(r"[^A-Za-z' ]+", " ", sent).upper()
        s = " ".join(w.strip("'") for w in s.split() if w.strip("'"))
        if len(s.split()) >= 2:
            out.append(s)
    return out


class Tok:
    def __init__(self, token_list: list[str]):
        if not os.path.isfile(SPM):
            setup = r".\setup.ps1" if os.name == "nt" else "./setup.sh"
            raise FileNotFoundError(
                f"SentencePiece tokenizer not found at {SPM}. Training needs it; download the models "
                f"with {setup} (it fetches models/lm/unigram5000.model), then try again.")
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=SPM)
        self.id = {t: i for i, t in enumerate(token_list)}
        self.unk = self.id["<unk>"]

    def __call__(self, s: str) -> list[int]:
        return [self.id.get(p, self.unk) for p in self.sp.encode(s, out_type=str)]


def _batches(seqs, eos, bs, shuffle=True):
    seqs = list(seqs)
    if shuffle:
        random.shuffle(seqs)
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        n = max(len(s) for s in chunk) + 1
        x = torch.zeros(len(chunk), n, dtype=torch.long)
        t = torch.zeros(len(chunk), n, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, :len(s) + 1] = torch.tensor([eos] + s)
            t[j, :len(s) + 1] = torch.tensor(s + [eos])
        yield x, t


@torch.no_grad()
def perplexity(lm, seqs, eos, device) -> float:
    lm.eval()
    nll, n = 0.0, 0
    for x, t in _batches(seqs, eos, 64, shuffle=False):
        _, s_nll, count = lm(x.to(device), t.to(device))
        nll += float(s_nll)
        n += int(count)
    return math.exp(nll / max(n, 1))


def train(
    epochs: int = 3,
    lr: float = 3e-5,
    bs: int = 32,
    max_general_loss: float = MAX_GENERAL_LOSS,
    min_relative_gain: float = MIN_RELATIVE_GAIN,
    seed: int = 0,
    phrases: Iterable[str] | None = None,
    validation_phrases: Iterable[str] | None = None,
    output_path: str | os.PathLike | None = None,
    device: str | torch.device | None = None,
) -> dict:
    from espnet.asr.asr_utils import get_model_conf, torch_load
    from espnet.nets.lm_interface import dynamic_import_lm

    target_path: Path | None = None
    if output_path is not None:
        target_path = Path(output_path).expanduser().resolve()
        if target_path.exists():
            raise FileExistsError(f"Candidate output path already exists: {target_path}")

    random.seed(seed)
    torch.manual_seed(seed)

    # Validate the inputs before loading any asset, so bad phrases get the error about phrases.
    if phrases is None:
        if not os.path.exists(PHRASES):
            raise FileNotFoundError("No phrases yet. Run `lipflow import-wispr` first.")
        with open(PHRASES, encoding="utf-8") as f:
            tr_sents = sorted({s for line in f for s in normalise(line)})
    else:
        tr_sents = [s for line in phrases for s in normalise(line)]
    val_sents = None
    if validation_phrases is not None:
        val_sents = [s for line in validation_phrases for s in normalise(line)]
        if not tr_sents:
            raise ValueError("No valid training sentences after normalization")
        if not val_sents:
            raise ValueError("No valid validation sentences after normalization")
    elif not tr_sents:
        raise ValueError("No valid sentences found")

    units = os.path.join(os.path.dirname(__file__), "unigram5000_units.txt")
    with open(units, encoding="utf-8") as f:
        token_list = ["<blank>"] + [l.split()[0] for l in f.read().splitlines()] + ["<eos>"]
    eos = len(token_list) - 1
    tok = Tok(token_list)

    if val_sents is not None:
        val = [tok(s)[:58] for s in val_sents]
        tr = [tok(s)[:58] for s in tr_sents]
        all_sents = tr_sents + val_sents
    else:
        sents = tr_sents
        random.shuffle(sents)
        seqs = [tok(s)[:58] for s in sents]
        n_val = max(20, len(seqs) // 10)
        val, tr = seqs[:n_val], seqs[n_val:]
        all_sents = sents

    general = [tok(s) for s in GENERAL_SAMPLE]

    lm_path = os.path.join(MODELS, "lm", "model.pth")
    args = get_model_conf(lm_path, os.path.join(MODELS, "lm", "model.json"))
    lm = dynamic_import_lm(getattr(args, "model_module", "default"), args.backend)(len(token_list), args)
    torch_load(lm_path, lm)
    if device is None or device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)
    lm.to(device)

    before = {"yours": perplexity(lm, val, eos, device), "general": perplexity(lm, general, eos, device)}
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    # Only an epoch that beats the stock model by the minimum relative gain can be saved.
    required = before["yours"] * (1.0 - min_relative_gain)
    best, best_state = before["yours"], None
    best_seen = before["yours"]
    general_blocked = False
    for ep in range(epochs):
        lm.train()
        for x, t in _batches(tr, eos, bs):
            loss, _, _ = lm(x.to(device), t.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
        p_y, p_g = perplexity(lm, val, eos, device), perplexity(lm, general, eos, device)
        print(f"  epoch {ep + 1}: your held-out phrases {p_y:.1f}, general text {p_g:.1f}  ({time.time() - t0:.0f}s)")
        best_seen = min(best_seen, p_y)
        if p_y < best and p_y <= required:
            if p_g <= before["general"] * max_general_loss:
                best, best_state = p_y, {k: v.detach().cpu().clone() for k, v in lm.state_dict().items()}
            else:
                general_blocked = True
    after = {"yours": best, "general": None}
    saved = best_state is not None
    reason = ("saved" if saved else "general_text_regressed" if general_blocked
              else "validation_gain_below_minimum")
    if saved:
        if target_path is not None:
            if target_path.exists():
                raise FileExistsError(f"Candidate output path already exists: {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("xb") as f:
                torch.save(best_state, f)
        else:
            os.makedirs(os.path.dirname(PERSONAL_LM), exist_ok=True)
            torch.save(best_state, PERSONAL_LM)
        lm.load_state_dict(best_state)
        after["general"] = perplexity(lm, general, eos, device)
    return {"sentences": len(all_sents), "train": len(tr), "val": len(val), "before": before, "after": after,
            "saved": saved, "seconds": time.time() - t0, "reason": reason,
            "rule": {"min_relative_gain": min_relative_gain, "max_general_loss": max_general_loss,
                     "best_validation_perplexity_seen": best_seen,
                     "note": "Perplexity is not recognition quality; compare on an independent test session "
                             "(quietbridge.evaluate_language) before promoting."}}

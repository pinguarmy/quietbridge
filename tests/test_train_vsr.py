"""Lip-reading fine-tune recipe contracts with a tiny fake model (no base weights).

These check what is fine-tuned vs frozen, determinism under a seed, frozen
BatchNorm statistics, and that augmentation touches training clips only. They
say nothing about whether the recipe improves lip reading.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lipflow import train_vsr

V = 6  # tiny vocabulary: 0 blank ... 5 eos


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.BatchNorm1d(4))
        self.embed = torch.nn.Linear(4, 4)
        self.encoders = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])

    def forward(self, x, mask):
        # x: (1, 1, T, 88, 88) -> (1, T, 4)
        h = x.mean(dim=(-1, -2)).transpose(1, 2)          # (1, T, 1)
        h = self.frontend[0](h)
        h = self.frontend[1](h.transpose(1, 2)).transpose(1, 2)
        h = self.embed(h)
        for layer in self.encoders:
            h = torch.tanh(layer(h))
        return h, None


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 4)
        self.out = torch.nn.Linear(4, V)

    def forward(self, ys_in, mask, hs, cache):
        return self.out(self.emb(ys_in) + hs.mean(dim=1, keepdim=True)), None


class _Criterion(torch.nn.Module):
    def forward(self, pred, ys_out):
        return torch.nn.functional.cross_entropy(pred.reshape(-1, V), ys_out.reshape(-1), reduction="sum")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.ctc = torch.nn.Module()
        self.ctc.ctc_lo = torch.nn.Linear(4, V)
        self.decoder = _Decoder()
        self.criterion = _Criterion()
        self.eos = V - 1


class FakeReader:
    def __init__(self, seed=123):
        torch.manual_seed(seed)
        self.model = _Model()
        self.device = self.enc_device = "cpu"
        self._tok = lambda text: [1, 2, 3]  # skip the real tokenizer
        self.read_calls = []

    def read(self, rois, fast=False):  # evaluation path
        self.read_calls.append(rois)
        return "hello", 0.0


def clips(n=3, t=12):
    rng = np.random.default_rng(0)
    return [{"rois": rng.integers(0, 255, size=(t, 96, 96), dtype=np.uint8), "text": "hello there"}
            for _ in range(n)]


def state(reader):
    return {k: v.detach().clone() for k, v in reader.model.state_dict().items()}


def test_only_the_chosen_scope_is_trained_and_batchnorm_statistics_stay_frozen():
    reader = FakeReader()
    before = state(reader)
    train_vsr.finetune(clips(), epochs=2, lr=1e-2, scope="frontend", reader=reader, log=lambda *_: None)
    after = state(reader)
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("encoder.frontend.") for k in changed)
    assert torch.equal(before["encoder.frontend.1.running_mean"], after["encoder.frontend.1.running_mean"])
    assert torch.equal(before["encoder.frontend.1.running_var"], after["encoder.frontend.1.running_var"])
    assert not any(p.requires_grad for p in reader.model.parameters())  # frozen again afterwards

    reader = FakeReader()
    before = state(reader)
    train_vsr.finetune(clips(), epochs=1, lr=1e-2, scope="frontend+encoder1", reader=reader, log=lambda *_: None)
    changed = {k for k, v in state(reader).items() if not torch.equal(before[k], v)}
    assert any(k.startswith("encoder.encoders.0.") for k in changed)
    assert not any(k.startswith(("encoder.encoders.1.", "decoder.", "ctc.")) for k in changed)


def test_same_seed_reproduces_the_adaptation_and_another_seed_differs():
    runs = []
    for seed in (5, 5, 6):
        reader = FakeReader()
        train_vsr.finetune(clips(), epochs=2, lr=1e-2, scope="frontend", reader=reader, seed=seed,
                           log=lambda *_: None)
        runs.append(state(reader))
    assert all(torch.equal(runs[0][k], runs[1][k]) for k in runs[0])
    assert not all(torch.equal(runs[0][k], runs[2][k]) for k in runs[0])


def test_augmentation_is_applied_to_training_clips_only():
    seen = []

    def spy(rois):
        seen.append(id(rois))
        return train_vsr._augment(rois)

    reader = FakeReader()
    train_set = clips(3)
    train_vsr.finetune(train_set, epochs=2, scope="frontend", reader=reader, augment=spy, log=lambda *_: None)
    assert sorted(seen) == sorted([id(c["rois"]) for c in train_set] * 2)
    # Evaluation decodes the stored clip unchanged.
    from quietbridge.signs.train_lips import evaluate
    held_out = [{"path": "v.npz", "trial_id": "v", "text": "hello", "rois": clips(1)[0]["rois"]}]
    evaluate(reader, held_out)
    assert reader.read_calls[0] is held_out[0]["rois"] and len(seen) == 6


def test_augment_output_shape_and_range():
    import random
    random.seed(0)
    x = train_vsr._augment(np.full((20, 96, 96), 128, dtype=np.uint8))
    assert x.shape == (1, 20, 88, 88) and torch.isfinite(x).all()


def test_no_training_clips_fails_clearly():
    with pytest.raises(ValueError, match="at least one training clip"):
        train_vsr.finetune([], reader=FakeReader(), log=lambda *_: None)

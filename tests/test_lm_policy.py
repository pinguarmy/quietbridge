"""Personal language model policy: shared (not added) LM weight, a meaningful-gain saving rule,
and a forgetting check that is not made of evaluation text. Contracts with fake models only."""
import argparse

import pytest
import torch

from lipflow import train_lm
from lipflow.vsr import PERSONAL_LM_SHARE, PERSONAL_LM_WEIGHT, LipReader, lm_weights

# Text of the public sample clips (samples/2016-03-12.mov, samples/2017-01-07.mov) that the earlier
# forgetting check contained; tests and benchmarks decode these clips.
BENCHMARK_LINES = [
    "SINCE THE DAYS OF GEORGE WASHINGTON PRESIDENTS HAVE DELIVERED SOME FORM OF FINAL MESSAGE WHILE IN OFFICE",
    "A FAREWELL ADDRESS TO THE AMERICAN PEOPLE",
    "ON TUESDAY NIGHT IN CHICAGO I'LL DELIVER MINE",
    "I CHOSE CHICAGO NOT ONLY BECAUSE IT'S MY HOMETOWN WHERE I MET MY WIFE AND WE STARTED A FAMILY",
    "THIS PAST WEEK WE LOST AN AMERICAN ICON AND ONE OF THE MOST INFLUENTIAL FIGURES OF HER TIME",
    "BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE",
    "AS AN ACTRESS SHE APPEARED IN ELEVEN FILMS",
    "AND OFF SCREEN SHE STARRED IN A REAL LIFE HOLLYWOOD ROMANCE WITH THE LOVE OF HER LIFE",
]


def grams(text, n=4):
    w = text.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


# --- decoding weights ---------------------------------------------------------------

def test_shared_personal_lm_keeps_the_total_lm_weight():
    w = lm_weights(0.3, True)
    assert set(w) == {"lm", "plm"} and sum(w.values()) == pytest.approx(0.3)
    assert w["plm"] == pytest.approx(0.3 * PERSONAL_LM_SHARE)
    assert lm_weights(0.3, True, "shared", 0.25) == pytest.approx({"lm": 0.225, "plm": 0.075})
    assert lm_weights(0.3, True, "shared", 0.0) == {"lm": 0.3}
    assert lm_weights(0.3, True, "shared", 7.0) == pytest.approx({"lm": 0.0, "plm": 0.3})


def test_legacy_additive_mode_is_explicit_and_unchanged():
    assert lm_weights(0.3, True, "additive") == pytest.approx({"lm": 0.3, "plm": PERSONAL_LM_WEIGHT})
    assert lm_weights(0.3, True, "additive", 0.0) == {"lm": 0.3}


def test_no_personal_lm_or_no_lm_means_stock_weights_only():
    assert lm_weights(0.3, False) == {"lm": 0.3}
    assert lm_weights(0.0, True) == {"lm": 0.0}
    with pytest.raises(ValueError):
        lm_weights(0.3, True, "louder")


def test_reader_rejects_an_unknown_mode_before_loading_anything(monkeypatch):
    monkeypatch.setattr("lipflow.vsr.E2E", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="personal_lm_mode"):
        LipReader(personal=False, personal_lm_mode="additive-ish")


# --- forgetting check text --------------------------------------------------------------

def test_general_sample_is_not_benchmark_text():
    benchmark = set().union(*(grams(line) for line in BENCHMARK_LINES))
    assert len(train_lm.GENERAL_SAMPLE) >= 12
    for line in train_lm.GENERAL_SAMPLE:
        assert not grams(line) & benchmark, line
        assert train_lm.normalise(line) == [line]  # already LRS3-style


# --- saving rule ------------------------------------------------------------------------

class _FakeTok:
    def __init__(self, token_list):
        self.size = len(token_list)

    def __call__(self, s):
        return [1 + (sum(map(ord, word)) % (self.size - 2)) for word in s.split()]


@pytest.fixture
def fake_lm(monkeypatch):
    class FakeLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t):
            return self.param.sum() + 1.0, torch.tensor(1.0), torch.tensor(1)

    lm = FakeLM()
    monkeypatch.setattr(train_lm, "Tok", _FakeTok)
    monkeypatch.setattr("espnet.asr.asr_utils.get_model_conf", lambda *a: argparse.Namespace(backend="pytorch"))
    monkeypatch.setattr("espnet.nets.lm_interface.dynamic_import_lm", lambda *a: lambda *args, **kwargs: lm)
    monkeypatch.setattr("espnet.asr.asr_utils.torch_load", lambda *a: None)

    def run(yours_after, general_after, tmp_path, **kwargs):
        values = iter([50.0, 30.0, yours_after, general_after, general_after])
        monkeypatch.setattr(train_lm, "perplexity", lambda *a: next(values))
        out = tmp_path / "candidate.pth"
        res = train_lm.train(epochs=1, phrases=["I NEED SOME WATER PLEASE"],
                             validation_phrases=["CAN YOU CALL MY SISTER"], output_path=out, device="cpu", **kwargs)
        return res, out

    return run


def test_a_tiny_perplexity_gain_is_not_saved(fake_lm, tmp_path):
    res, out = fake_lm(50.0 * 0.985, 30.0, tmp_path)  # the 1.5 % gain measured on the real clips
    assert res["saved"] is False and not out.exists()
    assert res["reason"] == "validation_gain_below_minimum"
    assert "not recognition quality" in res["rule"]["note"]


def test_a_meaningful_gain_without_forgetting_is_saved(fake_lm, tmp_path):
    res, out = fake_lm(50.0 * (1 - train_lm.MIN_RELATIVE_GAIN) - 0.01, 30.0 * 1.04, tmp_path)
    assert res["saved"] is True and out.is_file() and res["reason"] == "saved"


def test_general_text_regression_blocks_saving(fake_lm, tmp_path):
    res, out = fake_lm(25.0, 30.0 * (train_lm.MAX_GENERAL_LOSS + 0.02), tmp_path)
    assert res["saved"] is False and not out.exists() and res["reason"] == "general_text_regressed"


def test_callers_may_still_pass_their_own_limits(fake_lm, tmp_path):
    res, _ = fake_lm(25.0, 30.0 * 1.10, tmp_path, max_general_loss=1.15)
    assert res["saved"] is True and res["rule"]["max_general_loss"] == 1.15

"""Gesture agreement must never replace a lip hypothesis that differs in a
negation, number or safety word, whatever its spelling."""
import pytest

from quietbridge.contracts import LipResult, SignEvent
from quietbridge.fusion import fuse, tokens

VOCAB = {"WATER": {"phrase": "water", "matches": ["water"], "verified": True}}
EVENTS = [SignEvent("WATER", 0.9, 100, 200, True)]


@pytest.mark.parametrize("top", [
    "I DIDN'T DRINK",           # contraction outside the old list
    "I DIDNT DRINK",
    "IT ISN'T COLD",
    "I WANT NOTHING",
    "NOBODY CAME",
    "NEITHER OF THEM",
    "I DONʼT WANT",        # modifier-letter apostrophe
    "I DON‘T WANT",
    "I NEED FIFTEEN",           # number words beyond ten
    "I NEED TWENTY",
    "TAKE 2ND",                 # digits inside a token
    "TAKE 10MG",
    "TAKE ２ PILLS",        # fullwidth digit
    "I NEED MEDICINES",         # plural safety word
])
def test_agreement_does_not_drop_sensitive_word_from_top_candidate(top):
    decision = fuse(LipResult([top, "I DRINK WATER"], 0), EVENTS, VOCAB, 300, 0)
    assert decision.text == top
    assert decision.reason == "conflict"
    assert decision.status == "confirmation_needed"


def test_plain_agreement_still_reranks_without_sensitive_words():
    decision = fuse(LipResult(["I DRINK MILK", "I DRINK WATER"], 0), EVENTS, VOCAB, 300, 0)
    assert (decision.text, decision.reason, decision.matched_signs) == ("I DRINK WATER", "agreement", ["WATER"])


def test_contraction_candidate_cannot_be_agreement_matched():
    # Both hypotheses carry the same negation; neither may be confirmed by an
    # object gesture, so the leading hypothesis stays and needs confirmation.
    decision = fuse(LipResult(["I DIDN'T WANT WATER", "I DIDN'T WANT WATER NOW"], 0), EVENTS, VOCAB, 300, 0)
    assert (decision.text, decision.reason) == ("I DIDN'T WANT WATER", "conflict")


def test_tokens_fold_apostrophes_and_fullwidth():
    assert tokens("DONʼT ７") == {"don't", "7"}

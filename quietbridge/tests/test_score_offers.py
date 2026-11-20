"""Model-scored phrase offers: prefilter, offer rule and the look-alike flag fix.

Software contracts with constructed scores; no lip model and no recognition
claim (see docs/experiments/real-clip-experiments.md for measured behaviour)."""
import math

import pytest

from quietbridge import phrase_match as pm


# --- look-alike counting is a flag, not a cost coincidence -------------------------

def test_plain_substitution_is_not_a_lookalike_even_when_the_discount_is_off(monkeypatch):
    assert not pm.lookalike("yes", "help") and not pm.lookalike("now", "car")
    monkeypatch.setattr(pm, "LOOKALIKE_COST", 1.0)
    cost, exact, look = pm._align(["yes", "now"], ["help", "car"])
    assert (cost, exact, look) == (2.0, 0, 0)
    # The short-phrase rule (every word exact or a look-alike) must still refuse it.
    assert pm.suggest(["HELP CAR"], ["Yes now"]) == []
    assert pm.suggest(["YES CAR"], ["Yes now"]) == []


def test_real_lookalike_is_still_counted_at_cost_one(monkeypatch):
    assert pm.lookalike("bat", "pat")
    monkeypatch.setattr(pm, "LOOKALIKE_COST", 1.0)
    assert pm._align(["pat", "it"], ["bat", "it"]) == (1.0, 1, 1)


def test_default_discount_unchanged():
    assert pm.LOOKALIKE_COST == 0.5
    assert pm._align(["pat", "it"], ["bat", "it"]) == (0.5, 1, 1)


# --- which phrases the model scores ---------------------------------------------

def test_small_profiles_score_every_distinct_unlisted_phrase():
    phrases = ["I need help", "i need help!", "", "Call Sam", "OPEN THE DOOR", 7]
    assert pm.phrases_to_score(["OPEN THE DOOR"], phrases) == [0, 3]
    assert pm.phrases_to_score(["   "], phrases) == [] and pm.phrases_to_score([], phrases) == []
    assert pm.phrases_to_score(["X"], phrases, existing=["Call Sam"]) == [0, 4]


def test_large_profiles_score_only_the_nearest_within_the_loose_ceiling(monkeypatch):
    monkeypatch.setattr(pm, "SCORE_ALL_PHRASES", 3)
    monkeypatch.setattr(pm, "SCORE_MAX_PHRASES", 2)
    phrases = ["please open the window", "please close the window", "open the door now",
               "what time is it", "completely unrelated words here today"]
    chosen = pm.phrases_to_score(["PLEASE OPEN THE WINDOWS"], phrases)
    assert chosen == [0, 1]
    # Looser than the offer threshold: two of four words wrong is still scored.
    assert pm.phrases_to_score(["PEAS HOPE THE WINDOWS"], phrases)[0] == 0
    assert pm.phrases_to_score(["ZZZ"], phrases) == []


def test_prefilter_is_bounded_by_max_phrases(monkeypatch):
    monkeypatch.setattr(pm, "SCORE_ALL_PHRASES", 1000)
    phrases = [f"phrase number {i} words" for i in range(pm.MAX_PHRASES + 30)]
    assert max(pm.phrases_to_score(["PHRASE NUMBER"], phrases)) < pm.MAX_PHRASES


# --- offer rule ---------------------------------------------------------------------

CANDS = ["I NEED SOME HELP", "I NEED SOME HELL"]


def test_a_variant_scoring_below_the_decoded_text_is_not_offered():
    phrases = ["I need some water", "I do not need some help"]
    scores = {0: -1.30, 1: -1.25}
    assert pm.model_offers(CANDS, phrases, scores, top_score=-1.20, margin=0.0) == []
    # ...the text matcher alone would have offered the variant
    assert [m.phrase for m in pm.suggest(CANDS, phrases)] == ["I need some water"]


def test_offers_are_ordered_by_model_score_and_capped():
    phrases = ["I need some hello", "I need some halp", "I need some helps", "Call Sam"]
    scores = {0: -1.10, 1: -0.90, 2: -1.00, 3: -0.5}
    offers = pm.model_offers(CANDS, phrases, scores, top_score=-1.20, margin=0.0)
    assert [o.phrase for o in offers] == ["Call Sam", "I need some halp"]
    assert offers[0].score_gap == pytest.approx(0.7)
    assert pm.model_offers(CANDS, phrases, scores, top_score=-1.20, margin=0.0, limit=1)[0].phrase == "Call Sam"
    assert len(pm.model_offers(CANDS, phrases, scores, -1.2, margin=0.0, limit=9)) == pm.MAX_SUGGESTIONS


def test_margin_admits_a_phrase_slightly_below_the_top():
    phrases = ["I need some halp"]
    assert pm.model_offers(CANDS, phrases, {0: -1.25}, -1.20, margin=0.1)[0].phrase == "I need some halp"
    assert pm.model_offers(CANDS, phrases, {0: -1.25}, -1.20, margin=0.01) == []


def test_listed_unknown_or_invalid_scores_are_never_offered():
    phrases = ["I need some help.", "I need water", "Stop"]
    scores = {0: 5.0, 1: float("nan"), 2: None, 9: 3.0, -1: 3.0}
    assert pm.model_offers(CANDS, phrases, scores, -1.2) == []
    assert pm.model_offers(CANDS, ["Stop"], {0: 1.0}, None) == []
    assert pm.model_offers(CANDS, ["Stop"], {0: 1.0}, math.inf) == []
    assert pm.model_offers(["  "], ["Stop"], {0: 1.0}, -1.0) == []


def test_sensitive_difference_and_text_free_evidence():
    offers = pm.model_offers(["I CAN COME"], ["I can not come"], {0: -0.5}, -1.0)
    assert offers[0].sensitive_difference is True
    evidence = offers[0].evidence()
    assert evidence["path"] == "model" and "phrase" not in evidence
    assert "can" not in repr(evidence).lower()

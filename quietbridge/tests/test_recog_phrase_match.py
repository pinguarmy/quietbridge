"""Pure phrase matcher: constructed examples only. These prove the matching
contract and its safety rules, not that suggestions help a real person."""
import pytest

from quietbridge import phrase_match as pm

PHRASES = ["Hello, how are you?", "I need some help.", "I would like some water.",
           "No, I do not agree.", "Please write it down.", "Yes", "Thank you"]


def offered(candidates, phrases=PHRASES, **kwargs):
    return [m.phrase for m in pm.suggest(candidates, phrases, **kwargs)]


def test_normalisation_folds_case_punctuation_unicode_and_small_numbers():
    assert pm.normalized("DONʼT  Stop!") == "dont stop"
    assert pm.normalized("I’m ＨＥＲＥ.") == "im here"
    assert pm.normalized("Take 2 at 10") == "take two at ten"
    assert pm.sensitive_words("I don't need 2") == {"dont", "two"}
    assert pm.sensitive_words("shan't") == {"shant"}


def test_lookalike_words_are_viseme_identical_and_never_sensitive():
    assert pm.lookalike("write", "ride") and pm.lookalike("yes", "yet")
    assert pm.lookalike("back", "pack") and pm.lookalike("fan", "van")
    assert not pm.lookalike("help", "hell")  # p is visible as a lip closure
    assert not pm.lookalike("not", "nod")    # same visemes, but a negation
    assert pm.word_cost("bat", "mat") == pm.LOOKALIKE_COST
    assert pm.word_cost("bat", "cup") == 1.0


def test_every_candidate_is_compared_and_evidence_names_the_best_one():
    matches = pm.match_phrases(["I WANT SOME TEA", "I NEED SOME HELL"], PHRASES)
    assert [m.phrase for m in matches] == ["I need some help."]
    assert matches[0].candidate_index == 1
    assert matches[0].distance == 0.25 and matches[0].exact_words == 3
    assert "phrase" not in matches[0].evidence()


def test_lookalike_substitution_is_cheaper_than_an_unrelated_word():
    match = pm.match_phrases(["PLEASE RIDE IT DOWN"], PHRASES)[0]
    assert match.phrase == "Please write it down."
    assert match.lookalike_words == 1 and match.distance == 0.125


def test_nothing_from_empty_or_missing_candidates():
    for candidates in ([], None, ["", "   "], ["!!!"]):
        assert pm.match_phrases(candidates, PHRASES) == []
        assert offered(candidates) == []


def test_below_threshold_offers_nothing():
    assert offered(["I NEED WATER"]) == []          # 1 of 3 words wrong: too much for 3 words
    assert offered(["I'D LIKE SOME WATER"]) == []   # 2 edits in 5 words
    assert offered(["HELLO"]) == []


def test_short_phrases_need_a_word_for_word_lookalike_reading():
    assert offered(["YET"]) == ["Yes"]
    assert offered(["YES PLEASE"]) == []      # extra word
    assert offered(["YOU"]) == []             # unrelated single word
    assert offered(["THANK YOU VERY"]) == []
    assert offered(["NOD"], ["No"]) == []     # a negation never matches by look-alike
    # A one-word difference in three words is allowed only as a look-alike.
    assert offered(["PLEASE BACK UP"], ["Please pack up"]) == ["Please pack up"]
    assert offered(["PLEASE TAKE UP"], ["Please pack up"]) == []


def test_identical_phrase_is_already_listed_not_offered_again():
    matches = pm.match_phrases(["HELLO HOW ARE YOU"], PHRASES)
    assert matches[0].already_listed
    assert offered(["HELLO HOW ARE YOU"]) == []
    assert offered(["I NEED SOME HELL"], existing=["I need some help"]) == []


def test_suggestions_are_capped_ranked_and_tie_broken_by_characters():
    phrases = ["I need some tea.", "I need some help.", "I need some milk.", "I need some home."]
    matches = pm.match_phrases(["I NEED SOME HELL"], phrases)
    assert len(matches) == 4 and {m.distance for m in matches} == {0.25}
    assert matches[0].phrase == "I need some help."   # closest spelling wins the tie
    assert len(pm.suggest(["I NEED SOME HELL"], phrases)) == pm.MAX_SUGGESTIONS == 2
    assert pm.suggest(["I NEED SOME HELL"], phrases, limit=5) == pm.suggest(["I NEED SOME HELL"], phrases)


def test_sensitive_difference_from_displayed_text_is_reported():
    match = pm.suggest(["I DO AGREE", "I DO NOT AGREE"], PHRASES)[0]
    assert match.phrase == "No, I do not agree." and match.sensitive_difference
    same = pm.suggest(["I NEED SOME HELL"], PHRASES)[0]
    assert not same.sensitive_difference
    # Compared with the displayed text, not with the best-matching candidate.
    flagged = pm.suggest(["NO I DO NOT AGREED"], ["No, I do not agree."], top="I DO AGREE")
    assert flagged[0].sensitive_difference


def test_bounded_inputs():
    long_phrase = " ".join(["word"] * (pm.MAX_WORDS + 1))
    assert pm.match_phrases([long_phrase], [long_phrase]) == []
    many = [f"I need item {i}" for i in range(pm.MAX_PHRASES + 10)]
    assert all(m.phrase_index < pm.MAX_PHRASES for m in pm.match_phrases(["I NEED ITEM"], many))


@pytest.mark.parametrize("value,enabled", [("", True), ("1", True), ("0", False), ("off", False),
                                           ("False", False), ("no", False)])
def test_environment_switch(value, enabled):
    assert pm.enabled_by_environment({pm.ENV_SWITCH: value}) is enabled


def test_rescue_outcome_counts_rescues_and_wrong_offers():
    phrases = ["I need some help.", "Please write it down."]
    rescued = pm.rescue_outcome(["I NEED SOME HELL", "I NEED SOME HEL"], "I need some help", phrases)
    assert rescued["rescued"] and not rescued["top1_exact"] and rescued["top1_or_suggestion_exact"]
    assert rescued["reference_in_phrases"] and not rescued["wrong_offer"]
    wrong = pm.rescue_outcome(["I NEED SOME HELL"], "I need some hell", phrases)
    assert wrong["top1_exact"] and wrong["wrong_offer"] and not wrong["rescued"]
    nothing = pm.rescue_outcome([], "I need some help", phrases)
    assert nothing["offered"] == 0 and not nothing["top1_or_suggestion_exact"]
    summary = pm.summarize_rescue([rescued, wrong, nothing])
    assert summary["count"] == 3 and summary["rescued"] == 1
    assert summary["top1_exact"] == 1 and summary["top1_or_suggestion_exact"] == 2
    assert summary["wrong_offers"] == 1 and summary["wrong_offer_rate"] == 0.5
    assert summary["reference_in_phrases"]["count"] == 2
    assert summary["reference_not_in_phrases"]["wrong_offers"] == 1
    assert pm.summarize_rescue([])["top1_exact_rate"] is None

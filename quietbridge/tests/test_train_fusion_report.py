"""Evaluation-report completeness: recall scopes and the phrase-suggestion section."""
from __future__ import annotations

import quietbridge.evaluate_fusion as evaluator
from quietbridge.evaluate_fusion import evaluate, phrase_suggestion_section
from quietbridge.tests.test_evaluate_fusion import Motion, Reader, samples


def test_recall_is_reported_over_all_labels_and_text_producing_labels(tmp_path, monkeypatch):
    profile, _, _ = samples(tmp_path)
    vocabulary = {"WATER": {"display_name": "Water", "phrase": "water", "matches": ["water"], "enabled": False}}
    monkeypatch.setattr(evaluator, "profile_vocabulary", lambda p: vocabulary)
    report = evaluate(profile, "test", Reader(), Motion())
    split = report["summary"]["gesture_recall_by_text_output"]
    assert split["all_vocabulary_labels"] == {"labels": ["WATER"], "count": 1, "correct": 1, "recall": 1.0}
    # WATER is disabled: recognising it can never produce text.
    assert split["text_producing"] == {"labels": [], "count": 0, "correct": 0, "recall": None}
    assert split["not_text_producing"]["count"] == 1
    assert report["summary"]["gesture_classification"]["recall"] == 1  # unchanged, all-label scope
    vocabulary["WATER"]["enabled"] = True
    split = evaluate(profile, "test", Reader(), Motion())["summary"]["gesture_recall_by_text_output"]
    assert split["text_producing"]["recall"] == 1.0 and split["not_text_producing"]["recall"] is None


def test_phrase_suggestion_section_is_wired_into_the_fusion_report(tmp_path):
    profile, _, _ = samples(tmp_path)
    profile.lip_prompts = ["water", "Please pass me the water bottle now"]
    report = evaluate(profile, "test", Reader(), Motion())
    section = report["summary"]["phrase_suggestions"]
    assert section["measured"] is True and section["phrases"] == 2 and section["count"] == 1
    assert section["reference_in_phrases"]["count"] == 1  # the recording prompt is a saved phrase
    assert "phrase_suggestion" in report["pairs"][0]
    assert {"rescued", "wrong_offer", "top1_exact"} <= set(report["pairs"][0]["phrase_suggestion"])


def test_phrase_section_without_phrases_is_unmeasured_not_zero():
    assert phrase_suggestion_section([], []) == {"measured": False, "reason": "profile has no saved phrases"}
    rows = [{"raw_candidates": [{"text": "please pass me the water bottle now"}],
             "reference": "Please pass me the water bottle now"}]
    section = phrase_suggestion_section(rows, ["Please pass me the water bottle now"])
    assert section["top1_exact"] == 1 and section["rescued"] == 0
    assert section["reference_not_in_phrases"]["wrong_offer_rate"] is None  # no rows: not measured

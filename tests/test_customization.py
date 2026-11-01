"""Tests for customization, practice corpus generation, tuning replay, and metrics logic."""
from __future__ import annotations

import json
import os
import pytest

from lipflow import customize, dictation, practice, tune


def test_merge_unique_order_and_duplicates():
    existing = ["apple", "Banana", "cherry"]
    incoming = ["banana", "date", "Apple", "fig"]
    merged = customize.merge_unique(existing, incoming, case_insensitive=True)
    assert merged == ["apple", "Banana", "cherry", "date", "fig"]


def test_init_starter_files_non_destructive(tmp_path, monkeypatch):
    words_file = tmp_path / "words.txt"
    phrases_file = tmp_path / "phrases.txt"
    monkeypatch.setattr(customize, "WORDS_PATH", str(words_file))
    monkeypatch.setattr(customize, "PHRASES", str(phrases_file))

    # Pre-populate existing phrase
    phrases_file.write_text("Existing custom phrase for test.\n", encoding="utf-8")

    res = customize.init_starter_files()
    assert res["total_phrases"] >= 1
    # Check that existing phrase was NOT wiped out
    content = phrases_file.read_text(encoding="utf-8")
    assert "Existing custom phrase for test." in content
    # Generic starters were merged
    assert "Sounds good to me, let us proceed." in content


def test_import_plain_text_merges_without_overwrite(tmp_path, monkeypatch):
    words_file = tmp_path / "words.txt"
    phrases_file = tmp_path / "phrases.txt"
    monkeypatch.setattr(customize, "WORDS_PATH", str(words_file))
    monkeypatch.setattr(customize, "PHRASES", str(phrases_file))

    # Existing content
    words_file.write_text("OriginalWord\n", encoding="utf-8")
    phrases_file.write_text("Original phrase number one.\n", encoding="utf-8")

    # Import plain text
    res = customize.import_plain_text(
        phrases=["New phrase number two.", "Original phrase number one."],
        words=["NewWord", "OriginalWord"],
    )

    assert res["phrases_added"] == 1
    assert res["words_added"] == 1

    words_content = words_file.read_text(encoding="utf-8")
    phrases_content = phrases_file.read_text(encoding="utf-8")

    assert "OriginalWord" in words_content
    assert "NewWord" in words_content
    assert "Original phrase number one." in phrases_content
    assert "New phrase number two." in phrases_content


def test_custom_practice_sentences_corpus(tmp_path, monkeypatch):
    custom_phrases = tmp_path / "custom_phrases.txt"
    # Provide 10 custom sentences
    lines = [
        "This is my first custom test sentence",
        "We are verifying the customized practice flow",
        "Another sentence to make sure it loads",
        "The model learns from your everyday words",
        "Short sentence one two three four five",
        "Let us test this thoroughly today",
    ]
    custom_phrases.write_text("\n".join(lines), encoding="utf-8")

    sentences = practice.practice_sentences(n=10, phrases_path=str(custom_phrases))
    assert len(sentences) == 10
    # Must contain both custom sentences and Harvard fallback sentences
    custom_count = sum(1 for s in sentences if any(s.lower() in l.lower() for l in lines))
    harvard_count = sum(1 for s in sentences if s in practice.HARVARD)
    assert custom_count > 0
    assert harvard_count > 0
    assert custom_count + harvard_count == 10


def test_update_and_get_custom_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(dictation, "SETTINGS", str(settings_file))

    res = customize.update_custom_settings(
        cleanup_backend="gemini",
        cleanup_style="polish",
        gemini_model="gemini-3.8-flash",
        cpa_base_url="https://cpa.peterhpy.dpdns.org/v1",
        cleanup_timeout=10.0,
    )

    assert res["cleanup_backend"] == "gemini"
    assert res["cleanup_style"] == "polish"
    assert res["gemini_model"] == "gemini-3.8-flash"
    assert res["cpa_base_url"] == "https://cpa.peterhpy.dpdns.org/v1"
    assert res["cleanup_timeout"] == 10.0

    loaded = dictation.load_settings()
    assert loaded["cleanup_backend"] == "gemini"


def test_wer_calculation():
    # Exact match
    edits, ref_len, wer = tune.compute_wer("hello world", "Hello, World!")
    assert edits == 0
    assert wer == 0.0

    # 1 edit
    edits, ref_len, wer = tune.compute_wer("hello there", "hello world")
    assert edits == 1
    assert ref_len == 2
    assert wer == 0.5


def test_semantic_risk_detection():
    # Negation dropped
    r1 = tune.detect_semantic_risks("I will not attend the meeting", "I will attend the meeting")
    assert r1["negation_risk"] is True
    assert "not" in r1["negation_diff"]

    # Number altered
    r2 = tune.detect_semantic_risks("We have twenty apples", "We have 30 apples")
    assert r2["number_risk"] is True

    # Custom entity dropped
    r3 = tune.detect_semantic_risks("Ping Miguel right now", "Ping someone right now", custom_words=["Miguel"])
    assert r3["entity_risk"] is True
    assert "miguel" in r3["entity_diff"]

    # No risks
    r4 = tune.detect_semantic_risks("The birch canoe slid on smooth planks", "The birch canoe slid on smooth planks.")
    assert r4["has_risk"] is False


def test_replay_evaluation_faithful_vs_polish(tmp_path):
    # Mock cleaner
    class MockCleaner:
        def __init__(self, backend="basic", style="faithful"):
            self.backend = backend
            self.style = style
            self.model = "mock-model"
            self.last_result = {"cleanup_ms": 12.5}

        def __call__(self, candidates, words=None, **kwargs):
            if not candidates:
                return ""
            if self.style == "polish":
                # Simulated polish rewriting
                return "Sounds good to me, let us proceed."
            # Faithful formatting
            return candidates[0].capitalize() + "."

    dataset = [
        {"candidates": ["SOUNDS GOOD TO ME"], "reference": "Sounds good to me.", "id": "1"},
        {"candidates": ["I AM NOT SURE"], "reference": "I am not sure.", "id": "2"},
    ]

    # Faithful test: WER must be calculated
    faithful_rep = tune.replay_evaluation(
        dataset,
        style="faithful",
        cleaner_instance=MockCleaner(style="faithful"),
    )
    assert faithful_rep["wer"] is not None
    assert faithful_rep["style"] == "faithful"
    assert faithful_rep["sample_count"] == 2
    assert faithful_rep["cleanup_latency_ms"]["p50"] > 0

    # Polish test: WER must be None (not misleading)
    polish_rep = tune.replay_evaluation(
        dataset,
        style="polish",
        cleaner_instance=MockCleaner(style="polish"),
    )
    assert polish_rep["wer"] is None
    assert "omitted" in polish_rep["wer_note"].lower()
    assert polish_rep["style"] == "polish"

    md = tune.generate_markdown_report(polish_rep)
    assert "Omitted" in md
    assert "Performance & Latency" in md

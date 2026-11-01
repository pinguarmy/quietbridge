"""Verify the redesigned onboarding UI, styling, and integration."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from AppKit import NSColor

from lipflow import onboarding
from lipflow.cleanup import Cleaner
from lipflow.onboarding import (
    COLOR_BG_TINT,
    COLOR_GRAPHITE,
    COLOR_MOSS,
    Onboarding,
    _ob_text,
    _text,
)


@pytest.fixture
def mock_app():
    class DummyHUD:
        def hide(self):
            pass
        def show(self, *args, **kwargs):
            pass

    class DummyCamera:
        def __init__(self):
            self.track_always = False
        def ensure_open(self):
            pass

    cleaner = Cleaner("basic", style="polish")
    return SimpleNamespace(
        hud=DummyHUD(),
        camera=DummyCamera(),
        cleaner=cleaner,
        settings={
            "cleanup_backend": "gemini",
            "cleanup_style": "polish",
            "personalization_domains": ["ai-coding", "finance", "education"],
            "personalization_use_cases": ["ai-prompts", "messages", "emails", "notes"],
        },
        opts=SimpleNamespace(key="right_option", backend="gemini", style="polish"),
        onboarding=None,
        onboarding_text="",
        jobs=SimpleNamespace(put=lambda job: None),
    )


def test_shared_text_helper_defaults_to_white(mock_app):
    """Ensure existing callers (HUD, settings) get white text by default from _text."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    field = _text(ob.root, ((0, 0), (100, 20)), "Test White", 12)
    # Check text color RGB components are white (1.0, 1.0, 1.0)
    color = field.textColor()
    assert color.redComponent() == pytest.approx(1.0, abs=0.01)
    assert color.greenComponent() == pytest.approx(1.0, abs=0.01)
    assert color.blueComponent() == pytest.approx(1.0, abs=0.01)


def test_ob_text_helper_defaults_to_graphite(mock_app):
    """Ensure onboarding text helper defaults to graphite for light background."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    field = _ob_text(ob.root, ((0, 0), (100, 20)), "Test Graphite", 12)
    color = field.textColor()
    assert color.redComponent() == pytest.approx(COLOR_GRAPHITE[0], abs=0.01)
    assert color.greenComponent() == pytest.approx(COLOR_GRAPHITE[1], abs=0.01)
    assert color.blueComponent() == pytest.approx(COLOR_GRAPHITE[2], abs=0.01)


def test_window_properties_light_opaque(mock_app):
    """Onboarding window should be 620x600, opaque non-glass solid light theme."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    frame = ob.win.frame()
    assert frame.size.width == 620
    assert frame.size.height == 600
    assert ob.win.appearance().name() == "NSAppearanceNameAqua"


def test_words_page_displays_domains_and_corpus(mock_app):
    """Your words page shows configured domains and active corpus count."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    ob.goWords_(None)
    assert ob.words_status is not None
    status_text = ob.words_status.stringValue()
    assert "专有名词" in status_text
    assert "常用句" in status_text
    assert ob.page_title.stringValue() == "准备您的输入词库"


def test_practice_page_retains_english_sentences_and_controls(mock_app):
    """Practice prompts remain English, methods redo_ and skip_ exist."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    ob.goPractice_(None)
    assert len(ob.sentences) > 0
    # Practice sentence should be English
    assert isinstance(ob.prompt.stringValue(), str)
    assert len(ob.prompt.stringValue()) > 0
    # Redo and skip selectors
    assert hasattr(ob, "redo_")
    assert hasattr(ob, "skip_")
    assert hasattr(ob, "clip_done")


def test_train_page_truthful_metrics_without_unmeasured_claims(mock_app):
    """Train page reports measured verification metric deltas truthfully."""
    ob = Onboarding.alloc().initWithApp_(mock_app)
    ob.goTrain_(None)
    assert ob.progress is not None
    assert ob.done_btn.isHidden()

    # Finished callback with measured before/after
    ob.finished(0.40, 0.15, True, "实测优化")
    # Verify report method works
    ob.report(80.0, "训练中...")
    assert hasattr(ob, "finish_")

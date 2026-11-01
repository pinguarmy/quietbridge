"""Wiring: CLI flags → Options/Cleaner, persisted cleanup_backend/cleanup_style, and the
model-thread Cleaner swap the Settings window / tray queue. No camera, no network, no UI."""
import sys
import types
from types import SimpleNamespace

import pytest

from lipflow import cloud
from lipflow.__main__ import main
import lipflow.__main__ as main_mod
from lipflow.cleanup import Cleaner


# -- persisted settings --------------------------------------------------------

def test_cleaner_no_args_honors_persisted_gemini_polish(monkeypatch):
    """The app started with no CLI flags must pick up a saved gemini/polish config."""
    monkeypatch.setattr("lipflow.dictation.load_settings",
                        lambda: {"cleanup_backend": "gemini", "cleanup_style": "polish"})
    monkeypatch.setattr(cloud, "resolve_cpa_key", lambda: None)
    c = Cleaner("auto", style=None)
    assert c.backend == "gemini"
    assert c.style == "polish"


def test_explicit_backend_and_style_beat_settings(monkeypatch):
    monkeypatch.setattr("lipflow.dictation.load_settings",
                        lambda: {"cleanup_backend": "gemini", "cleanup_style": "polish"})
    c = Cleaner("basic", style="faithful")
    assert c.backend == "basic"
    assert c.style == "faithful"


# -- CLI ------------------------------------------------------------------------

def _capture_app(monkeypatch):
    captured = {}

    class FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(main_mod, "_app", lambda: (FakeOptions, lambda opts: None))
    return captured


def test_cli_run_passes_cleanup_and_style(monkeypatch):
    captured = _capture_app(monkeypatch)
    main(["run", "--cleanup", "gemini", "--cleanup-style", "polish"])
    assert captured["backend"] == "gemini"
    assert captured["style"] == "polish"


def test_cli_run_defaults_leave_cleanup_to_settings(monkeypatch):
    captured = _capture_app(monkeypatch)
    main([])
    assert captured["backend"] == "auto"
    assert captured["style"] is None


def test_cli_file_passes_cleanup_style(monkeypatch, capsys):
    made = {}

    class FakeCleaner:
        def __init__(self, backend, style=None):
            made.update(backend=backend, style=style)
            self.style = style or "faithful"

        def __call__(self, candidates, **kw):
            return "Text."

        def describe(self):
            return "fake"

    # fake the heavy pipeline modules so the wiring is all that runs
    offline = types.ModuleType("lipflow.offline")
    offline.transcribe_file = lambda *a, **k: "RAW WORDS"
    vsr = types.ModuleType("lipflow.vsr")
    vsr.LipReader = lambda beam_size: object()
    monkeypatch.setitem(sys.modules, "lipflow.offline", offline)
    monkeypatch.setitem(sys.modules, "lipflow.vsr", vsr)
    monkeypatch.setattr("lipflow.cleanup.Cleaner", FakeCleaner)
    main(["file", "clip.mp4", "--cleanup", "gemini", "--cleanup-style", "polish"])
    assert made == {"backend": "gemini", "style": "polish"}
    assert "polish" in capsys.readouterr().out


def test_customize_and_tune_dispatch_before_ui(monkeypatch):
    """lipflow customize/tune reach their own argparse handlers (no hotkey/UI needed)."""
    calls = []
    monkeypatch.setattr("lipflow.customize.customize_handler",
                        lambda argv=None: calls.append(("customize", argv)) or 0)
    monkeypatch.setattr("lipflow.tune.tune_handler",
                        lambda argv=None: calls.append(("tune", argv)) or 0)
    with pytest.raises(SystemExit) as e1:
        main(["customize", "--status"])
    with pytest.raises(SystemExit) as e2:
        main(["tune", "--backend", "gemini", "--style", "polish"])
    assert e1.value.code == 0 and e2.value.code == 0
    assert calls == [("customize", ["--status"]), ("tune", ["--backend", "gemini", "--style", "polish"])]


# -- macOS app wiring -----------------------------------------------------------

def test_worker_cleanup_job_swaps_cleaner_and_updates_menu(monkeypatch):
    """Settings queues ('cleanup', backend, style); the model thread builds the Cleaner."""
    pytest.importorskip("AppKit")
    from lipflow import app, cleanup
    monkeypatch.setattr(cleanup, "_settings", lambda: {})
    monkeypatch.setattr(cloud, "resolve_cpa_key", lambda: None)
    monkeypatch.setattr(app, "ui", lambda fn, *a, **kw: fn(*a, **kw))
    titles = []
    controller = SimpleNamespace(cleaner=None,
                                 cleanup_item=SimpleNamespace(setTitle_=titles.append))
    controller._cleanup_label = lambda: app.Lipflow._cleanup_label(controller)
    app.Lipflow._set_cleanup(controller, "gemini", "polish")
    assert controller.cleaner.backend == "gemini"
    assert controller.cleaner.style == "polish"
    assert titles, "the menu label must be refreshed after the swap"
    assert "gemini" in titles[-1] and "polish" in titles[-1]


def test_faithful_cleanup_still_learns_from_corrections(monkeypatch):
    """The faithful gate keeps corrections learning on; only polished output is excluded."""
    pytest.importorskip("AppKit")
    from lipflow import app, cleanup, corrections
    output, learned = [], []
    monkeypatch.setattr(cleanup, "_settings", lambda: {})
    monkeypatch.setattr(cloud, "resolve_cpa_key", lambda: "fake")
    monkeypatch.setattr(cloud, "cleanup", lambda *a, **k: "I think we should send the email tomorrow.")
    monkeypatch.setattr(app, "clip_problem", lambda rec: None)
    monkeypatch.setattr(app, "ui", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(app, "paste_text", output.append)
    monkeypatch.setattr(corrections, "Watcher", lambda *a, **k: learned.append(a) or SimpleNamespace(start=lambda: None))
    reader = SimpleNamespace(encode=lambda rois: rois,
                             beam_search=lambda enc, nbest: ["I THINK WE SHOULD SEND THE EMAIL TOMORROW"])
    cleaner = cleanup.Cleaner("gemini", style="faithful")
    rec = SimpleNamespace(duration=3.0)
    controller = SimpleNamespace(
        onboarding=None, reader=reader, cleaner=cleaner,
        _rois=lambda rec: "mouth-crops", _av_candidates=lambda rec, rois: None,
        hud=SimpleNamespace(set_text=lambda *a: None, show=lambda *a, **kw: None),
        ctx=SimpleNamespace(names=[], app="fixture", element=object(), value=""),
        context=[], last_paste_at=0.0, last_output="",
        opts=SimpleNamespace(paste=True), settings={"learn_corrections": True},
        _log=lambda *a, **kw: None, _save_clip=lambda *a: None,
    )
    app.Lipflow._final(controller, rec)
    assert output == ["I think we should send the email tomorrow."]
    assert len(learned) == 1, "faithful output is word-for-word and must feed corrections learning"


def test_log_history_receives_cleaner_last_result(monkeypatch):
    pytest.importorskip("AppKit")
    from lipflow import app
    seen = {}
    monkeypatch.setattr(app, "log_history", lambda *a, **kw: seen.update(args=a, kw=kw))
    last = {"backend": "gemini", "style": "polish", "fallback": None, "cleanup_ms": 12}
    cleaner = SimpleNamespace(describe=lambda: "gemini (gemini-3.8-flash)", last_result=last)
    app.Lipflow._log(SimpleNamespace(cleaner=cleaner), SimpleNamespace(duration=1.0), ["A"], "A.", 0.5)
    assert seen["kw"]["cleanup_details"] == last


# -- settings window constants ---------------------------------------------------

def test_settings_window_backend_style_choices_match_cleaner():
    pytest.importorskip("AppKit")
    from lipflow import settings_window
    keys = [k for _, k in settings_window.BACKENDS]
    assert keys == ["auto", "gemini", "local", "ollama", "claude", "basic"]
    assert [k for _, k in settings_window.STYLES] == ["faithful", "polish"]
    # every popup title maps back to its key (both lookup directions used by the window)
    assert dict(settings_window.BACKENDS)["Gemini (cloud, text only)"] == "gemini"
    assert dict(settings_window.STYLES)["Polish (also tidy phrasing)"] == "polish"

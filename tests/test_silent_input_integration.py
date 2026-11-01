"""Exercise the actual macOS final-dictation path without camera or real keystrokes."""
from types import SimpleNamespace

import pytest

pytest.importorskip("AppKit")

from lipflow import app, cleanup, cloud


def test_polished_cloud_text_reaches_cursor_path_without_learning_rewrites(monkeypatch):
    output, histories, clips, learned = [], [], [], []
    monkeypatch.setattr(cleanup, "_settings", lambda: {})
    monkeypatch.setattr(cloud, "resolve_cpa_key", lambda: "fake")
    monkeypatch.setattr(cloud, "cleanup", lambda *a, **k: "I think we should send the email tomorrow.")
    monkeypatch.setattr(app, "clip_problem", lambda rec: None)
    monkeypatch.setattr(app, "ui", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(app, "paste_text", output.append)
    from lipflow import corrections
    monkeypatch.setattr(corrections, "Watcher", lambda *a, **k: learned.append(a))
    reader = SimpleNamespace(encode=lambda rois: rois,
                             beam_search=lambda enc, nbest: ["I THINK UM WE SHOULD WE SHOULD SEND THE EMAIL TOMORROW"])
    cleaner = cleanup.Cleaner("gemini", style="polish")
    rec = SimpleNamespace(duration=3.0)
    controller = SimpleNamespace(
        onboarding=None, reader=reader, cleaner=cleaner,
        _rois=lambda rec: "mouth-crops-local-only", _av_candidates=lambda rec, rois: None,
        hud=SimpleNamespace(set_text=lambda *a: None, show=lambda *a, **kw: None),
        ctx=SimpleNamespace(names=[], app="fixture", element=object(), value=""),
        context=[], last_paste_at=0.0, last_output="",
        opts=SimpleNamespace(paste=True), settings={"learn_corrections": True},
        _log=lambda *a: histories.append(a), _save_clip=lambda *a: clips.append(a),
    )
    app.Lipflow._final(controller, rec)
    assert output == ["I think we should send the email tomorrow."]
    assert controller.last_output == output[0]
    assert controller.context == output
    assert len(histories) == len(clips) == 1
    assert not learned, "Polish rewrites must not become word-for-word visual training targets"
    assert cleaner.last_result["backend"] == "gemini"


def test_network_failure_still_produces_nonempty_pastable_text(monkeypatch):
    monkeypatch.setattr(cleanup, "_settings", lambda: {})
    monkeypatch.setattr(cloud, "resolve_cpa_key", lambda: "fake")
    monkeypatch.setattr(cloud, "cleanup", lambda *a, **k: None)
    cleaner = cleanup.Cleaner("gemini", style="polish")
    monkeypatch.setattr(cleaner, "_local_fallback", lambda *a: None)
    assert cleaner(["I DO NOT WANT TO CHANGE THE MEETING"], words=[]) == "I do not want to change the meeting."
    assert cleaner.last_result["backend"] == "basic"
    assert cleaner.last_result["fallback"]

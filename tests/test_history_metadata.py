import builtins
import json
from types import SimpleNamespace

from lipflow import dictation


def test_history_keeps_actual_cleanup_metadata_not_credentials(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(dictation, "HISTORY", str(path))
    dictation.log_history(SimpleNamespace(duration=2.5), ["HELLO"], "Hello.", 1.25,
                           "gemini (gemini-3.8-flash)",
                           {"backend": "local", "style": "polish", "fallback": True,
                            "cleanup_ms": 800, "api_key": "must-not-be-recorded",
                            "response": {"headers": "must-not-be-recorded"}})
    row = json.loads(path.read_text())
    assert row["cleanup_details"] == {"backend": "local", "style": "polish", "fallback": True,
                                      "cleanup_ms": 800}
    assert row["latency"] == 1.25
    assert "must-not-be-recorded" not in path.read_text()


def test_settings_atomic_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "profile" / "settings.json"
    monkeypatch.setattr(dictation, "SETTINGS", str(path))
    settings = {"cleanup_backend": "gemini", "cleanup_style": "polish", "existing": "保留"}
    dictation.save_settings(settings)
    assert dictation.load_settings() == settings
    assert list(path.parent.iterdir()) == [path]


def test_settings_load_keeps_unicode_with_windows_default_encoding(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(dictation, "SETTINGS", str(path))
    settings = {"existing": "保留", "custom_words": ["José", "QuietBridge"]}
    dictation.save_settings(settings)

    def windows_default_open(file, *args, **kwargs):
        # Emulate Windows' legacy default without changing the runner's locale.
        kwargs.setdefault("encoding", "cp1252")
        return builtins.open(file, *args, **kwargs)

    monkeypatch.setattr(dictation, "open", windows_default_open, raising=False)

    assert dictation.load_settings() == settings


def test_history_old_call_remains_compatible(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(dictation, "HISTORY", str(path))
    dictation.log_history(SimpleNamespace(duration=1), ["A"], "A.", 0.4, "basic")
    assert json.loads(path.read_text())["cleanup"] == "basic"

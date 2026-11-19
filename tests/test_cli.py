"""The `lipflow` CLI parses arguments and runs doctor/file without a global-hotkey backend.

The platform hotkey modules (macOS Quartz, Windows pynput) and the GUI frameworks are blocked here,
so these tests prove the same thing on macOS and Windows as on headless Linux."""
import ast
import os
import sys

import pytest

import lipflow.__main__ as main_mod
from lipflow.__main__ import main

BLOCKED = ["Quartz", "AppKit", "Foundation", "AVFoundation", "objc", "PyObjCTools",
           "pynput", "pynput.keyboard", "pystray", "tkinter"]
HOTKEY_USERS = ["lipflow.hotkey", "lipflow.app", "lipflow.hud", "lipflow.paste", "lipflow.onboarding",
                "lipflow.settings_window", "lipflow.win.hotkey", "lipflow.win.app", "lipflow.win.hud",
                "lipflow.win.paste"]


@pytest.fixture()
def no_hotkey_backends(monkeypatch):
    """Make every hotkey/GUI framework import fail, as on a machine that lacks them."""
    for name in BLOCKED:
        monkeypatch.setitem(sys.modules, name, None)  # `import name` now raises ImportError
    for name in HOTKEY_USERS:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    leaked = [name for name in HOTKEY_USERS if sys.modules.get(name) is not None]
    assert not leaked, f"the CLI imported {leaked}"


@pytest.mark.usefixtures("no_hotkey_backends")
def test_help_works_without_any_hotkey_backend(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for command in ("run", "file", "doctor", "train-lm", "import-wispr"):
        assert command in out


@pytest.mark.usefixtures("no_hotkey_backends")
@pytest.mark.parametrize("command", ["file", "doctor", "run", "onboard", "train-lm", "import-wispr"])
def test_subcommand_help_works_without_any_hotkey_backend(command, capsys):
    with pytest.raises(SystemExit) as e:
        main([command, "--help"])
    assert e.value.code == 0
    assert f"usage: lipflow {command}" in capsys.readouterr().out


@pytest.mark.usefixtures("no_hotkey_backends")
def test_doctor_runs_to_its_summary_without_any_hotkey_backend(monkeypatch, capsys):
    monkeypatch.setattr("lipflow.paths.MODELS", "/nonexistent/models")
    monkeypatch.setattr("lipflow.doctor.MODELS", "/nonexistent/models")
    with pytest.raises(SystemExit) as e:
        main(["doctor", "--no-camera"])
    out = capsys.readouterr().out
    assert "Summary:" in out
    assert "Traceback" not in out
    # Models are missing here, so the doctor must say so and fail, never print a pass for them.
    assert e.value.code == 1
    assert "FAIL  models/vsr/model.pth: missing" in out
    assert "SKIP  camera: not opened (--no-camera)" in out


@pytest.mark.parametrize("platform", ["linux", "freebsd14"])
def test_run_and_onboard_explain_the_platform_requirement(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    for argv in ([], ["run"], ["onboard"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert "needs macOS or Windows" in str(e.value.code) and platform in str(e.value.code)


@pytest.mark.usefixtures("no_hotkey_backends")
def test_run_on_macos_without_pyobjc_is_one_line_not_a_traceback(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SystemExit) as e:
        main(["run"])
    assert str(e.value.code).startswith("lipflow: the dictation app could not load")


def test_file_reports_a_missing_video_and_missing_models(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["file", str(tmp_path / "nope.mp4")])
    assert "no such video file" in str(e.value.code)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr("lipflow.paths.MODELS", str(tmp_path / "models"))
    with pytest.raises(SystemExit) as e:
        main(["file", str(clip)])
    assert "missing model files" in str(e.value.code) and "vsr/model.pth" in str(e.value.code)


def _module_constants(*parts):
    """Literal top-level assignments of a hotkey module, read from its source: importing it would need
    Quartz (macOS) or a pynput keyboard backend (Windows; a display on Linux)."""
    src = os.path.join(os.path.dirname(main_mod.__file__), *parts)
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {t.id: node.value for node in tree.body if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name)}


def test_cli_key_lists_match_the_hotkey_modules():
    """The CLI keeps the key names itself so it never imports Quartz/pynput just to parse --key."""
    mac = _module_constants("hotkey.py")
    assert tuple(ast.literal_eval(k) for k in mac["KEYS"].keys) == main_mod.MAC_KEYS
    assert main_mod.MAC_DEFAULT_KEY in main_mod.MAC_KEYS
    win = _module_constants("win", "hotkey.py")
    assert tuple(ast.literal_eval(k) for k in win["KEYS"].keys) == main_mod.WIN_KEYS
    assert ast.literal_eval(win["DEFAULT_KEY"]) == main_mod.WIN_DEFAULT_KEY


def test_import_wispr_errors_are_one_line(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["import-wispr", "--from-text", str(tmp_path / "missing.txt")])
    assert str(e.value.code).startswith("lipflow import-wispr:")

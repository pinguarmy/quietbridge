"""Server command line: browser camera is the default camera source."""
import json
from unittest.mock import Mock, patch

import pytest

from hackathon import server as server_module
from hackathon.runtime import Runtime


def run_main(argv, monkeypatch, tmp_path):
    monkeypatch.setenv("QUIETBRIDGE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("QUIETBRIDGE_MOTION_MODEL", raising=False)
    captured = {}
    real_make_server = server_module.make_server

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        # Never open a camera or load models in this test.
        return Runtime(rehearsal=True, vocabulary=kwargs.get("vocabulary"),
                       profile=kwargs.get("profile"), profile_manager=kwargs.get("profile_manager"),
                       configuration_warning=kwargs.get("configuration_warning", ""))

    def fake_make_server(runtime, port):
        server = real_make_server(runtime, 0)
        server.serve_forever = Mock(side_effect=KeyboardInterrupt)
        captured["runtime"] = runtime
        return server

    with patch("sys.argv", ["hackathon.server", *argv, "--port", "0"]), \
            patch.object(server_module, "Runtime", fake_runtime), \
            patch.object(server_module, "make_server", fake_make_server):
        server_module.main()
    return captured


def test_no_flags_uses_the_browser_camera(monkeypatch, tmp_path, capsys):
    captured = run_main([], monkeypatch, tmp_path)
    assert captured["browser_camera"] is True and captured["rehearsal"] is False
    assert "BROWSER CAMERA" in capsys.readouterr().out


def test_browser_camera_flag_is_an_accepted_no_op(monkeypatch, tmp_path):
    assert run_main(["--browser-camera"], monkeypatch, tmp_path)["browser_camera"] is True


def test_server_camera_opens_the_local_camera(monkeypatch, tmp_path, capsys):
    captured = run_main(["--server-camera", "--camera", "2"], monkeypatch, tmp_path)
    assert captured["browser_camera"] is False and captured["camera"] == 2
    assert "SERVER CAMERA 2" in capsys.readouterr().out
    assert run_main(["--server-camera"], monkeypatch, tmp_path)["camera"] == 0


def test_rehearsal_is_not_a_camera_mode(monkeypatch, tmp_path, capsys):
    captured = run_main(["--rehearsal"], monkeypatch, tmp_path)
    assert captured["rehearsal"] is True and captured["browser_camera"] is False
    assert "REHEARSAL" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--rehearsal", "--server-camera"], ["--rehearsal", "--browser-camera"],
    ["--browser-camera", "--server-camera"], ["--camera", "1"], ["--rehearsal", "--camera", "1"],
])
def test_conflicting_camera_options_fail_visibly(argv, monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(argv, monkeypatch, tmp_path)


def test_active_profile_is_shown_even_without_promoted_models(monkeypatch, tmp_path):
    from hackathon.profiles import ProfileManager
    manager = ProfileManager(root_dir=tmp_path)
    manager.create_profile("alice", "Alice", lip_prompts=["Tea please"])
    manager.set_active_profile("alice")
    captured = run_main(["--rehearsal"], monkeypatch, tmp_path)
    assert captured["profile"].id == "alice" and captured["profile_name"] == "Alice"
    assert captured["motion_model"] is None and captured["lip_model"] is None  # stock models
    state = captured["runtime"].snapshot()
    assert state["profile_id"] == "alice" and state["quick_phrases"] == ["Tea please"]
    assert "no trained motion model" in state["profile_warning"]
    json.dumps(state)

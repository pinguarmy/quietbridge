"""Server contract for setup downloads, speech settings, app info, shutdown and launch behaviour.

Fakes only: downloads use an injected fetcher (no network), speech uses fake
processes (nothing is audible), and no camera or model is loaded.
"""
import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lipflow import paths
from quietbridge import assets
from quietbridge import server as server_module
from quietbridge.assets import DownloadCancelled, SetupDownloads
from quietbridge.profiles import ProfileManager, load_speech_settings
from quietbridge.runtime import Runtime
from quietbridge.server import make_server
from quietbridge.speech import Speaker

SAY_VOICES = b"Alex                en_US    # Hello.\nSamantha            en_US    # Hi.\n"


class SilentSpeaker:
    """Like scripts/web_check_fixture.py: most Speaker attributes are missing."""
    available = True

    def speak(self, text):
        pass

    def close(self):
        pass


class FakeRun:
    def __init__(self, stdout):
        self.stdout = stdout

    def __call__(self, args, **kwargs):
        import subprocess
        return subprocess.CompletedProcess(args, 0, self.stdout, b"")


@pytest.fixture(autouse=True)
def close_download_workers(monkeypatch):
    """Join every download worker before the temporary models folder is undone."""
    created = []
    original = SetupDownloads.__init__

    def tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(SetupDownloads, "__init__", tracking)
    yield
    for manager in created:
        manager.close(timeout=10)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # Small stand-in sizes: extending a file to a gigabyte writes real zeros on Windows.
    import dataclasses
    monkeypatch.setattr(assets, "LIPS_ASSETS", tuple(
        dataclasses.replace(a, size=min(a.size, 4096), min_bytes=min(a.min_bytes, 512)) for a in assets.LIPS_ASSETS))
    monkeypatch.setenv("QUIETBRIDGE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(paths, "MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", str(tmp_path / "cache" / "gesture.task"))
    monkeypatch.delenv("QUIETBRIDGE_MOTION_MODEL", raising=False)
    monkeypatch.setenv("QUIETBRIDGE_NO_OPEN", "1")
    return tmp_path


def make_runtime(root, rehearsal=True, speaker=None, profile_id=None):
    manager = ProfileManager(root_dir=root / "data")
    profile = manager.get_profile(profile_id) if profile_id else manager.get_active_profile()
    kwargs = {"rehearsal": True} if rehearsal else {"browser_camera": True}
    runtime = Runtime(vocabulary={}, profile=profile, profile_manager=manager, **kwargs)
    if not rehearsal:
        runtime._start_workers = lambda: None  # never load models or open a camera here
    runtime.speaker.close()
    runtime.speaker = speaker if speaker is not None else SilentSpeaker()
    return runtime


@contextmanager
def serving(runtime, downloads=None):
    server = make_server(runtime, 0, downloads=downloads)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def call(server, method, path, body=None, token=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    sent = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}"}
    if token is not None:
        sent["X-Lipflow-Token"] = token
    sent.update(headers or {})
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=sent)
    response = conn.getresponse()
    data = json.loads(response.read() or b"null")
    conn.close()
    return response.status, data


def state(server):
    return call(server, "GET", "/api/state")[1]


def sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


class Fetcher:
    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.calls = []

    def __call__(self, asset, path=None, progress=None, cancel=None):
        self.calls.append(asset.name)
        while not self.gate.wait(0.01):
            if cancel.is_set():
                raise DownloadCancelled(asset.name)
        sparse(path or asset.path(), asset.size)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------ the state contract


def test_state_carries_setup_speech_settings_and_app(isolated):
    with serving(make_runtime(isolated)) as server:
        data = state(server)
    json.dumps(data)
    setup = data["setup"]
    assert setup["busy"] is False and setup["reload_needed"] is False
    assert [a["id"] for a in setup["assets"]] == ["lips", "gestures"]
    for group in setup["assets"]:
        assert set(group) == {"id", "label", "status", "bytes_done", "bytes_total", "error", "size_note",
                              "license_note"}
        assert group["status"] == "missing" and group["error"] == "" and group["bytes_total"] > 0
        assert isinstance(group["bytes_done"], int) and group["size_note"].endswith("one time")
    assert [a["label"] for a in setup["assets"]] == ["Lip reading", "Stock hand gestures"]
    assert "non-commercial research use only (see NOTICE)" in setup["assets"][0]["license_note"]
    assert data["speech_settings"] == {"rate": "normal", "voice": "", "voices": []}  # a fake speaker
    assert data["app"]["started_with"] == "rehearsal" and isinstance(data["app"]["version"], str)


def test_started_with_reflects_the_camera_source():
    assert server_module._started_with(Mock(rehearsal=False, browser_camera=True)) == "browser_camera"
    assert server_module._started_with(Mock(rehearsal=False, browser_camera=False)) == "server_camera"
    assert server_module._started_with(Mock()) == "server_camera"  # truthy Mocks are not True


# ------------------------------------------------------------------ setup endpoints


def test_download_requests_are_checked(isolated):
    fetcher = Fetcher()
    with serving(make_runtime(isolated, rehearsal=False), SetupDownloads(fetcher=fetcher)) as server:
        token = state(server)["token"]
        assert call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True})[0] == 403
        assert call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True}, token,
                    {"Origin": "http://evil.example"})[0] == 403
        assert call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True}, token,
                    {"Host": f"evil.example:{server.server_port}"})[0] == 403
        for body in ({"id": "lips"}, {"id": "lips", "accept_license": "true"}, {"id": "lips", "accept_license": 1}):
            status, data = call(server, "POST", "/api/setup/download", body, token)
            assert status == 400 and "accept the model's license" in data["error"]
        for bad in ("../../etc", "https://example.com/x.pth", None, ["lips"]):
            status, data = call(server, "POST", "/api/setup/download", {"id": bad, "accept_license": True}, token)
            assert status == 400 and data["error"] == "Unknown download; choose lips or gestures"
        status, data = call(server, "POST", "/api/setup/download",
                            {"id": "gestures", "accept_license": True, "url": "https://example.com/evil"}, token)
        assert status == 200
        assert wait_for(lambda: state(server)["setup"]["assets"][1]["status"] == "ready")
    assert fetcher.calls == ["gesture_recognizer.task"]  # only the fixed list


def test_rehearsal_reports_truthfully_but_refuses_downloads(isolated):
    sparse(Path(paths.MODELS) / "face_landmarker.task", assets.LIPS_ASSETS[-1].size)
    with serving(make_runtime(isolated), SetupDownloads(fetcher=Fetcher())) as server:
        data = state(server)
        lips = data["setup"]["assets"][0]
        assert lips["status"] == "missing" and lips["bytes_done"] == assets.LIPS_ASSETS[-1].size
        status, error = call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True},
                             data["token"])
    assert status == 400 and "rehearsal" in error["error"] and "without --rehearsal" in error["error"]


def test_download_ready_then_reload_needed_until_the_explicit_reload(isolated):
    fetcher = Fetcher()
    runtime = make_runtime(isolated, rehearsal=False)
    with serving(runtime, SetupDownloads(fetcher=fetcher)) as server:
        token = state(server)["token"]
        runtime.state["lips"] = "unavailable"  # started without the files
        status, data = call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True}, token)
        assert status == 200 and data["setup"]["assets"][0]["status"] in {"downloading", "ready"}
        # idempotent while running
        assert call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True}, token)[0] == 200
        assert wait_for(lambda: state(server)["setup"]["reload_needed"] is True)
        data = state(server)
        assert data["setup"]["assets"][0]["status"] == "ready" and data["setup"]["busy"] is False
        assert data["lips"] == "unavailable"  # nothing was loaded automatically
        assert fetcher.calls == [a.name for a in assets.LIPS_ASSETS]
        status, data = call(server, "POST", "/api/profile/select", {"id": data["profile_id"]}, token)
        assert status == 200 and data["setup"]["reload_needed"] is False
        assert data["lips"] == "warming"  # the explicit reload restarts the lip worker


def test_no_reload_needed_when_the_models_are_already_loaded(isolated):
    runtime = make_runtime(isolated, rehearsal=False)
    with serving(runtime, SetupDownloads(fetcher=Fetcher())) as server:
        token = state(server)["token"]
        runtime.state["signs"] = "ready"
        call(server, "POST", "/api/setup/download", {"id": "gestures", "accept_license": True}, token)
        assert wait_for(lambda: state(server)["setup"]["assets"][1]["status"] == "ready")
        assert state(server)["setup"]["reload_needed"] is False


def test_cancel_stops_the_download(isolated):
    fetcher = Fetcher()
    fetcher.gate.clear()
    with serving(make_runtime(isolated, rehearsal=False), SetupDownloads(fetcher=fetcher)) as server:
        token = state(server)["token"]
        call(server, "POST", "/api/setup/download", {"id": "lips", "accept_license": True}, token)
        assert state(server)["setup"]["busy"] is True
        status, data = call(server, "POST", "/api/setup/cancel", {"id": "lips"}, token)
        assert status == 200
        assert wait_for(lambda: state(server)["setup"]["busy"] is False)
        lips = state(server)["setup"]["assets"][0]
        assert lips["status"] == "missing" and lips["error"] == ""
        assert call(server, "POST", "/api/setup/cancel", {"id": "x"}, token)[0] == 400


def test_server_close_stops_the_download_worker(isolated):
    downloads = Mock(busy=False)
    downloads._on_ready = None
    downloads.group_state.side_effect = lambda gid: {"id": gid}
    server = make_server(make_runtime(isolated), 0, downloads=downloads)
    server.server_close()
    downloads.close.assert_called_once()


def test_a_broken_model_path_does_not_break_polling(isolated):
    downloads = SetupDownloads(fetcher=Fetcher())
    downloads.group_state = Mock(side_effect=OSError("disk gone"))
    with serving(make_runtime(isolated), downloads) as server:
        status, data = call(server, "GET", "/api/state")
    assert status == 200 and data["setup"]["assets"][0]["status"] == "error"


# ------------------------------------------------------------------ speech settings


def test_speech_settings_are_validated_saved_per_profile_and_reapplied(isolated):
    run = FakeRun(SAY_VOICES)
    speaker = Speaker(platform="darwin", which=lambda name: "/usr/bin/say" if name == "say" else None, run=run)
    manager = ProfileManager(root_dir=isolated / "data")
    manager.ensure_default_profile()
    manager.create_profile("bob", "Bob")
    runtime = make_runtime(isolated, speaker=speaker)
    with serving(runtime) as server:
        token = state(server)["token"]
        assert wait_for(lambda: len(state(server)["speech_settings"]["voices"]) == 2)
        status, data = call(server, "POST", "/api/speech/settings", {"rate": "slow", "voice": "Samantha"}, token)
        assert status == 200
        assert data["speech_settings"] == {"rate": "slow", "voice": "Samantha", "voices": [
            {"id": "Alex", "label": "Alex (en_US)"}, {"id": "Samantha", "label": "Samantha (en_US)"}]}
        assert speaker.settings() == {"rate": "slow", "voice": "Samantha"}
        assert load_speech_settings(runtime.profile) == {"rate": "slow", "voice": "Samantha"}
        for body, message in (({"rate": "turbo"}, "slow, normal or fast"),
                              ({"voice": "Nobody"}, "not installed"),
                              ({"voice": "--output-file=/tmp/x"}, "not installed"),
                              ({"voice": 3}, "Voice must be text"),
                              ({"volume": 9}, "Unknown speech setting")):
            status, data = call(server, "POST", "/api/speech/settings", body, token)
            assert status == 400 and message in data["error"]
        assert call(server, "POST", "/api/speech/settings", {"rate": "fast"})[0] == 403  # token
        # Another profile has its own settings (defaults here) ...
        status, data = call(server, "POST", "/api/profile/select", {"id": "bob"}, token)
        assert data["speech_settings"]["rate"] == "normal" and speaker.settings()["rate"] == "normal"
        # ... and switching back restores the saved ones.
        status, data = call(server, "POST", "/api/profile/select", {"id": "default"}, token)
        assert data["speech_settings"]["rate"] == "slow" and data["speech_settings"]["voice"] == "Samantha"
        assert speaker.settings() == {"rate": "slow", "voice": "Samantha"}


def test_a_fake_speaker_still_validates_and_saves(isolated):
    runtime = make_runtime(isolated)
    with serving(runtime) as server:
        token = state(server)["token"]
        status, data = call(server, "POST", "/api/speech/settings", {"rate": "fast"}, token)
        assert status == 200 and data["speech_settings"] == {"rate": "fast", "voice": "", "voices": []}
        assert call(server, "POST", "/api/speech/settings", {"voice": "Alex"}, token)[0] == 400
    assert load_speech_settings(runtime.profile)["rate"] == "fast"


def test_existing_profiles_load_unchanged_and_bad_files_mean_defaults(isolated):
    manager = ProfileManager(root_dir=isolated / "data")
    profile = manager.ensure_default_profile()
    before = profile.config_path.read_text()
    assert load_speech_settings(profile) == {"rate": "normal", "voice": ""}  # no speech.json yet
    for content in ("{not json", "[]", '{"rate": "warp", "voice": 7}'):
        (profile.profile_dir / "speech.json").write_text(content)
        assert load_speech_settings(profile) == {"rate": "normal", "voice": ""}
    assert profile.config_path.read_text() == before
    assert manager.get_profile("default").to_dict() == profile.to_dict()


# ------------------------------------------------------------------ shutdown and launch


def run_main_in_thread(argv, isolated, servers):
    real_make_server = server_module.make_server

    def fake_runtime(**kwargs):
        runtime = Runtime(rehearsal=True, vocabulary=kwargs.get("vocabulary"), profile=kwargs.get("profile"),
                          profile_manager=kwargs.get("profile_manager"))
        runtime.close = Mock(wraps=runtime.close)
        servers.append(runtime)
        return runtime

    def fake_make_server(runtime, port):
        server = real_make_server(runtime, port)
        servers.append(server)
        return server

    errors = []

    def target():
        try:
            with patch("sys.argv", ["quietbridge.server", *argv]), \
                    patch.object(server_module, "Runtime", fake_runtime), \
                    patch.object(server_module, "make_server", fake_make_server):
                server_module.main()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            errors.append(exc)
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, errors


def test_shutdown_ends_the_process_main_loop(isolated, capsys):
    objects = []
    thread, errors = run_main_in_thread(["--rehearsal", "--port", "0"], isolated, objects)
    assert wait_for(lambda: len(objects) == 2)
    runtime, server = objects
    data = state(server)
    status, reply = call(server, "POST", "/api/shutdown", {}, data["token"])
    assert status == 200 and reply == {"ok": True}
    thread.join(timeout=10)
    assert not thread.is_alive() and errors == []
    runtime.close.assert_called_once()
    assert server.closing and "QuietBridge stopped from the page." in capsys.readouterr().out


def test_shutdown_needs_the_token(isolated):
    with serving(make_runtime(isolated)) as server:
        assert call(server, "POST", "/api/shutdown", {})[0] == 403
        assert call(server, "POST", "/api/shutdown", {}, state(server)["token"],
                    {"Origin": "http://evil.example"})[0] == 403
        assert not server.shutdown_requested


def test_already_running_quietbridge_is_reused(isolated, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(server_module, "open_in_browser",
                        lambda url, opener=None: opened.append(url) or Mock(join=lambda timeout=None: None))
    monkeypatch.setattr(server_module, "should_open_browser", lambda no_open=False: not no_open)
    with serving(make_runtime(isolated)) as running:
        port = running.server_port
        with patch("sys.argv", ["quietbridge.server", "--rehearsal", "--port", str(port)]):
            server_module.main()  # returns: exit status 0
    out = capsys.readouterr()
    assert f"QuietBridge is already running at http://127.0.0.1:{port}" in out.out
    assert opened == [f"http://127.0.0.1:{port}"] and "Traceback" not in out.err


def test_another_program_on_the_port_is_named_without_a_traceback(isolated, capsys):
    other = socket.socket()
    other.bind(("127.0.0.1", 0))
    other.listen(1)
    port = other.getsockname()[1]
    stop = threading.Event()

    def answer():
        while not stop.is_set():
            try:
                conn, _ = other.accept()
            except OSError:
                return
            with conn:
                conn.recv(4096)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello")
    threading.Thread(target=answer, daemon=True).start()
    try:
        with patch("sys.argv", ["quietbridge.server", "--rehearsal", "--port", str(port)]):
            with pytest.raises(SystemExit) as excinfo:
                server_module.main()
    finally:
        stop.set()
        other.close()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"could not use port {port}" in err and "--port" in err and "Traceback" not in err


@pytest.mark.parametrize("no_open, environ, platform, expected", [
    (False, {}, "darwin", True),
    (False, {}, "win32", True),
    (False, {"DISPLAY": ":0"}, "linux", True),
    (False, {"WAYLAND_DISPLAY": "wayland-0"}, "linux", True),
    (False, {}, "linux", False),  # SSH / headless: no text browser in the terminal
    (True, {"DISPLAY": ":0"}, "linux", False),
    (False, {"DISPLAY": ":0", "QUIETBRIDGE_NO_OPEN": "1"}, "linux", False),
    (False, {"QUIETBRIDGE_NO_OPEN": "yes"}, "win32", False),
    (False, {"QUIETBRIDGE_NO_OPEN": ""}, "darwin", True),
])
def test_open_rule(no_open, environ, platform, expected):
    assert server_module.should_open_browser(no_open, environ, platform) is expected


def test_banner_says_what_to_download(isolated, capsys, monkeypatch):
    runtime = Mock(rehearsal=False, enable_lips=True, isolated_sign_model_path=None)
    line = server_module._missing_downloads_line(runtime)
    assert line.startswith("Not downloaded yet: lip reading and hand gestures.") and "Setup" in line
    sparse(Path(paths.MODELS) / "x", 1)
    for a in assets.LIPS_ASSETS:
        sparse(a.path(), a.size)
    assert server_module._missing_downloads_line(runtime).startswith("Not downloaded yet: hand gestures.")
    assert server_module._missing_downloads_line(Mock(rehearsal=True)) == ""

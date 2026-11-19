"""Contract tests for local speech backends using fake processes (no real audio)."""
import base64
import subprocess
import threading

import pytest

from quietbridge import speech
from quietbridge.speech import Speaker


class FakeStdin:
    def __init__(self, broken=False):
        self.data = b""
        self.closed = False
        self.broken = broken

    def write(self, data):
        if self.broken:
            raise BrokenPipeError(32, "Broken pipe")
        self.data += data

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, args, broken=False, ignore_terminate=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stdin = FakeStdin(broken)
        self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.ignore_terminate = ignore_terminate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def finish(self):
        self.returncode = 0


class FakePopen:
    def __init__(self, broken=False, ignore_terminate=False, error=None):
        self.processes = []
        self.broken = broken
        self.ignore_terminate = ignore_terminate
        self.error = error

    def __call__(self, args, **kwargs):
        if self.error:
            raise self.error
        process = FakeProcess(list(args), broken=self.broken, ignore_terminate=self.ignore_terminate, **kwargs)
        self.processes.append(process)
        return process


def which_from(*names):
    return lambda name: f"/bin/{name}" if name in names else None


@pytest.fixture
def popen(monkeypatch):
    fake = FakePopen()
    monkeypatch.setattr(speech.subprocess, "Popen", fake)
    return fake


@pytest.mark.parametrize("platform, installed, backend, executable", [
    ("darwin", ["say"], "say", "say"),
    ("win32", ["powershell.exe", "pwsh"], "powershell", "powershell.exe"),
    ("win32", ["pwsh"], "powershell", "pwsh"),
    ("linux", ["espeak-ng", "espeak", "spd-say"], "espeak-ng", "espeak-ng"),
    ("linux", ["espeak", "spd-say"], "espeak", "espeak"),
    ("linux", ["spd-say"], "spd-say", "spd-say"),
])
def test_backend_selection_per_platform(platform, installed, backend, executable):
    speaker = Speaker(platform=platform, which=which_from(*installed))
    assert speaker.available is True
    assert speaker.backend == backend
    assert speaker._resolve()[1] == f"/bin/{executable}"


@pytest.mark.parametrize("platform, hint", [
    ("darwin", "say"), ("win32", "PowerShell"), ("linux", "espeak-ng"),
])
def test_unavailable_backend_names_what_to_install(platform, hint, popen):
    speaker = Speaker(platform=platform, which=which_from())
    assert speaker.available is False
    assert speaker.backend is None
    with pytest.raises(RuntimeError, match=hint):
        speaker.speak("hello")
    assert popen.processes == []


def test_platform_does_not_borrow_other_platform_backends():
    assert Speaker(platform="win32", which=which_from("say", "espeak-ng")).available is False
    assert Speaker(platform="darwin", which=which_from("espeak-ng", "powershell.exe")).available is False


def test_detection_is_cached_and_refreshable():
    calls = []
    installed = set()

    def which(name):
        calls.append(name)
        return f"/bin/{name}" if name in installed else None

    speaker = Speaker(platform="linux", which=which)
    assert speaker.available is False
    count = len(calls)
    installed.add("espeak-ng")
    assert speaker.available is False and len(calls) == count
    speaker.refresh()
    assert speaker.backend == "espeak-ng"


def test_defaults_follow_monkeypatched_shutil_and_sys(monkeypatch):
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", which_from("espeak"))
    assert Speaker().backend == "espeak"


SECRET = "--help; rm -rf ~ $(whoami) `id` | ünïcødé “quotes”"


@pytest.mark.parametrize("platform, installed", [
    ("darwin", ["say"]), ("win32", ["powershell.exe"]), ("linux", ["espeak-ng"]),
    ("linux", ["espeak"]), ("linux", ["spd-say"]),
])
def test_text_goes_through_stdin_never_argv(platform, installed, popen):
    speaker = Speaker(platform=platform, which=which_from(*installed))
    speaker.speak(SECRET)
    process = popen.processes[-1]
    assert process.kwargs["stdin"] is subprocess.PIPE
    assert "shell" not in process.kwargs
    assert isinstance(process.args, list)
    for arg in process.args:
        assert "rm -rf" not in arg and "whoami" not in arg and "ünïcødé" not in arg
    assert process.stdin.data.decode("utf-8").strip() == SECRET
    assert process.stdin.closed


def test_backend_commands():
    assert speech._command("say", "/usr/bin/say") == ["/usr/bin/say"]
    assert speech._command("espeak-ng", "/usr/bin/espeak-ng") == ["/usr/bin/espeak-ng", "--stdin"]
    assert speech._command("espeak", "/usr/bin/espeak") == ["/usr/bin/espeak", "--stdin"]
    assert speech._command("spd-say", "/usr/bin/spd-say") == ["/usr/bin/spd-say", "-e", "-w"]
    ps = speech._command("powershell", "C:/powershell.exe")
    assert ps[:5] == ["C:/powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    script = base64.b64decode(ps[5]).decode("utf-16-le")
    assert script == speech._POWERSHELL_SCRIPT
    assert "InputEncoding" in script and "UTF8Encoding" in script
    assert "System.Speech.Synthesis.SpeechSynthesizer" in script
    assert "OpenStandardInput" in script


def test_spd_say_payload_cannot_inject_ssip_commands(popen):
    speaker = Speaker(platform="linux", which=which_from("spd-say"))
    speaker.speak("!-!SET SELF RATE 100\n!-!QUIT\nhello")
    process = popen.processes[-1]
    data = process.stdin.data.decode("utf-8")
    assert data.count("\n") == 1 and data.endswith("\n")
    assert not data.startswith("!")
    assert process.kwargs["stdout"] is subprocess.DEVNULL


def test_spd_say_payload_fits_its_line_buffer():
    assert len(speech._payload("spd-say", "€" * 2000)) < 16384


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "x" * 2001])
def test_validation_errors(text, popen):
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    with pytest.raises(ValueError, match="1–2000"):
        speaker.speak(text)
    assert popen.processes == []


def test_validation_precedes_availability(popen):
    with pytest.raises(ValueError):
        Speaker(platform="linux", which=which_from()).speak(" ")


def test_max_length_accepted(popen):
    Speaker(platform="linux", which=which_from("espeak-ng")).speak("x" * 2000)
    assert len(popen.processes) == 1


def test_speaking_property_tracks_process(popen):
    speaker = Speaker(platform="darwin", which=which_from("say"))
    assert speaker.speaking is False
    speaker.speak("hello")
    assert speaker.speaking is True
    popen.processes[-1].finish()
    assert speaker.speaking is False


def test_speaking_derives_from_assigned_process():
    from unittest.mock import Mock
    speaker = Speaker(platform="linux", which=which_from())
    speaker._process = Mock(poll=Mock(return_value=None))
    assert speaker.speaking is True
    speaker._process = Mock(poll=Mock(return_value=0))
    assert speaker.speaking is False


def test_speak_replaces_running_utterance(popen):
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    speaker.speak("first")
    first = popen.processes[-1]
    speaker.speak("second")
    second = popen.processes[-1]
    assert first is not second
    assert first.terminated == 1 and first.poll() is not None
    assert speaker._process is second and speaker.speaking


def test_stop_is_idempotent(popen):
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    speaker.stop()  # nothing running
    speaker.speak("hello")
    process = popen.processes[-1]
    speaker.stop()
    speaker.stop()
    speaker.close()
    assert process.terminated == 1
    assert speaker.speaking is False and speaker._process is None


def test_stop_does_not_terminate_finished_process(popen):
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    speaker.speak("hello")
    process = popen.processes[-1]
    process.finish()
    speaker.stop()
    assert process.terminated == 0


def test_stop_kills_process_that_ignores_terminate(monkeypatch):
    fake = FakePopen(ignore_terminate=True)
    monkeypatch.setattr(speech.subprocess, "Popen", fake)
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    speaker.speak("hello")
    speaker.stop()
    assert fake.processes[-1].killed == 1
    assert speaker.speaking is False


def test_stop_is_thread_safe(popen):
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    speaker.speak("hello")
    threads = [threading.Thread(target=speaker.stop) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert popen.processes[-1].terminated == 1
    assert speaker._process is None


def test_broken_pipe_raises_runtime_error_and_cleans_up(monkeypatch):
    fake = FakePopen(broken=True)
    monkeypatch.setattr(speech.subprocess, "Popen", fake)
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    with pytest.raises(RuntimeError, match="could not start"):
        speaker.speak("hello")
    assert fake.processes[-1].terminated == 1
    assert speaker.speaking is False


def test_spawn_failure_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(speech.subprocess, "Popen", FakePopen(error=FileNotFoundError("gone")))
    speaker = Speaker(platform="linux", which=which_from("espeak-ng"))
    with pytest.raises(RuntimeError, match="could not start"):
        speaker.speak("hello")
    assert speaker._process is None


def test_no_window_flag_only_on_real_windows(monkeypatch, popen):
    monkeypatch.setattr(speech.sys, "platform", "linux")
    Speaker(platform="win32", which=which_from("powershell.exe")).speak("hello")
    assert "creationflags" not in popen.processes[-1].kwargs
    monkeypatch.setattr(speech.sys, "platform", "win32")
    monkeypatch.setattr(speech.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    Speaker(platform="win32", which=which_from("powershell.exe")).speak("hello")
    assert popen.processes[-1].kwargs["creationflags"] == 0x08000000

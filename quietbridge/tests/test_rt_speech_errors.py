"""Speech failure reporting with real (deliberately failing) child processes.

These prove the reporting contract only. No audio device exists in the test
environment, so nothing here shows that speech is audible.
"""
import os
import stat
import sys
import time

import pytest

from quietbridge import speech
from quietbridge.speech import Speaker

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell scripts as fake voices")


def fake_voice(tmp_path, name, body):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def speaker_for(path, name="espeak-ng", platform="linux"):
    return Speaker(platform=platform, which=lambda candidate: path if candidate == name else None)


def wait_finished(speaker, timeout=5):
    deadline = time.monotonic() + timeout
    while speaker.speaking and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not speaker.speaking


def test_nonzero_exit_after_start_is_reported_with_stderr(tmp_path):
    path = fake_voice(tmp_path, "espeak-ng",
                      "cat >/dev/null; echo 'ALSA lib pcm.c: Unknown PCM default' >&2; exit 1")
    speaker = speaker_for(path)
    speaker.speak("hello")  # starting succeeds: the failure happens afterwards
    wait_finished(speaker)
    error = speaker.last_error
    assert error.startswith("Speech did not play")
    assert "exited with status 1" in error and "Unknown PCM default" in error
    assert "speaker or headphones" in error
    assert speaker.last_error == error  # stable on repeated polls
    speaker.close()


def test_clean_exit_reports_no_error_and_claims_nothing(tmp_path):
    speaker = speaker_for(fake_voice(tmp_path, "espeak-ng", "cat >/dev/null; exit 0"))
    speaker.speak("hello")
    wait_finished(speaker)
    assert speaker.last_error == ""
    speaker.close()


def test_error_output_is_bounded(tmp_path):
    body = "cat >/dev/null; i=0; while [ $i -lt 2000 ]; do echo \"noise line $i\" >&2; i=$((i+1)); done; exit 2"
    speaker = speaker_for(fake_voice(tmp_path, "espeak-ng", body))
    speaker.speak("hello")
    wait_finished(speaker)
    error = speaker.last_error
    assert "status 2" in error and "noise line 1999" in error
    assert len(error) < 600
    speaker.close()


def test_running_and_deliberately_stopped_speech_is_not_an_error(tmp_path):
    speaker = speaker_for(fake_voice(tmp_path, "espeak-ng", "cat >/dev/null; sleep 5; exit 1"))
    speaker.speak("hello")
    assert speaker.speaking and speaker.last_error == ""
    speaker.stop()
    assert not speaker.speaking and speaker.last_error == ""
    speaker.close()


def test_new_successful_utterance_and_clear_error_dismiss_a_failure(tmp_path):
    failing = fake_voice(tmp_path, "espeak-ng", "cat >/dev/null; exit 1")
    speaker = speaker_for(failing)
    speaker.speak("first")
    wait_finished(speaker)
    assert speaker.last_error
    speaker.clear_error()
    assert speaker.last_error == ""

    speaker.speak("again")
    wait_finished(speaker)
    assert speaker.last_error
    working = fake_voice(tmp_path, "ok-voice", "cat >/dev/null; sleep 5")
    speaker._detected = ("espeak-ng", working)
    speaker.speak("third")
    assert speaker.last_error == ""  # cleared as soon as the new utterance started
    speaker.close()


def test_error_file_is_closed_and_text_stays_on_stdin(tmp_path):
    capture = tmp_path / "argv.txt"
    path = fake_voice(tmp_path, "espeak-ng", f'echo "$@" > "{capture}"; cat >/dev/null; exit 1')
    speaker = speaker_for(path)
    speaker.speak("--secret text")
    wait_finished(speaker)
    assert speaker.last_error
    assert "secret" not in capture.read_text()
    utterance = speaker._utterance
    assert utterance.stderr_file is None  # read once, then closed
    speaker.speak("--next")
    previous = utterance
    speaker.close()
    assert previous.stderr_file is None and speaker._utterance.stderr_file is None


def test_pwsh_without_system_speech_names_windows_powershell_51(tmp_path):
    path = fake_voice(tmp_path, "pwsh",
                      "cat >/dev/null; echo 'QUIETBRIDGE_NO_SYSTEM_SPEECH: System.Speech cannot be loaded' >&2; exit 3")
    speaker = speaker_for(path, name="pwsh", platform="win32")
    assert speaker.backend == "powershell"
    speaker.speak("hello")
    wait_finished(speaker)
    error = speaker.last_error
    assert "Windows PowerShell 5.1" in error and "powershell.exe" in error
    assert "System.Speech" in error
    speaker.close()


def test_powershell_script_detects_missing_system_speech():
    script = speech._POWERSHELL_SCRIPT
    assert "catch" in script and "exit 3" in script and speech._NO_SYSTEM_SPEECH_MARKER in script
    # The message is read from stdin before anything can fail, never interpolated.
    assert script.index("ReadToEnd") < script.index("Add-Type")


def test_windows_powershell_found_at_its_fixed_path_when_not_on_path(monkeypatch):
    monkeypatch.setenv("SystemRoot", "D:\\Win")
    fixed = "D:\\Win\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
    installed = {fixed: fixed, "pwsh": "/bin/pwsh"}
    speaker = Speaker(platform="win32", which=installed.get)
    assert speaker._resolve() == ("powershell", fixed)  # preferred over pwsh


def test_unavailable_windows_hint_names_windows_powershell_51():
    speaker = Speaker(platform="win32", which=lambda name: None)
    with pytest.raises(RuntimeError, match="Windows PowerShell 5.1"):
        speaker.speak("hello")


def test_signal_exit_is_described(tmp_path):
    speaker = speaker_for(fake_voice(tmp_path, "espeak-ng", "cat >/dev/null; kill -9 $$"))
    speaker.speak("hello")
    wait_finished(speaker)
    assert "stopped by signal 9" in speaker.last_error
    speaker.close()


def test_no_descriptor_leak_over_many_failures(tmp_path):
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("needs /proc")
    speaker = speaker_for(fake_voice(tmp_path, "espeak-ng", "cat >/dev/null; echo bad >&2; exit 1"))
    speaker.speak("warm up")
    wait_finished(speaker)
    assert speaker.last_error
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        speaker.speak("hello")
        wait_finished(speaker)
        assert speaker.last_error
    speaker.close()
    assert len(os.listdir("/proc/self/fd")) <= before

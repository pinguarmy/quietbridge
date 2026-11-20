"""Speaking rate and voice per backend, with fake processes only.

Nothing here is audible, and no real say / espeak-ng / spd-say / PowerShell runs:
the tests pin the exact argv and environment each backend receives, and that
neither the message nor a voice id is ever interpreted by a shell.
"""
import base64
import subprocess
import time

import pytest

from quietbridge import speech
from quietbridge.speech import Speaker

SAY_VOICES = b"""\
Alex                en_US    # Most people recognize me by my voice.
Amelie              fr_CA    # Bonjour, je m'appelle Amelie.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Bad News            en_US    # The light you see at the end of the tunnel is the headlamp.
$(touch pwned)      en_US    # a hostile name is still only one argv element
-v                  en_US    # never accepted: looks like an option
"""
ESPEAK_VOICES = b"""\
Pty Language       Age/Gender VoiceName          File                 Other Languages
 5  af              --/M      Afrikaans          gmw/af
 2  en-gb           --/M      English_(Great_Britain) gmw/en             (en 2)
 2  en-us           --/M      English_(America)  gmw/en-US            (en 3)
 5  fr-fr           --/M      French_(France)    roa/fr               (fr 5)
"""
POWERSHELL_VOICES = "Microsoft David Desktop\ten-US\r\nMicrosoft Zira Desktop\ten-US\r\n".encode()


class FakeStdin:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    def close(self):
        pass


class FakeProcess:
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stdin = FakeStdin()
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakePopen(list):
    def __call__(self, args, **kwargs):
        assert not kwargs.get("shell"), "speech must never use a shell"
        assert isinstance(args, list)
        process = FakeProcess(list(args), **kwargs)
        self.append(process)
        return process


class FakeRun:
    def __init__(self, stdout=b"", returncode=0, error=None):
        self.stdout, self.returncode, self.error = stdout, returncode, error
        self.calls = []

    def __call__(self, args, **kwargs):
        assert not kwargs.get("shell") and kwargs.get("timeout")
        self.calls.append(list(args))
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, b"")


@pytest.fixture
def popen(monkeypatch):
    fake = FakePopen()
    monkeypatch.setattr(speech.subprocess, "Popen", fake)
    return fake


def which(*names):
    return lambda name: f"/bin/{name}" if name in names else None


@pytest.mark.parametrize("backend, rate, voice, expected", [
    ("say", "normal", "", ["/bin/say"]),
    ("say", "slow", "", ["/bin/say", "-r", "140"]),
    ("say", "fast", "Eddy (English (US))", ["/bin/say", "-r", "240", "-v", "Eddy (English (US))"]),
    ("espeak-ng", "normal", "", ["/bin/espeak-ng", "--stdin"]),
    ("espeak-ng", "slow", "en-us", ["/bin/espeak-ng", "-s", "130", "-v", "en-us", "--stdin"]),
    ("espeak", "fast", "", ["/bin/espeak", "-s", "220", "--stdin"]),
    ("spd-say", "normal", "", ["/bin/spd-say", "-e", "-w"]),
    ("spd-say", "slow", "", ["/bin/spd-say", "-r", "-40", "-e", "-w"]),
    ("spd-say", "fast", "", ["/bin/spd-say", "-r", "40", "-e", "-w"]),
])
def test_exact_argv_per_backend(backend, rate, voice, expected):
    assert speech._command(backend, f"/bin/{backend}", rate, voice) == expected


def test_powershell_argv_is_fixed_and_settings_travel_as_environment_data(monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_SPEECH_VOICE", "inherited must not leak")
    command = speech._command("powershell", "C:/ps.exe", "slow", "")
    assert command == ["C:/ps.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                       speech._POWERSHELL_ENCODED]
    script = base64.b64decode(command[5]).decode("utf-16-le")
    assert "$env:QUIETBRIDGE_SPEECH_RATE" in script and "$env:QUIETBRIDGE_SPEECH_VOICE" in script
    assert "SelectVoice($voice)" in script and "OpenStandardInput" in script
    env = speech._environment("powershell", "slow", "Microsoft Zira Desktop")
    assert env["QUIETBRIDGE_SPEECH_RATE"] == "-3" and env["QUIETBRIDGE_SPEECH_VOICE"] == "Microsoft Zira Desktop"
    default = speech._environment("powershell", "normal", "")
    assert "QUIETBRIDGE_SPEECH_RATE" not in default and "QUIETBRIDGE_SPEECH_VOICE" not in default
    assert speech._environment("say", "fast", "Alex") is None  # other backends inherit unchanged
    assert speech._command("powershell", "C:/ps.exe", "fast", "Any voice") == command  # never in argv


@pytest.mark.parametrize("backend, voice", [
    ("say", "-o/tmp/x"), ("espeak-ng", "--stdout"), ("say", "a\nb"), ("say", " Alex"), ("say", "x" * 200),
    ("spd-say", "anything"), ("powershell", "-x\nexit"),
])
def test_unsafe_or_unsupported_voice_never_reaches_argv(backend, voice):
    with pytest.raises(ValueError):
        speech._command(backend, f"/bin/{backend}", "normal", voice)


def test_voice_lists_are_parsed_per_backend():
    say = speech._parse_voices("say", SAY_VOICES.decode())
    assert [v["id"] for v in say] == ["Alex", "Amelie", "Eddy (English (US))", "Bad News", "$(touch pwned)"]
    assert say[2]["label"] == "Eddy (English (US)) (en_US)"
    espeak = speech._parse_voices("espeak-ng", ESPEAK_VOICES.decode())
    assert [v["id"] for v in espeak] == ["af", "en-gb", "en-us", "fr-fr"]
    assert espeak[2]["label"] == "English (America) (en-us)"
    ps = speech._parse_voices("powershell", POWERSHELL_VOICES.decode())
    assert ps == [{"id": "Microsoft David Desktop", "label": "Microsoft David Desktop (en-US)"},
                  {"id": "Microsoft Zira Desktop", "label": "Microsoft Zira Desktop (en-US)"}]


@pytest.mark.parametrize("platform, installed, argv", [
    ("darwin", ["say"], ["/bin/say", "-v", "?"]),
    ("linux", ["espeak-ng"], ["/bin/espeak-ng", "--voices"]),
    ("linux", ["espeak"], ["/bin/espeak", "--voices"]),
])
def test_voice_listing_commands(platform, installed, argv):
    run = FakeRun(SAY_VOICES)
    Speaker(platform=platform, which=which(*installed), run=run).voices(wait=True)
    assert run.calls == [argv]


def test_powershell_voice_listing_is_a_fixed_encoded_script():
    run = FakeRun(POWERSHELL_VOICES)
    voices = Speaker(platform="win32", which=which("powershell.exe"), run=run).voices(wait=True)
    assert voices[1]["id"] == "Microsoft Zira Desktop"
    script = base64.b64decode(run.calls[0][5]).decode("utf-16-le")
    assert "GetInstalledVoices()" in script and run.calls[0][4] == "-EncodedCommand"


def test_spd_say_lists_no_voices_and_runs_nothing():
    run = FakeRun(b"NAME LANGUAGE VARIANT\n")
    speaker = Speaker(platform="linux", which=which("spd-say"), run=run)
    assert speaker.voices(wait=True) == [] and run.calls == []


@pytest.mark.parametrize("run", [
    FakeRun(error=subprocess.TimeoutExpired(["say"], 5)),
    FakeRun(error=FileNotFoundError("say")),
    FakeRun(b"garbage\x00\xff", returncode=0),
    FakeRun(SAY_VOICES, returncode=1),
])
def test_listing_failures_yield_an_empty_list(run):
    assert Speaker(platform="darwin", which=which("say"), run=run).voices(wait=True) == []


def test_no_backend_lists_nothing():
    assert Speaker(platform="linux", which=which()).voices(wait=True) == []


def test_voice_listing_is_cached_and_never_blocks_a_poll():
    class SlowRun(FakeRun):
        def __call__(self, args, **kwargs):
            time.sleep(0.3)
            return super().__call__(args, **kwargs)
    run = SlowRun(SAY_VOICES)
    speaker = Speaker(platform="darwin", which=which("say"), run=run)
    started = time.monotonic()
    assert speaker.voices() == []  # still listing in the background
    assert time.monotonic() - started < 0.2
    assert len(speaker.voices(wait=True)) == 5
    speaker.voices()
    speaker.voices(wait=True)
    assert len(run.calls) == 1


def test_rate_and_strictly_listed_voice_apply_to_the_next_utterance(popen):
    speaker = Speaker(platform="darwin", which=which("say"), run=FakeRun(SAY_VOICES))
    speaker.speak("first")
    speaker.set_rate("slow")
    speaker.set_voice("$(touch pwned)")  # listed by the system: passed as one argv element, no shell
    speaker.speak("; rm -rf ~ `id`")
    assert popen[0].args == ["/bin/say"]
    assert popen[1].args == ["/bin/say", "-r", "140", "-v", "$(touch pwned)"]
    assert popen[1].stdin.data == "; rm -rf ~ `id`".encode()  # text only on stdin
    assert speaker.settings() == {"rate": "slow", "voice": "$(touch pwned)"}


def test_unlisted_voice_is_refused_and_a_stale_saved_voice_falls_back(popen):
    speaker = Speaker(platform="darwin", which=which("say"), run=FakeRun(SAY_VOICES))
    for bad in ("Nobody", "-v", "--help", "Alex\n"):
        with pytest.raises(ValueError, match="not installed"):
            speaker.set_voice(bad)
    with pytest.raises(ValueError):
        speaker.set_rate("turbo")
    speaker.set_voice("Removed Voice", strict=False)  # a saved preference
    speaker.speak("hello")
    assert popen[-1].args == ["/bin/say"]  # the system default, never an unlisted id
    assert speaker.settings()["voice"] == ""


def test_powershell_receives_the_voice_in_its_environment_only(popen):
    speaker = Speaker(platform="win32", which=which("powershell.exe"), run=FakeRun(POWERSHELL_VOICES))
    speaker.set_rate("fast")
    speaker.set_voice("Microsoft Zira Desktop")
    speaker.speak("hello")
    process = popen[-1]
    assert "Microsoft Zira Desktop" not in " ".join(process.args)
    assert process.kwargs["env"]["QUIETBRIDGE_SPEECH_VOICE"] == "Microsoft Zira Desktop"
    assert process.kwargs["env"]["QUIETBRIDGE_SPEECH_RATE"] == "3"


def test_unusable_powershell_voice_has_an_actionable_message():
    message = speech._failure_message("powershell", 4, "QUIETBRIDGE_VOICE_UNAVAILABLE: x")
    assert "Choose another voice" in message

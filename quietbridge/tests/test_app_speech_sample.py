"""Settings "Test voice": an explicit sample that never touches the message.

Fake speakers over loopback HTTP: these check the software contract only.
Nothing here is audible.
"""
from __future__ import annotations

import pytest

from quietbridge.tests.test_app_server import call, isolated, make_runtime, serving, state  # noqa: F401


class RecordingSpeaker:
    available = True
    backend = "fake"

    def __init__(self):
        self.spoken = []
        self.speaking = False
        self.last_error = ""

    def speak(self, text):
        self.spoken.append(text)

    def stop(self):
        self.speaking = False

    def close(self):
        pass


def test_sample_speaks_a_fixed_sentence_and_leaves_the_message_alone(isolated):
    speaker = RecordingSpeaker()
    runtime = make_runtime(isolated, speaker=speaker)
    with serving(runtime) as server:
        token = state(server)["token"]
        status, applied = call(server, "POST", "/api/action/correct",
                               {"revision": state(server)["revision"], "text": "Please wait a moment.", "learn": False}, token)
        assert status == 200
        before = state(server)
        status, after = call(server, "POST", "/api/speech/test", {}, token)
        assert status == 200
        assert speaker.spoken == [runtime.SPEECH_SAMPLE]
        for key in ("revision", "decision", "alternatives", "last_spoken"):
            assert after[key] == before[key], key
        assert after["last_spoken"] == ""


def test_sample_needs_the_token_and_is_refused_while_speech_is_busy(isolated):
    speaker = RecordingSpeaker()
    runtime = make_runtime(isolated, speaker=speaker)
    with serving(runtime) as server:
        assert call(server, "POST", "/api/speech/test", {}, "wrong")[0] == 403
        speaker.speaking = True
        status, body = call(server, "POST", "/api/speech/test", {}, state(server)["token"])
        assert status == 400 and "already playing" in body["error"]
        assert speaker.spoken == []


def test_a_failed_sample_start_reports_an_error_and_releases_the_voice(isolated):
    class FailingSpeaker(RecordingSpeaker):
        def speak(self, text):
            raise RuntimeError("Native speech could not start (fake)")

    runtime = make_runtime(isolated, speaker=FailingSpeaker())
    with serving(runtime) as server:
        token = state(server)["token"]
        status, body = call(server, "POST", "/api/speech/test", {}, token)
        assert status == 400 and "could not start" in body["error"]
        # The dispatch flag was released: a later request is not stuck on "already playing".
        status, body = call(server, "POST", "/api/speech/test", {}, token)
        assert "already playing" not in body.get("error", "")

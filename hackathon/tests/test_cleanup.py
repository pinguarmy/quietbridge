import io
import json
import threading
import time
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest

from hackathon.cleanup import CleanupClient, normalize_base_url
from hackathon.runtime import Runtime


VOCAB = {"HELP": {"phrase": "I need help.", "matches": ["help"], "verified": True}}
SECRET = "test-only-secret-never-return"


class Response:
    def __init__(self, text="OK"):
        self.body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size=-1):
        return io.BytesIO(self.body).read(size)


def wait_for(condition):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for cleanup worker")


def enabled_runtime():
    runtime = Runtime(rehearsal=True, vocabulary=VOCAB)
    runtime.speaker = Mock(available=True)
    runtime.configure_cleanup({"base_url": "https://cleanup.invalid/v1", "model": "test-model",
                               "api_key": SECRET, "enabled": True})
    return runtime


def test_config_normalizes_url_and_keeps_memory_key_out_of_reports():
    client = CleanupClient()
    try:
        assert client.public_config() == {"base_url": "https://api.openai.com/v1", "model": "",
                                          "enabled": False, "configured": False, "service_tier": "default"}
        with pytest.raises(ValueError):
            client.configure({"enabled": True, "model": "m"})
        config = client.configure({"base_url": "https://example.com", "model": "m",
                                   "api_key": SECRET, "enabled": True})
        assert config["base_url"] == "https://example.com/v1"
        assert config["configured"] is True
        assert SECRET not in json.dumps(config)
        # Empty key edits preserve the current in-memory credential.
        config = client.configure({"api_key": "", "model": "other"})
        assert config["configured"] is True and config["model"] == "other"
        for bad in ["http://example.com", "https://user:pass@example.com/v1",
                    "https://example.com/v1?q=secret", "https://example.com/v1#secret"]:
            with pytest.raises(ValueError):
                normalize_base_url(bad)
        assert normalize_base_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1"
        assert normalize_base_url("http://localhost:8000") == "http://localhost:8000/v1"
    finally:
        client.close()


def test_explicit_test_sends_only_constant_with_fake_urlopen():
    client = CleanupClient()
    client.configure({"base_url": "https://cleanup.invalid/v1", "model": "m", "api_key": SECRET})
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response("OK")

    try:
        with patch("hackathon.cleanup.urlopen", fake_urlopen):
            assert client.test() == {"ok": True}
        request, timeout = calls[0]
        assert timeout == 6
        assert request.full_url == "https://cleanup.invalid/v1/chat/completions"
        payload = json.loads(request.data)
        assert payload["messages"] == [{"role": "user", "content": "Please return exactly OK."}]
        assert request.get_header("Authorization") == "Bearer " + SECRET
    finally:
        client.close()


def test_fast_tier_is_explicit_and_luna_disables_reasoning():
    client = CleanupClient()
    client.configure({"model": "gpt-6-luna", "api_key": SECRET, "service_tier": "fast"})
    calls = []
    try:
        def fake(request, timeout):
            calls.append(json.loads(request.data))
            return Response("OK")
        with patch("hackathon.cleanup.urlopen", fake):
            client.test()
        assert calls[0]["service_tier"] == "fast"
        assert calls[0]["reasoning_effort"] == "none"
        assert calls[0]["max_completion_tokens"] == 256
    finally:
        client.close()


def test_explicit_welcome_hint_is_labelled_and_uses_requested_dee_spelling():
    client = CleanupClient()
    calls = []
    config = {"model": "gpt-6-luna", "api_key": SECRET,
              "base_url": "https://cleanup.invalid/v1", "service_tier": "fast"}
    try:
        def fake(request, timeout):
            calls.append(json.loads(request.data))
            return Response("Welcome to OpenAI DEE Hackathon")
        with patch("hackathon.cleanup.urlopen", fake):
            assert client._chat(config, '{"candidates":["Please wait."],"gesture_labels":[]}') == "Welcome to OpenAI DEE Hackathon"
        system = calls[0]["messages"][0]["content"]
        assert "Hi, Welcome to OpenAI DEE Hackathon" in system
        assert "Return exactly one" in system
        assert "not unrestricted transcription" in system
        assert "preserve raw recognition separately" in system
        assert calls[0]["messages"][1]["content"].startswith('{"candidates"')
    finally:
        client.close()


@pytest.mark.parametrize("text", ["garbled phrase", "Welcome to OpenAI DDE Hackathon", "Here is your answer: Welcome to OpenAI DEE Hackathon", ""])
def test_strict_welcome_allowlist_rejects_other_provider_output(text):
    client = CleanupClient()
    config = {"model": "gpt-6-luna", "api_key": SECRET, "base_url": "https://cleanup.invalid/v1"}
    try:
        with patch("hackathon.cleanup.urlopen", lambda *args, **kwargs: Response(text)):
            with pytest.raises(ValueError, match="two allowed phrases"):
                client._chat(config, '{"candidates":["raw"]}')
    finally:
        client.close()


def test_http_error_is_generic_and_does_not_leak_key():
    client = CleanupClient()
    client.configure({"model": "m", "api_key": SECRET})
    error = HTTPError("https://cleanup.invalid", 429, "denied " + SECRET, {}, None)
    try:
        with patch("hackathon.cleanup.urlopen", side_effect=error):
            with pytest.raises(ValueError) as caught:
                client.test()
        assert str(caught.value) == "Cleanup service returned HTTP 429"
        assert SECRET not in str(caught.value)
    finally:
        client.close()


def test_ready_suggestion_preserves_raw_and_never_auto_speaks():
    runtime = enabled_runtime()
    try:
        with patch("hackathon.cleanup.urlopen", lambda *args, **kwargs: Response("Hi, Welcome to OpenAI DEE Hackathon")):
            runtime.action("rehearse", {"text": "raw lip text"})
            revision = runtime.revision
            wait_for(lambda: runtime.snapshot()["cleanup"]["status"] == "ready")
        state = runtime.snapshot()
        assert state["decision"]["text"] == "raw lip text"
        assert state["candidates"] == ["raw lip text"]
        assert state["cleanup"] == {"status": "ready", "suggestion": "Hi, Welcome to OpenAI DEE Hackathon",
                                    "latency_ms": state["cleanup"]["latency_ms"], "error": "",
                                    "source_revision": revision}
        assert state["alternatives"] == ["raw lip text", "Hi, Welcome to OpenAI DEE Hackathon"]
        assert SECRET not in json.dumps(state)
        runtime.speaker.speak.assert_not_called()
        runtime.action("confirm", {"revision": revision, "text": "Hi, Welcome to OpenAI DEE Hackathon"})
        runtime.speaker.speak.assert_not_called()
        runtime.action("speak", {"revision": revision, "text": "Hi, Welcome to OpenAI DEE Hackathon"})
        runtime.speaker.speak.assert_called_once_with("Hi, Welcome to OpenAI DEE Hackathon")
    finally:
        runtime.close()


def blocked_response(started, proceed, text="Too late"):
    def fake_urlopen(*args, **kwargs):
        started.set()
        assert proceed.wait(2)
        return Response(text)
    return fake_urlopen


def test_worker_discards_results_after_confirm_and_cancel():
    for action in ["confirm", "cancel"]:
        runtime = enabled_runtime()
        started, proceed = threading.Event(), threading.Event()
        try:
            with patch("hackathon.cleanup.urlopen", blocked_response(started, proceed)):
                runtime.action("rehearse", {"text": "raw lip text"})
                old_revision = runtime.revision
                assert started.wait(1)
                if action == "confirm":
                    runtime.action("confirm", {"revision": old_revision, "text": "raw lip text"})
                else:
                    runtime.action("cancel", {})
                proceed.set()
                time.sleep(.05)
            state = runtime.snapshot()
            assert "Too late" not in state["alternatives"]
            assert state["cleanup"]["suggestion"] == ""
            assert state["decision"]["text"] == ("raw lip text" if action == "confirm" else "")
            runtime.speaker.speak.assert_not_called()
        finally:
            proceed.set()
            runtime.close()

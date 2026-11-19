"""State contract added for the main page: quick phrases, model-change notice,
speech errors, and the speak/dispatch contract. Rehearsal runtimes and fake
speakers only: these are software contracts, not recognition or audio checks.
"""
import http.client
import json
import os
import stat
import sys
import threading
import time
from unittest.mock import Mock

import pytest

from quietbridge import runtime as runtime_module
from quietbridge.profiles import ProfileManager, profile_lip_model, profile_motion_model
from quietbridge.runtime import ProfileSwitchConflict, Runtime, quick_phrases
from quietbridge.server import make_server
from quietbridge.speech import Speaker


class FakeSpeaker:
    available = True
    backend = "fake"

    def __init__(self):
        self.calls = []
        self.speaking = False
        self.last_error = ""
        self.cleared = 0
        self.block = None
        self.entered = threading.Event()
        self.raise_error = None

    def speak(self, text):
        self.entered.set()
        if self.block is not None:
            assert self.block.wait(5)
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append(text)
        self.speaking = True

    def stop(self):
        self.speaking = False

    def clear_error(self):
        self.cleared += 1
        self.last_error = ""

    def close(self):
        pass


def make_runtime(tmp_path, prompts=None, **kwargs):
    manager = ProfileManager(root_dir=tmp_path)
    profile = manager.create_profile("alice", "Alice", lip_prompts=prompts)
    kwargs.setdefault("rehearsal", True)
    runtime = Runtime(profile=profile, profile_manager=manager, **kwargs)
    runtime.speaker.close()
    runtime.speaker = FakeSpeaker()
    runtime._model_watch._interval = 0  # tests observe changes immediately
    return runtime, manager, profile


def accepted(runtime, text="I need some help."):
    runtime.action("rehearse", {"text": text, "label": ""})
    runtime.action("confirm", {"revision": runtime.revision, "text": text})
    return {"revision": runtime.revision, "text": text}


# quick_phrases ---------------------------------------------------------------

def test_quick_phrases_are_trimmed_unique_bounded_and_ordered():
    prompts = ["  Hello  ", "Hello", "", "   ", "x" * 2001, "Bye", 7, *[f"p{i}" for i in range(60)]]
    phrases = quick_phrases(prompts)
    assert phrases[:2] == ["Hello", "Bye"]
    assert len(phrases) == 50 and phrases[-1] == "p47"
    assert quick_phrases(None) == [] and quick_phrases("Hello") == []


def test_state_carries_the_active_profiles_own_phrases(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, prompts=["Coffee, no sugar", " Call Sam ", "Coffee, no sugar"])
    state = runtime.snapshot()
    assert state["quick_phrases"] == ["Coffee, no sugar", "Call Sam"]
    runtime.close()


def test_stock_configuration_has_no_quick_phrases():
    runtime = Runtime(rehearsal=True)
    try:
        state = runtime.snapshot()
        assert state["quick_phrases"] == []
        assert state["models_reload_needed"] is False and state["models_reload_detail"] == ""
    finally:
        runtime.close()


def test_phrase_edits_saved_by_another_process_appear_without_reload(tmp_path):
    runtime, manager, _ = make_runtime(tmp_path, prompts=["Old phrase"])
    assert runtime.snapshot()["quick_phrases"] == ["Old phrase"]
    revision = runtime.revision
    time.sleep(0.01)
    manager.update_profile("alice", lip_prompts=["New phrase", "Another"])
    os.utime(runtime.profile.config_path, ns=(time.time_ns(), time.time_ns() + 10_000_000))
    state = runtime.snapshot()
    assert state["quick_phrases"] == ["New phrase", "Another"]
    assert state["revision"] == revision  # text shortcuts only: the message is untouched
    runtime.profile.config_path.write_text("{ damaged", encoding="utf-8")
    assert runtime.snapshot()["quick_phrases"] == ["New phrase", "Another"]
    runtime.close()


# models_reload_needed -----------------------------------------------------------

def write_model(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    # Distinct mtimes even on coarse filesystems.
    stamp = time.time_ns() + int.from_bytes(os.urandom(2), "big") * 1_000_000
    os.utime(path, ns=(stamp, stamp))


def test_newly_saved_gesture_model_is_flagged_and_reload_clears_it(tmp_path):
    runtime, _, profile = make_runtime(tmp_path)
    assert runtime.snapshot()["models_reload_needed"] is False
    revision = runtime.revision
    write_model(profile_motion_model(profile), b"motion-v1")
    state = runtime.snapshot()
    assert state["models_reload_needed"] is True
    assert state["models_reload_detail"] == "A newer gesture model was saved for this profile."
    assert state["revision"] == revision  # never reloads by itself
    assert runtime.motion_model_path is None

    runtime.switch_profile("alice")  # the explicit "Reload current models"
    assert runtime.motion_model_path == profile_motion_model(profile)
    state = runtime.snapshot()
    assert state["models_reload_needed"] is False and state["models_reload_detail"] == ""
    runtime.close()


def test_same_content_touch_is_ignored_changed_or_removed_is_flagged(tmp_path):
    runtime, _, profile = make_runtime(tmp_path)
    path = profile_motion_model(profile)
    write_model(path, b"motion-v1")
    runtime.switch_profile("alice")
    assert runtime.snapshot()["models_reload_needed"] is False  # also records the baseline hash
    write_model(path, b"motion-v1")  # re-saved identical file: new mtime only
    assert runtime.snapshot()["models_reload_needed"] is False
    write_model(path, b"motion-v2")
    state = runtime.snapshot()
    assert state["models_reload_needed"] is True and "gesture model" in state["models_reload_detail"]
    path.unlink()
    state = runtime.snapshot()
    assert state["models_reload_detail"] == "The gesture model this profile was using was removed."
    runtime.close()


def test_several_changed_models_are_all_named(tmp_path):
    runtime, _, profile = make_runtime(tmp_path)
    write_model(profile_motion_model(profile), b"m")
    write_model(profile_lip_model(profile), b"l")
    detail = runtime.snapshot()["models_reload_detail"]
    assert "gesture model" in detail and "lip-reading model" in detail
    runtime.close()


def test_slot_pinned_by_explicit_path_is_ignored(tmp_path):
    explicit = tmp_path / "cli-lip.pth"
    explicit.write_bytes(b"cli")
    runtime, _, profile = make_runtime(tmp_path, lip_model=explicit, explicit_lip_model=explicit)
    write_model(profile_lip_model(profile), b"promoted")
    assert runtime.snapshot()["models_reload_needed"] is False
    write_model(profile_motion_model(profile), b"motion")
    assert runtime.snapshot()["models_reload_detail"] == "A newer gesture model was saved for this profile."
    runtime.close()


def test_lip_slots_ignored_when_lips_are_disabled(tmp_path):
    runtime, _, profile = make_runtime(tmp_path, enable_lips=False)
    write_model(profile_lip_model(profile), b"promoted")
    assert runtime.snapshot()["models_reload_needed"] is False
    runtime.close()


def test_checks_are_throttled_and_run_outside_the_runtime_lock(tmp_path, monkeypatch):
    runtime, _, profile = make_runtime(tmp_path)
    now = [100.0]
    watch = runtime._model_watch
    watch._clock = lambda: now[0]
    watch._interval = 1.0
    watch._next_check = 0.0
    held = []
    real = runtime_module._stat_signature

    def recording(path):
        held.append(runtime.lock._is_owned())
        return real(path)

    monkeypatch.setattr(runtime_module, "_stat_signature", recording)
    runtime.snapshot()
    calls = len(held)
    assert calls > 0 and not any(held)
    write_model(profile_motion_model(profile), b"m")
    for _ in range(20):
        assert runtime.snapshot()["models_reload_needed"] is False  # cached within the interval
    assert len(held) == calls
    now[0] += 1.01
    assert runtime.snapshot()["models_reload_needed"] is True
    runtime.close()


def test_failed_reload_keeps_the_notice(tmp_path):
    runtime, _, profile = make_runtime(tmp_path)
    write_model(profile_motion_model(profile), b"m")
    assert runtime.snapshot()["models_reload_needed"] is True
    runtime._switching_profile = True
    with pytest.raises(ProfileSwitchConflict):
        runtime.switch_profile("alice")
    runtime._switching_profile = False
    assert runtime.snapshot()["models_reload_needed"] is True
    runtime.close()


# speak contract ----------------------------------------------------------------

def test_speak_while_speaking_is_rejected_without_restarting(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    request = accepted(runtime)
    runtime.action("speak", request)
    assert runtime.speaker.calls == ["I need some help."]
    assert runtime.snapshot()["speaking"] is True
    with pytest.raises(ValueError, match="already playing"):
        runtime.action("speak", request)
    assert runtime.speaker.calls == ["I need some help."]
    runtime.action("stop_speaking", {})
    runtime.action("speak", request)  # after stopping, an explicit Speak plays it again
    assert runtime.speaker.calls == ["I need some help."] * 2
    runtime.close()


def test_voice_starts_outside_the_runtime_lock(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    request = accepted(runtime)
    runtime.speaker.block = threading.Event()
    errors = []
    thread = threading.Thread(target=lambda: _call(runtime, request, errors))
    thread.start()
    assert runtime.speaker.entered.wait(2)
    # Polling and other actions stay responsive while the voice starts.
    assert runtime.lock.acquire(timeout=0.5)
    runtime.lock.release()
    state = runtime.snapshot()
    assert state["speaking"] is True and state["last_spoken"] == ""
    with pytest.raises(ValueError, match="already playing"):
        runtime.action("speak", request)
    with pytest.raises(ValueError, match="speech"):
        runtime.action("new_message", {"revision": runtime.revision})
    runtime.speaker.block.set()
    thread.join(2)
    assert errors == []
    state = runtime.snapshot()
    assert state["last_spoken"] == "I need some help."
    assert "tts_dispatch_ms" in state["metrics"]
    assert runtime.speaker.calls == ["I need some help."]
    runtime.close()


def _call(runtime, request, errors):
    try:
        runtime.action("speak", request)
    except Exception as exc:  # pragma: no cover - reported by the assertion
        errors.append(exc)


def test_failed_voice_start_changes_nothing(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    request = accepted(runtime)
    runtime.speaker.raise_error = RuntimeError("Native speech could not start (fake)")
    with pytest.raises(RuntimeError, match="could not start"):
        runtime.action("speak", request)
    state = runtime.snapshot()
    assert state["last_spoken"] == "" and state["speaking"] is False
    assert state["revision"] == request["revision"] and state["confirmed"] is True
    runtime.close()


def test_speak_stays_revision_bound(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    request = accepted(runtime)
    runtime.action("rehearse", {"text": "Something newer", "label": ""})
    with pytest.raises(ValueError, match="result changed"):
        runtime.action("speak", request)
    assert runtime.speaker.calls == []
    runtime.close()


def test_profile_switch_during_voice_start_keeps_the_fresh_state(tmp_path):
    runtime, manager, _ = make_runtime(tmp_path)
    manager.create_profile("bob", "Bob")
    request = accepted(runtime)
    runtime.speaker.block = threading.Event()
    errors = []
    thread = threading.Thread(target=lambda: _call(runtime, request, errors))
    thread.start()
    assert runtime.speaker.entered.wait(2)
    runtime.switch_profile("bob")
    runtime.speaker.block.set()
    thread.join(2)
    state = runtime.snapshot()
    assert state["profile_id"] == "bob" and state["last_spoken"] == ""
    assert "tts_dispatch_ms" not in state["metrics"]
    runtime.close()


def test_new_message_clears_the_speech_error(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    runtime.speaker.last_error = "Speech did not play: the fake voice exited with status 1."
    assert runtime.snapshot()["speech_error"].startswith("Speech did not play")
    runtime.action("new_message", {"revision": runtime.revision})
    assert runtime.speaker.cleared == 1
    assert runtime.snapshot()["speech_error"] == ""
    runtime.close()


def test_mock_and_minimal_speakers_report_no_error():
    runtime = Runtime(rehearsal=True)
    runtime.speaker.close()
    runtime.speaker = Mock(available=True)
    assert runtime.snapshot()["speech_error"] == "" and runtime.snapshot()["speaking"] is False

    class Minimal:  # the browser-check fixture's SilentSpeaker shape
        available = True

        def speak(self, text):
            pass

        def close(self):
            pass

    runtime.speaker = Minimal()
    request = accepted(runtime)
    runtime.action("speak", request)
    runtime.action("new_message", {"revision": runtime.revision})
    assert runtime.snapshot()["speech_error"] == ""
    runtime.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script as a failing voice")
def test_real_failing_voice_surfaces_in_state_then_new_message_clears_it(tmp_path):
    voice = tmp_path / "espeak-ng"
    voice.write_text("#!/bin/sh\ncat >/dev/null\necho 'no audio device' >&2\nexit 1\n")
    voice.chmod(voice.stat().st_mode | stat.S_IXUSR)
    runtime, _, _ = make_runtime(tmp_path)
    runtime.speaker = Speaker(platform="linux", which=lambda name: str(voice) if name == "espeak-ng" else None)
    runtime.action("speak", accepted(runtime))
    deadline = time.monotonic() + 5
    while runtime.snapshot()["speaking"] and time.monotonic() < deadline:
        time.sleep(0.01)
    state = runtime.snapshot()
    assert state["last_spoken"] == "I need some help."  # it was dispatched ...
    assert "no audio device" in state["speech_error"]   # ... and the failure is visible
    runtime.action("new_message", {"revision": runtime.revision})
    assert runtime.snapshot()["speech_error"] == ""
    runtime.close()


# HTTP ---------------------------------------------------------------------------

def test_http_state_exposes_the_contract_types(tmp_path):
    runtime, _, profile = make_runtime(tmp_path, prompts=["Hello there"])
    write_model(profile_motion_model(profile), b"m")
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{server.server_port}"})
        state = json.loads(connection.getresponse().read())
        connection.close()
        assert state["quick_phrases"] == ["Hello there"]
        assert state["models_reload_needed"] is True
        assert isinstance(state["models_reload_detail"], str) and state["models_reload_detail"]
        assert state["speech_error"] == ""
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()

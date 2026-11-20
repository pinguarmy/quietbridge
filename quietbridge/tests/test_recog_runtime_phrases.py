"""Runtime wiring of personal phrase suggestions (rehearsal and direct publishes
with constructed lip results). Software contracts only, not recognition."""
import json
from unittest.mock import Mock, patch

import pytest

from quietbridge.contracts import LipResult, SignEvent
from quietbridge.profiles import ProfileManager
from quietbridge.runtime import Runtime

PROMPTS = ["I need some help.", "Please write it down.", "No, I do not agree.", "Yes"]


def make(tmp_path, prompts=PROMPTS, **kwargs):
    manager = ProfileManager(root_dir=tmp_path)
    profile = manager.create_profile("alice", "Alice", lip_prompts=prompts)
    kwargs.setdefault("rehearsal", True)
    runtime = Runtime(profile=profile, profile_manager=manager, **kwargs)
    runtime.speaker.close()
    runtime.speaker = Mock(available=True, speaking=False, last_error="", backend="fake")
    runtime.cleanup.submit = Mock(return_value=False)
    return runtime


@pytest.fixture
def runtime(tmp_path):
    rt = make(tmp_path)
    yield rt
    rt.close()


def publish(rt, candidates, events=(), scores=None):
    with rt.lock:
        rt._publish(LipResult(list(candidates), 1, scores or []), list(events), 5000, 1000)
    return rt.snapshot()


def test_phrase_is_added_behind_lip_candidates_and_never_replaces_the_top(runtime):
    state = publish(runtime, ["I NEED SOME HELL", "I NEED SOME HEL", "I KNEAD SOME HELL"], scores=[-1, -2, -3])
    assert state["decision"]["text"] == "I NEED SOME HELL"
    assert state["candidates"] == ["I NEED SOME HELL", "I NEED SOME HEL", "I KNEAD SOME HELL"]
    assert state["candidate_scores"] == [-1, -2, -3]
    assert state["alternatives"] == [*state["candidates"], "I need some help."]
    assert state["phrase_matches"] == ["I need some help."]
    assert state["decision"]["status"] == "confirmation_needed"
    evidence = state["metrics"]["phrase_match"]
    assert evidence["offered"][0]["candidate_index"] == 0 and evidence["offered"][0]["scope"] == "window"
    assert "I need some help." not in json.dumps(state["metrics"])  # no text in metrics


def test_rehearsal_uses_the_same_path_and_confirm_is_unchanged(runtime):
    runtime.action("rehearse", {"text": "PLEASE RIDE IT DOWN", "label": ""})
    state = runtime.snapshot()
    assert state["phrase_matches"] == ["Please write it down."]
    runtime.action("confirm", {"revision": state["revision"], "text": "Please write it down."})
    state = runtime.snapshot()
    assert state["decision"]["text"] == "Please write it down." and state["confirmed"]
    assert state["metrics"]["accepted_source"] == "phrase_suggestion"
    runtime.speaker.speak.assert_not_called()  # speech still needs the person's action


def test_at_most_two_suggestions(tmp_path):
    rt = make(tmp_path, prompts=["I need some tea.", "I need some help.", "I need some milk."])
    try:
        state = publish(rt, ["I NEED SOME HELL"])
        assert len(state["phrase_matches"]) == 2
        assert state["alternatives"][0] == "I NEED SOME HELL"
    finally:
        rt.close()


@pytest.mark.parametrize("lips", [None, LipResult([], 0), LipResult(["", "  "], 0)])
def test_nothing_without_lip_candidates(tmp_path, lips):
    vocabulary = {"HELP": {"phrase": "I need some help", "matches": ["help"], "verified": True}}
    rt = make(tmp_path, rehearsal=False, vocabulary=vocabulary)
    try:
        with rt.lock:
            rt._publish(lips, [SignEvent("HELP", .9, 1000, 1500, True)], 2000)
        state = rt.snapshot()
        assert state["phrase_matches"] == []
        assert "I need some help." not in state["alternatives"]
    finally:
        rt.close()


def test_below_threshold_offers_nothing(runtime):
    state = publish(runtime, ["I NEED WATER"])
    assert state["phrase_matches"] == [] and state["alternatives"] == ["I NEED WATER"]


def test_sensitive_difference_is_a_visible_conflict_not_a_default(runtime):
    state = publish(runtime, ["I DO AGREE", "I DO NOT AGREE"])
    assert state["decision"]["text"] == "I DO AGREE"   # top candidate kept
    assert state["phrase_matches"] == ["No, I do not agree."]
    assert state["decision"]["reason"] == "conflict"  # UI clears any preselection
    assert state["decision"]["status"] == "confirmation_needed"
    assert state["metrics"]["phrase_match"]["offered"][0]["sensitive_difference"] is True


def test_gesture_agreement_keeps_its_decision_and_gains_only_a_row(tmp_path):
    vocabulary = {"HELP": {"phrase": "help", "matches": ["help"], "verified": True}}
    rt = make(tmp_path, rehearsal=False, vocabulary=vocabulary)
    try:
        state = publish(rt, ["I NEED SOME HELL", "I NEED HELP"], [SignEvent("HELP", .9, 1100, 1500, True)])
        assert state["decision"]["text"] == "I NEED HELP" and state["decision"]["reason"] == "agreement"
        assert state["alternatives"][-1] == "I need some help."
        assert state["decision"]["status"] == "confirmation_needed"
    finally:
        rt.close()


def test_new_revision_clears_stale_matches(runtime):
    assert publish(runtime, ["I NEED SOME HELL"])["phrase_matches"]
    state = publish(runtime, ["GOOD MORNING"])
    assert state["phrase_matches"] == [] and "I need some help." not in state["alternatives"]
    publish(runtime, ["I NEED SOME HELL"])
    runtime.action("cancel", {})
    assert runtime.snapshot()["phrase_matches"] == []


def test_disabled_by_constructor_environment_or_method(tmp_path, monkeypatch):
    rt = make(tmp_path / "a", phrase_suggestions=False)
    try:
        state = publish(rt, ["I NEED SOME HELL"])
        assert state["phrase_matches"] == [] and state["metrics"]["phrase_match"]["enabled"] is False
    finally:
        rt.close()
    monkeypatch.setenv("QUIETBRIDGE_PHRASE_SUGGESTIONS", "0")
    rt = make(tmp_path / "b")
    try:
        assert publish(rt, ["I NEED SOME HELL"])["phrase_matches"] == []
        rt.set_phrase_suggestions(True)
        assert publish(rt, ["I NEED SOME HELL"])["phrase_matches"] == ["I need some help."]
    finally:
        rt.close()


def test_no_profile_means_no_phrases():
    rt = Runtime(rehearsal=True)
    try:
        rt.action("rehearse", {"text": "I NEED SOME HELL", "label": ""})
        assert rt.snapshot()["phrase_matches"] == []
    finally:
        rt.close()


def test_correction_drops_phrase_rows_and_learned_text_becomes_a_source(runtime):
    state = publish(runtime, ["I NEED SOME HELL"])
    runtime.action("correct", {"revision": state["revision"], "text": "Call my sister Ann", "learn": True})
    state = runtime.snapshot()
    assert state["phrase_matches"] == [] and "I need some help." not in state["alternatives"]
    assert state["metrics"]["accepted_source"] == "manual_edit"
    state = publish(runtime, ["CALL MY SISTER AND"])
    assert state["phrase_matches"] == ["Call my sister Ann"]


def test_unlearned_or_held_out_corrections_are_not_sources(tmp_path):
    rt = make(tmp_path, prompts=["Hello there friend"])
    try:
        held = rt.profile.mouths_dir / "validation" / "clip.npz"
        held.parent.mkdir(parents=True, exist_ok=True)
        held.write_bytes(b"")
        lines = [{"learn": True, "profile_id": rt.profile.id, "corrected": "Call my sister Ann",
                  "sources": ["mouths/validation/clip.npz"]},
                 {"learn": False, "profile_id": rt.profile.id, "corrected": "Open the back door"}]
        (rt.profile.profile_dir / "corrections.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        rt._model_watch._interval = 0
        rt.snapshot()  # the watch notices the changed file
        assert publish(rt, ["CALL MY SISTER AND"])["phrase_matches"] == []
        assert publish(rt, ["OPEN THE PACK DOOR"])["phrase_matches"] == []
        with (rt.profile.profile_dir / "corrections.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"learn": True, "profile_id": rt.profile.id,
                                     "corrected": "Open the back door"}) + "\n")
        rt.snapshot()
        assert publish(rt, ["OPEN THE PACK DOOR"])["phrase_matches"] == ["Open the back door"]
    finally:
        rt.close()


def test_cleanup_paths_keep_phrase_rows_and_get_no_phrase_text(runtime):
    runtime.cleanup.submit = Mock(return_value=True)
    with patch.object(runtime.cleanup, "public_config", return_value={"enabled": True}):
        state = publish(runtime, ["I NEED SOME HELL"])
        args = runtime.cleanup.submit.call_args[0]
        assert args[2] == ["I NEED SOME HELL"]  # only the displayed text leaves the device
        runtime._cleanup_result(runtime._cleanup_generation, runtime.revision,
                                {"status": "ready", "suggestion": "I need some help."})
        assert runtime.snapshot()["alternatives"].count("I need some help.") == 1
    with patch.object(runtime.cleanup, "configure", return_value={"enabled": False}):
        runtime.configure_cleanup({})
    state = runtime.snapshot()
    assert "I need some help." in state["alternatives"] and state["phrase_matches"] == ["I need some help."]
    runtime.action("confirm", {"revision": state["revision"], "text": "I need some help."})


def test_session_matches_whole_text_and_last_chunk(tmp_path):
    rt = make(tmp_path, rehearsal=False, browser_camera=True, vocabulary={})
    try:
        rt.action("camera_session", {"active": True})
        rt.action("mode", {"mode": "lips"})
        with rt.lock:
            rt._publish_session(LipResult(["I NEED"], 1, [-1.0], 100, 900, 20, 25.0, 40))
            rt._publish_session(LipResult(["SOME HELL", "SUM HELL"], 1, [-2.0, -3.0], 1500, 2300, 21, 25.0, 41))
        state = rt.snapshot()
        assert state["decision"]["text"] == "I NEED SOME HELL"
        assert state["alternatives"][:2] == ["I NEED SOME HELL", "I NEED SUM HELL"]
        assert state["phrase_matches"] == ["I need some help."]
        assert state["metrics"]["phrase_match"]["offered"][0]["scope"] == "session"
        evidence = state["metrics"]["session_chunk_evidence"]
        assert [e["candidate_scores"] for e in evidence] == [[-1.0], [-2.0, -3.0]]
        assert evidence[1]["raw_frames"] == 21 and evidence[1]["max_frame_gap_ms"] == 41
        with rt.lock:
            rt._publish_session(LipResult(["PLEASE RIDE IT DOWN"], 1))
        state = rt.snapshot()
        assert "I NEED SOME HELL Please write it down." in state["phrase_matches"]
        assert state["decision"]["text"] == "I NEED SOME HELL PLEASE RIDE IT DOWN"
        rt.action("camera_session", {"active": False})
        state = rt.snapshot()
        rt.action("confirm", {"revision": state["revision"], "text": state["phrase_matches"][0]})
        assert rt.snapshot()["metrics"]["accepted_source"] == "phrase_suggestion"
    finally:
        rt.close()


def test_provenance_of_lip_choices(runtime):
    state = publish(runtime, ["GOOD MORNING", "GOOD EVENING"])
    runtime.action("confirm", {"revision": state["revision"], "text": "GOOD EVENING"})
    metrics = runtime.snapshot()["metrics"]
    assert metrics["accepted_source"] == "lip_candidate" and metrics["accepted_candidate_index"] == 1
    state = publish(runtime, ["GOOD MORNING", "GOOD EVENING"])
    assert "accepted_source" not in state["metrics"]
    runtime.action("confirm", {"revision": state["revision"], "text": "GOOD MORNING"})
    assert runtime.snapshot()["metrics"]["accepted_source"] == "lip_top"


def test_profile_switch_resets_matches(tmp_path):
    rt = make(tmp_path)
    try:
        rt.profile_manager.create_profile("bob", "Bob", lip_prompts=["Something else entirely"])
        assert publish(rt, ["I NEED SOME HELL"])["phrase_matches"]
        rt.switch_profile("bob")
        assert rt.snapshot()["phrase_matches"] == []
        assert publish(rt, ["I NEED SOME HELL"])["phrase_matches"] == []
    finally:
        rt.close()

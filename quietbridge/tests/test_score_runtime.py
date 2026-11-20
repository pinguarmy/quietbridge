"""Runtime wiring of model-scored phrase offers and the capture warning.

A tiny fake scorer stands in for the lip model: these are software contracts
(gating, ordering, fallback, staleness, safety rules), not recognition."""
from collections import deque
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quietbridge import phrase_match
from quietbridge.contracts import LipResult
from quietbridge.lips_adapter import LipsAdapter
from quietbridge.profiles import ProfileManager
from quietbridge.runtime import CAPTURE_MAX_GAP_MS, CAPTURE_MIN_FPS, Runtime

TOP = "I NEED SOME HELP NOW"
PROMPTS = ["I need some help now please", "I do not need some help now", "I need some milk now",
           "Call my sister"]


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


class FakeScorer:
    """Adapter-shaped: score_phrases/release_scoring like LipsAdapter."""

    def __init__(self, scores, top_score=-1.0, error=None):
        self.scores, self.top_score, self.error = scores, top_score, error
        self.asked, self.released = [], 0

    def score_phrases(self, phrases):
        self.asked.append(list(phrases))
        if self.error:
            raise self.error
        return {"top_score": self.top_score, "scores": [self.scores.get(p) for p in phrases], "scoring_ms": 3.0}

    def release_scoring(self):
        self.released += 1


def publish(rt, candidates, scoring):
    result = LipResult(list(candidates), 1, [-1.0] * len(candidates))
    with rt.lock:
        rt._publish(result, [], 5000, 1000, phrase_scoring=scoring)
    return rt.snapshot()


def scored(rt, candidates, scorer):
    return rt._score_saved_phrases(scorer, LipResult(list(candidates), 1))


def test_variant_below_the_decoded_text_is_not_offered_but_the_text_matcher_would(runtime):
    below = -1.0 - phrase_match.MODEL_OFFER_MARGIN
    scorer = FakeScorer({"I need some help now please": below - 0.2, "I do not need some help now": below - 0.1,
                         "I need some milk now": below - 0.01, "Call my sister": -3.0}, top_score=-1.0)
    scoring = scored(runtime, [TOP], scorer)
    assert scoring["path"] == "model" and scorer.released == 1
    state = publish(runtime, [TOP], scoring)
    assert state["phrase_matches"] == [] and state["alternatives"] == [TOP]
    evidence = state["metrics"]["phrase_match"]
    assert evidence["path"] == "model" and evidence["offered"] == []
    assert evidence["best_gap"] == pytest.approx(below - 0.01 + 1.0) and evidence["top_score"] == -1.0
    assert evidence["score_margin"] == phrase_match.MODEL_OFFER_MARGIN
    # Before (text only): the same correct result would have been sent to confirmation with an offer.
    before = publish(runtime, [TOP], None)
    assert before["phrase_matches"] and before["metrics"]["phrase_match"]["path"] == "text"
    assert before["metrics"]["phrase_match"]["fallback"] == "no_scorer"


def test_best_scoring_similar_phrase_is_offered_first_behind_lip_candidates(runtime):
    candidates = [TOP, "I NEED SOME HELP NOW PLEASE"[:-1]]
    scorer = FakeScorer({"I need some help now please": -0.9, "I do not need some help now": -0.7,
                         "I need some milk now": -0.95}, top_score=-1.0)
    state = publish(runtime, candidates, scored(runtime, candidates, scorer))
    assert state["candidates"] == candidates and state["decision"]["text"] == TOP  # lip ranking intact
    assert state["phrase_matches"] == ["I do not need some help now", "I need some help now please"]
    assert state["alternatives"] == [*candidates, *state["phrase_matches"]]
    assert state["decision"]["status"] == "confirmation_needed"
    assert state["decision"]["reason"] == "conflict"  # "not" differs from the displayed text
    offered = state["metrics"]["phrase_match"]["offered"]
    assert [o["path"] for o in offered] == ["model", "model"]
    assert offered[0]["score_gap"] == pytest.approx(0.3) and offered[0]["sensitive_difference"] is True
    dumped = json.dumps(state["metrics"])
    assert "sister" not in dumped and "milk" not in dumped.lower() and "%" not in dumped


def test_scores_never_add_anything_without_lip_candidates(runtime):
    scorer = FakeScorer({"Call my sister": 5.0})
    assert scored(runtime, ["", "  "], scorer) is None and scorer.asked == []
    state = publish(runtime, [], {"path": "model", "top_score": -1.0, "scores": {"Call my sister": 5.0}})
    assert state["phrase_matches"] == [] and state["alternatives"] == []


def test_fallbacks_are_recorded_and_keep_text_behaviour(runtime):
    assert scored(runtime, [TOP], object()) == {"path": "text", "fallback": "no_scorer"}
    broken = FakeScorer({}, error=RuntimeError("boom"))
    scoring = scored(runtime, [TOP], broken)
    assert scoring == {"path": "text", "fallback": "error", "error": "RuntimeError"} and broken.released == 1
    state = publish(runtime, [TOP], scoring)
    assert state["metrics"]["phrase_match"]["fallback"] == "error"
    assert state["phrase_matches"]  # the text matcher still decides
    none = FakeScorer({})
    none.score_phrases = lambda phrases: None
    assert scored(runtime, [TOP], none)["fallback"] == "unavailable"


def test_disabled_suggestions_do_not_score(runtime):
    runtime.set_phrase_suggestions(False)
    scorer = FakeScorer({p: 0.0 for p in PROMPTS})
    assert scored(runtime, [TOP], scorer) is None and scorer.asked == []


def test_unscorable_phrase_falls_back_to_the_text_rule(tmp_path):
    rt = make(tmp_path, prompts=["Room 112 please now", "Room one please now"])
    try:
        scorer = FakeScorer({"Room 112 please now": None, "Room one please now": -5.0}, top_score=-1.0)
        state = publish(rt, ["ROOM ONE TWELVE PLEASE NOW"], scored(rt, ["ROOM ONE TWELVE PLEASE NOW"], scorer))
        offered = state["metrics"]["phrase_match"]["offered"]
        assert all(o["path"] == "text" for o in offered)
        assert "Room one please now" not in state["phrase_matches"]  # model said no
        assert state["metrics"]["phrase_match"]["unscorable"] == 1
    finally:
        rt.close()


def test_a_phrase_removed_after_scoring_is_not_offered(runtime):
    scoring = {"path": "model", "top_score": -1.0, "scores": {"Deleted phrase please": 0.0}, "scoring_ms": 1}
    state = publish(runtime, [TOP], scoring)
    assert state["phrase_matches"] == []


def test_large_profiles_send_at_most_the_cap_to_the_scorer(tmp_path):
    prompts = [f"I need some help number {i}" for i in range(40)]
    rt = make(tmp_path, prompts=prompts)
    try:
        scorer = FakeScorer({})
        scored(rt, ["I NEED SOME HELP NUMBER"], scorer)
        assert len(scorer.asked[0]) == phrase_match.SCORE_MAX_PHRASES
    finally:
        rt.close()


# --- session: scores exist only for the last chunk -----------------------------------

def test_session_last_chunk_offers_are_model_gated(runtime):
    runtime.state["session_active"] = True
    with runtime.lock:
        runtime._publish_session(LipResult(["HELLO THERE"], 1))
    good = FakeScorer({"I need some milk now": -0.5, "I need some help now please": -2.0}, top_score=-1.0)
    with runtime.lock:
        runtime._publish_session(LipResult([TOP], 1), phrase_scoring=scored(runtime, [TOP], good))
    state = runtime.snapshot()
    assert "HELLO THERE I need some milk now" in state["phrase_matches"]
    assert not any("please" in text for text in state["phrase_matches"])
    scopes = {o["scope"]: o["path"] for o in state["metrics"]["phrase_match"]["offered"]}
    assert scopes.get("last_chunk") == "model"
    # A gesture-only republish keeps using the last chunk's scores, not the text rule.
    with runtime.lock:
        runtime._publish_session(None)
    assert runtime.snapshot()["phrase_matches"] == state["phrase_matches"]


# --- lips worker: scoring outside the lock, late scores discarded ---------------------

class Gate:
    def detect(self, frame, ts):
        pts = np.array([[60.0 + i, 50.0 + (i % 5)] for i in range(40)], dtype=np.float32)
        return SimpleNamespace(pts=pts, outer_lips=pts[:20], inner_lips=pts[20:], anchors=pts[:5], mouth_open=0.05)

    def close(self):
        pass


class Reader:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def encode(self, rois):
        return "enc"

    def beam_search_details(self, enc, nbest):
        return [{"text": TOP, "score": -1.0}]

    def score_texts(self, enc, texts):
        self.entered.set()
        assert self.release.wait(3)
        table = {TOP: -1.0, "I need some milk now": -0.5}
        return [{"normalized": table.get(t, -9.0)} for t in texts]


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr("lipflow.camera.face_crop", lambda gray, obs, rec: ("crop", (0, 0)))
    monkeypatch.setattr("lipflow.dictation.clip_problem", lambda rec: None)
    monkeypatch.setattr("lipflow.dictation.rois_for", lambda rec: np.zeros((12, 96, 96), dtype=np.uint8))
    adapter = object.__new__(LipsAdapter)
    adapter.tracker, adapter.reader = Gate(), Reader()
    adapter.model_info = {"mode": "stock"}
    adapter.recording = None
    adapter.last_tracking = adapter.last_mouth_open = adapter.record_start_ms = None
    adapter.pre_roll_ms = 300
    adapter._preroll = deque(maxlen=20)
    adapter._last_window_end_ms = None
    adapter.last_metrics = {}
    now = [1000]
    monkeypatch.setattr("quietbridge.lips_adapter.LipsAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr("quietbridge.runtime.clock_ms", lambda: now[0])
    rt = make(tmp_path, rehearsal=False, browser_camera=True, vocabulary={})
    rt.state["camera"] = "ready"
    thread = threading.Thread(target=rt._lips, daemon=True)
    rt.threads.append(thread)
    thread.start()
    wait_for(lambda: rt.snapshot()["lips"] == "ready")
    try:
        yield rt, adapter, now
    finally:
        adapter.reader.release.set()
        rt.close()
        thread.join(3)


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("worker did not reach the expected state")


def record(rt, adapter, now, timestamps):
    rt.action("start", {})
    for ts in timestamps:
        rt.receive_frame(np.zeros((120, 160, 3), dtype=np.uint8), ts)
    wait_for(lambda: adapter.recording is not None and len(adapter.recording.ts) == len(timestamps))
    now[0] = timestamps[-1] + 10
    rt.action("stop", {})
    now[0] += 400


def test_worker_offers_from_model_scores_and_releases_the_encoder(worker):
    rt, adapter, now = worker
    record(rt, adapter, now, [1000 + 40 * i for i in range(15)])
    wait_for(lambda: rt.snapshot()["phase"] == "idle" and rt.snapshot()["candidates"])
    state = rt.snapshot()
    assert state["phrase_matches"] == ["I need some milk now"]
    assert state["metrics"]["phrase_match"]["path"] == "model"
    assert state["metrics"]["phrase_scoring_ms"] >= 0
    assert adapter._scoring is None
    assert state["capture_warning"] == ""  # 25 fps, no gap, no drops


def test_cancel_while_scoring_discards_the_late_result_and_its_scores(worker):
    rt, adapter, now = worker
    adapter.reader.release.clear()
    record(rt, adapter, now, [1000 + 40 * i for i in range(15)])
    assert adapter.reader.entered.wait(3)  # scoring runs: the runtime lock must be free
    with rt.lock:
        revision = rt.revision
    rt.action("cancel", {})
    adapter.reader.release.set()
    time.sleep(0.2)
    state = rt.snapshot()
    assert state["candidates"] == [] and state["phrase_matches"] == []
    assert state["revision"] >= revision and adapter._scoring is None


def test_capture_warning_for_a_slow_camera_then_cleared(worker):
    rt, adapter, now = worker
    record(rt, adapter, now, [1000 + 70 * i for i in range(15)])  # ~14 fps
    wait_for(lambda: rt.snapshot()["phase"] == "idle" and rt.snapshot()["candidates"])
    state = rt.snapshot()
    assert state["capture_warning"].startswith("The camera sent only 14 pictures per second")
    quality = state["metrics"]["capture_quality"]
    assert quality["reasons"] == ["low_frame_rate"] and quality["min_fps"] == CAPTURE_MIN_FPS
    record(rt, adapter, now, [3000 + 40 * i for i in range(15)])
    wait_for(lambda: rt.snapshot()["revision"] > state["revision"] and rt.snapshot()["phase"] == "idle")
    assert rt.snapshot()["capture_warning"] == ""


def test_capture_warning_for_a_stall_inside_the_utterance(worker):
    rt, adapter, now = worker
    stamps = [1000 + 40 * i for i in range(8)] + [1280 + CAPTURE_MAX_GAP_MS + 40 * i for i in range(8)]
    record(rt, adapter, now, stamps)
    wait_for(lambda: rt.snapshot()["phase"] == "idle" and rt.snapshot()["candidates"])
    state = rt.snapshot()
    assert "paused for 0.3 seconds" in state["capture_warning"]
    assert "frame_gap" in state["metrics"]["capture_quality"]["reasons"]


def test_capture_warning_unit_rules(runtime):
    ok = LipResult(["A"], 1, raw_frames=30, capture_fps=29.9, max_frame_gap_ms=CAPTURE_MAX_GAP_MS - 1)
    assert runtime._capture_warning(ok, 0)[0] == ""
    short = LipResult(["A"], 1, raw_frames=5, capture_fps=8.0, max_frame_gap_ms=100)
    assert runtime._capture_warning(short, 0)[0] == ""  # too few frames to judge a rate
    message, quality = runtime._capture_warning(ok, 3)
    assert "skipped 3 camera pictures" in message and quality["reasons"] == ["dropped_frames"]
    assert runtime._capture_warning(None, 0) == ("", runtime._capture_warning(None, 0)[1])
    # A typed or gesture-only result clears an older warning.
    runtime.state["capture_warning"] = "old"
    with runtime.lock:
        runtime._publish(None, [], 5000)
    assert runtime.snapshot()["capture_warning"] == ""

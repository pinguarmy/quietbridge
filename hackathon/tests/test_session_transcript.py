"""Camera sessions accumulate raw chunks; Stop is the only AI boundary."""
from unittest.mock import Mock, patch

from hackathon.contracts import LipResult
from hackathon.runtime import Runtime


def make_runtime():
    r = Runtime(browser_camera=True, vocabulary={})
    r.state.update(camera="ready", lips="ready")
    r.action("camera_session", {"active": True})
    return r


def test_three_computation_chunks_append_without_cleanup_or_reset():
    r = make_runtime()
    try:
        with patch.object(r.cleanup, "submit", return_value=True) as submit:
            for word, expected in [("Hi welcome", "Hi welcome"), ("to OpenAI", "Hi welcome to OpenAI"),
                                   ("DDE", "Hi welcome to OpenAI DDE")]:
                r._publish_session(LipResult([word], 12))
                r.action("start", {})
                assert r.state["session_text"] == expected
                assert r.state["decision"]["text"] == expected
                r.state.update(phase="idle", recording=False)
            submit.assert_not_called()
            assert r.state["session_chunks"] == ["Hi welcome", "to OpenAI", "DDE"]
            r.action("camera_session", {"active": False})
            submit.assert_called_once()
            assert submit.call_args.args[2] == ["Hi welcome to OpenAI DDE"]
            assert r.state["session_text"] == "Hi welcome to OpenAI DDE"
    finally:
        r.close()


def test_stop_keeps_decode_generation_and_final_tail_gets_one_cleanup():
    r = make_runtime()
    try:
        with patch.object(r.cleanup, "submit", return_value=True) as submit:
            r._publish_session(LipResult(["Hi"], 1))
            r.action("start", {})
            generation = r._record_generation
            r.action("camera_session", {"active": False})
            assert r._record_generation == generation
            assert r.state["session_finalizing"]
            submit.assert_not_called()
            r._publish_session(LipResult(["welcome"], 1))
            assert r.state["session_text"] == "Hi welcome"
            assert not r.state["session_finalizing"]
            submit.assert_called_once()
    finally:
        r.close()


def test_only_explicit_new_session_clears_text_and_no_empty_fake_result():
    r = make_runtime()
    try:
        with patch.object(r.cleanup, "submit") as submit:
            r.action("camera_session", {"active": False})
            assert r.state["session_text"] == ""
            assert r.state["decision"]["text"] == ""
            submit.assert_not_called()
            r.action("camera_session", {"active": True})
            r._publish_session(LipResult(["old words"], 1))
            r.action("camera_session", {"active": False})
            assert r.state["session_text"] == "old words"
            r.action("camera_session", {"active": True})
            assert r.state["session_text"] == "" and r.state["session_chunks"] == []
    finally:
        r.close()


def test_explicit_cancel_still_invalidates_pending_result():
    r = make_runtime()
    try:
        r.action("start", {})
        generation = r._record_generation
        r.action("cancel", {})
        assert r._record_generation > generation
        assert r.record_start is None
    finally:
        r.close()

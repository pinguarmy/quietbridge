"""Session gesture context does not extend local fusion evidence lifetime."""
from unittest.mock import Mock, patch

from hackathon.contracts import LipResult, SignEvent
from hackathon.runtime import Runtime


VOCABULARY = {"DDE": {"phrase": "DDE", "matches": ["DDE"],
                       "enabled": True, "verified": False}}


def runtime():
    result = Runtime(browser_camera=True, vocabulary=VOCABULARY)
    result.cleanup.submit = Mock(return_value=True)
    result.action("camera_session", {"active": True})
    return result


def stable_dde():
    return SignEvent("DDE", .9, 9000, 10000, True)


def test_stop_retains_validated_session_context_for_final_decode():
    r = runtime()
    try:
        r.receive_signs([stable_dde()], now=10000)
        r.state["phase"] = "recording"
        r.action("camera_session", {"active": False})
        assert not r.events
        assert r.snapshot()["session_gestures"] == ["DDE"]
        with patch("hackathon.runtime.clock_ms", return_value=20000):
            r._publish_session(LipResult(["Hi welcome to open a i"], 50))
        call = r.cleanup.submit.call_args.args
        assert call[2] == ["Hi welcome to open a i"]
        assert call[3] == ["DDE"]
        assert r.state["decision"]["text"] == "Hi welcome to open a i"
        assert r.state["decision"]["reason"] == "lip_only"
        assert r.session_signs[0].end_ms == 10000  # Never re-time old evidence.
    finally:
        r.close()


def test_lips_only_never_sends_gesture_labels_to_cleanup():
    r = runtime()
    try:
        r.state["mode"] = "lips"
        r.receive_signs([stable_dde()], now=10000)
        r.action("camera_session", {"active": False})
        r._publish_session(LipResult(["Please wait"], 25))
        assert r.cleanup.submit.call_args.args[3] == []
        assert r.state["decision"]["text"] == "Please wait"
    finally:
        r.close()


def test_invalid_events_and_duplicates_cannot_enter_session_context():
    r = runtime()
    try:
        invalid = [SignEvent("UNKNOWN", .99, 9000, 10000, True),
                   SignEvent("DDE", .74, 9000, 10000, True),
                   SignEvent("DDE", .99, 9000, 10000, False),
                   SignEvent("DDE", .99, 1000, 2000, True)]
        r.receive_signs(invalid + [stable_dde(), stable_dde()], now=10000)
        assert r.state["session_gestures"] == ["DDE"]
        assert len(r.session_signs) == 1
    finally:
        r.close()


def test_next_chunk_retains_context_but_explicit_new_session_clears_it():
    r = runtime()
    try:
        r.receive_signs([stable_dde()], now=10000)
        r.cleanup.submit.reset_mock()
        r._publish_session(LipResult(["First chunk"], 25))
        r._publish_session(LipResult(["Second chunk"], 25))
        assert r.state["session_gestures"] == ["DDE"]
        r.cleanup.submit.assert_not_called()  # No AI request while camera is active.
        r.action("camera_session", {"active": False})
        assert r.cleanup.submit.call_args.args[2:4] == (["First chunk Second chunk"], ["DDE"])
        r.action("camera_session", {"active": True})
        assert r.session_signs == []
        assert r.state["session_gestures"] == []
    finally:
        r.close()


def test_current_valid_events_still_participate_in_local_fusion():
    r = runtime()
    try:
        r.receive_signs([stable_dde()], now=10000)
        with patch("hackathon.runtime.clock_ms", return_value=10001):
            r._publish_session(LipResult(["DDE"], 25))
        assert r.state["decision"]["reason"] == "agreement"
        # Once the original timestamp expires it cannot support local agreement.
        with patch("hackathon.runtime.clock_ms", return_value=20000):
            r._publish_session(None)
        assert r.state["decision"]["reason"] == "lip_only"
    finally:
        r.close()

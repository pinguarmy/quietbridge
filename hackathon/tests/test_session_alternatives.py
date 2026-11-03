"""Session alternatives retain the other computation windows."""
from unittest.mock import Mock, patch

import pytest

from hackathon.contracts import LipResult, SignEvent
from hackathon.runtime import Runtime


def test_displayed_automatic_fusion_result_can_be_confirmed():
    vocabulary = {"WATER": {"phrase": "water", "matches": ["water"], "enabled": True,
                              "verified": True, "intent_template": "I need {gesture}"}}
    runtime = Runtime(browser_camera=True, vocabulary=vocabulary)
    runtime.cleanup.submit = Mock(return_value=False)
    runtime.speaker.speak = Mock()
    try:
        with patch("hackathon.runtime.clock_ms", return_value=2000):
            runtime.action("camera_session", {"active": True})
            runtime.events.append(SignEvent("WATER", .9, 1100, 1800, True))
            runtime._publish_session(LipResult(["I need"], 1))
            runtime.action("camera_session", {"active": False})
        assert runtime.state["decision"]["text"] == "I need water"
        assert "I need water" in runtime.state["alternatives"]
        runtime.action("confirm", {"revision": runtime.revision, "text": "I need water"})
        assert runtime.state["decision"]["status"] == "accepted"
        runtime.speaker.speak.assert_not_called()
    finally:
        runtime.close()


@pytest.mark.parametrize("explicit", [True, False])
def test_early_alternative_keeps_later_window(explicit):
    runtime = Runtime(browser_camera=True, vocabulary={})
    runtime.cleanup.submit = Mock(return_value=False)
    try:
        runtime.action("camera_session", {"active": True})
        runtime.action("mode", {"mode": "lips"})
        for candidates in (["hello", "yellow"], ["world", "word"]):
            runtime._publish_session(LipResult(candidates, 1), [] if explicit else None)
        expected = ["hello world", "yellow world", "hello word"]
        assert runtime.state["alternatives"] == expected
        runtime.action("cancel", {})
        assert runtime.state["alternatives"] == expected
        runtime.action("mode", {"mode": "combined"})
        assert runtime.state["alternatives"] == expected
        runtime.action("camera_session", {"active": False})
        assert runtime.state["decision"]["text"] == "hello world"
        assert runtime.state["alternatives"] == expected
        revision = runtime.revision
        runtime.speaker.speak = Mock()
        runtime.action("confirm", {"revision": revision, "text": "yellow world"})
        assert runtime.state["decision"]["text"] == "yellow world"
        runtime.speaker.speak.assert_not_called()
        with pytest.raises(ValueError):
            runtime.action("speak", {"revision": revision - 1, "text": "yellow world"})
        runtime.action("camera_session", {"active": True})
        assert runtime.state["alternatives"] == []
    finally:
        runtime.close()

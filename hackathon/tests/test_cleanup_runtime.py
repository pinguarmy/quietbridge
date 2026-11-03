"""Offline checks for revision-bound cleanup and local settings routes."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from hackathon.runtime import Runtime
from hackathon.server import make_server


class DeferredCleanup:
    def __init__(self):
        self.jobs = []

    def public_config(self):
        return {"enabled": True}

    def submit(self, generation, revision, candidates, labels, callback):
        self.jobs.append((generation, revision, candidates, labels, callback))
        return True

    def finish(self, index, *, suggestion="", error=""):
        generation, revision, _, _, callback = self.jobs[index]
        callback(generation, revision, {"status": "error" if error else "ready",
                                       "suggestion": suggestion, "error": error,
                                       "latency_ms": 12})

    def close(self):
        pass


class CleanupRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cleanup = DeferredCleanup()
        with patch("hackathon.runtime.CleanupClient", return_value=self.cleanup):
            self.runtime = Runtime(rehearsal=True, vocabulary={})
        self.addCleanup(self.runtime.close)

    def publish(self, text):
        self.runtime.action("rehearse", {"text": text})
        return self.runtime.snapshot()

    def test_success_is_an_alternative_not_an_automatic_decision(self):
        original = self.publish("hello there")
        self.assertEqual(len(self.cleanup.jobs), 1)
        self.assertIn("hello there", self.cleanup.jobs[0][2])
        self.cleanup.finish(0, suggestion="Hello there.")
        state = self.runtime.snapshot()
        self.assertEqual(state["cleanup"]["status"], "ready")
        self.assertEqual(state["cleanup"]["source_revision"], original["revision"])
        self.assertIn("Hello there.", state["alternatives"])
        self.assertEqual(state["decision"], original["decision"])
        self.assertFalse(state["confirmed"])
        self.runtime.action("confirm", {"revision": state["revision"], "text": "Hello there."})
        self.assertEqual(self.runtime.snapshot()["decision"]["text"], "Hello there.")
        self.assertTrue(self.runtime.snapshot()["confirmed"])

    def test_late_result_does_not_change_new_revision(self):
        self.publish("first")
        newer = self.publish("second")
        self.cleanup.finish(0, suggestion="First.")
        state = self.runtime.snapshot()
        self.assertEqual(state["revision"], newer["revision"])
        self.assertNotIn("First.", state["alternatives"])
        self.assertNotEqual(state["cleanup"]["suggestion"], "First.")
        self.cleanup.finish(1, suggestion="Second.")
        self.assertIn("Second.", self.runtime.snapshot()["alternatives"])

    def test_disabling_removes_unselected_api_alternative(self):
        self.publish("Please wait")
        self.cleanup.finish(0, suggestion="Please wait.")
        with patch.object(self.cleanup, "configure", create=True, return_value={"enabled": False}):
            self.runtime.configure_cleanup({"enabled": False})
        state = self.runtime.snapshot()
        self.assertEqual(state["cleanup"]["status"], "disabled")
        self.assertNotIn("Please wait.", state["alternatives"])
        self.assertIn("Please wait", state["alternatives"])

    def test_formats_selected_fusion_proposal_not_different_lip_candidate(self):
        from hackathon.contracts import LipResult, SignEvent
        self.runtime._fusion_vocabulary = {"WATER": {"phrase": "water", "matches": ["water"], "verified": False, "enabled": True}}
        self.runtime._publish(LipResult(["tea", "water"], 1), [SignEvent("WATER", .9, 1000, 1500, True)], 1600)
        self.assertEqual(self.runtime.state["decision"]["text"], "water")
        self.assertEqual(self.cleanup.jobs[-1][2], ["water"])

    def test_failure_preserves_local_result(self):
        original = self.publish("local text")
        self.cleanup.finish(0, error="Service unavailable")
        state = self.runtime.snapshot()
        self.assertEqual(state["cleanup"]["status"], "error")
        self.assertEqual(state["cleanup"]["error"], "Service unavailable")
        self.assertEqual(state["decision"], original["decision"])
        self.assertEqual(state["alternatives"], original["alternatives"])

    def test_confirmation_before_callback_is_not_overwritten(self):
        state = self.publish("local text")
        self.runtime.action("confirm", {"revision": state["revision"], "text": "local text"})
        self.cleanup.finish(0, suggestion="Local text.")
        current = self.runtime.snapshot()
        self.assertEqual(current["decision"]["text"], "local text")
        self.assertTrue(current["confirmed"])


class CleanupWorkerTests(unittest.TestCase):
    def test_worker_success_and_cancelled_result_are_revision_bound(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                runtime = Runtime(rehearsal=True, vocabulary={})
                entered, release, finished = threading.Event(), threading.Event(), threading.Event()
                worker = runtime.cleanup._worker
                original_callback = runtime._cleanup_result

                def chat(config, prompt):
                    entered.set()
                    if not release.wait(2):
                        raise ValueError("Test timed out waiting for release")
                    return "local text."

                def callback(*args):
                    try:
                        original_callback(*args)
                    finally:
                        finished.set()

                try:
                    runtime.cleanup.configure({"base_url": "https://example.com", "model": "demo",
                                               "api_key": "test-key", "enabled": True})
                    with patch.object(runtime.cleanup, "_chat", side_effect=chat), \
                         patch.object(runtime, "_cleanup_result", side_effect=callback):
                        runtime.action("rehearse", {"text": "local text"})
                        self.assertTrue(entered.wait(2), "worker did not start")
                        self.assertEqual(runtime.snapshot()["cleanup"]["status"], "pending")
                        if cancelled:
                            runtime.action("cancel", {})
                        release.set()
                        self.assertTrue(finished.wait(2), "worker did not finish")
                        state = runtime.snapshot()
                        if cancelled:
                            self.assertNotIn("local text.", state["alternatives"])
                            self.assertEqual(state["cleanup"]["suggestion"], "")
                        else:
                            self.assertIn("local text.", state["alternatives"])
                            self.assertEqual(state["cleanup"]["status"], "ready")
                finally:
                    release.set()
                    runtime.close()
                    worker.join(2)
                self.assertFalse(worker.is_alive(), "cleanup worker did not close")


class CleanupRoutesTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(rehearsal=True, vocabulary={})
        self.server = make_server(self.runtime, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.token = self.request("/api/state")["token"]
        self.addCleanup(self.runtime.close)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.shutdown)

    def request(self, path, data=None, token=None):
        headers = {}
        if data is not None:
            headers = {"Content-Type": "application/json", "X-Lipflow-Token": token or self.token}
            data = json.dumps(data).encode()
        with urlopen(Request(self.base + path, data=data, headers=headers), timeout=2) as response:
            return json.loads(response.read())

    def test_config_roundtrip_redacts_key_and_test_is_mocked(self):
        config = self.request("/api/cleanup/config", {"base_url": "https://example.com", "model": "demo",
                                                       "api_key": "private-key", "enabled": True})
        self.assertTrue(config["configured"])
        self.assertNotIn("private-key", json.dumps(config))
        saved = self.request("/api/cleanup/config")
        self.assertTrue(saved["enabled"])
        self.assertNotIn("private-key", json.dumps(saved))
        with patch.object(self.runtime.cleanup, "_chat", return_value="OK") as chat:
            self.assertTrue(self.request("/api/cleanup/test", {})["ok"])
            chat.assert_called_once()

    def test_settings_mutation_requires_token(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/cleanup/config", {"enabled": False}, token="wrong")
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()

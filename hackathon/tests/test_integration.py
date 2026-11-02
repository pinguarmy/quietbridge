"""Run without model dependencies: python -m unittest discover -s hackathon/tests -v."""
import json
from pathlib import Path
import queue
import threading
import time
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch

from hackathon.contracts import LipResult, SignEvent
from hackathon.fusion import fuse, load_vocabulary
from hackathon.runtime import Runtime, put_latest
from hackathon.runtime import clock_ms as real_clock_ms
from hackathon.server import make_server

VOCAB = {
    "HELP": {"phrase": "I need help.", "matches": ["help"], "verified": True},
    "WATER": {"phrase": "I want water.", "matches": ["water"], "verified": True},
    "NO": {"phrase": "No.", "matches": ["no"], "verified": True},
}


def sign(label="HELP", score=.95, start=1000, end=1500, final=True):
    return SignEvent(label, score, start, end, final)


class FusionTests(unittest.TestCase):
    def test_no_input_never_generates_text(self):
        self.assertEqual(fuse(None, [], VOCAB, 2000).status, "unavailable")

    def test_fixed_sign_phrase(self):
        d = fuse(None, [sign()], VOCAB, 2000)
        self.assertEqual((d.text, d.status, d.reason), ("I need help.", "accepted", "sign_only"))

    def test_shared_vocabulary_matches_published_recognizer_labels(self):
        vocabulary = load_vocabulary()
        declared = json.loads((Path(__file__).parents[1] / "signs" / "labels.json").read_text())["labels"]
        self.assertTrue(vocabulary)
        self.assertTrue(set(vocabulary) <= set(declared))
        for label, entry in vocabulary.items():
            # Explicit synthetic verification tests wiring, not live accuracy.
            enabled = {label: {**entry, "verified": True}}
            decision = fuse(None, [sign(label)], enabled, 2000)
            self.assertEqual((decision.text, decision.status), (entry["phrase"], "accepted"))

    def test_bad_or_unverified_events_do_not_become_speech(self):
        # MotionModel applies its own calibrated acceptance before emission.
        # Stock MediaPipe still applies its 0.75 floor inside SignRecognizer.
        invalid = [sign("UNKNOWN"), sign(score=-.1), sign(score=float("nan")), sign(score=1.1),
                   sign(final=False), sign(start=2000, end=1500), sign(end=6000), sign(end=100)]
        for event in invalid:
            with self.subTest(event=event):
                self.assertEqual(fuse(None, [event], VOCAB, 5000, since_ms=1000).status, "unavailable")
        self.assertEqual(fuse(None, [sign()], {"HELP": {**VOCAB["HELP"], "verified": False}}, 2000).status, "unavailable")

    def test_agreement_reranks_candidates_but_still_confirms_sentence(self):
        d = fuse(LipResult(["I want water", "I need help"], 20), [sign()], VOCAB, 2000)
        self.assertEqual((d.text, d.reason, d.status), ("I need help", "agreement", "confirmation_needed"))

    def test_negation_and_numbers_cannot_be_erased_by_concept_match(self):
        for text in ["I do not need help", "I don't need help", "I need 2 help sessions", "I need two helpers"]:
            with self.subTest(text=text):
                d = fuse(LipResult([text], 10), [sign()], VOCAB, 2000)
                self.assertEqual((d.reason, d.status), ("conflict", "confirmation_needed"))

    def test_intent_template_requires_exact_static_context(self):
        vocab = {"WATER": {"phrase": "water", "matches": ["water"], "verified": True,
                           "intent_template": "I want {gesture}, please."}}
        for candidate in ["i WANT, please.", " I   want,   please. "]:
            with self.subTest(candidate=candidate):
                d = fuse(LipResult([candidate], 0), [sign("WATER")], vocab, 2000)
                self.assertEqual((d.text, d.status), ("I want water, please.", "confirmation_needed"))
        for candidate in ["I want coffee, please.", "I want, please", "I want water, please.",
                          "I do not want, please.", "I want 2, please."]:
            with self.subTest(candidate=candidate):
                d = fuse(LipResult([candidate], 0), [sign("WATER")], vocab, 2000)
                self.assertEqual(d.text, candidate)
                self.assertEqual(d.status, "confirmation_needed")

    def test_template_conflicts_do_not_compose(self):
        vocab = {"WATER": {"phrase": "water", "matches": ["water"], "verified": True,
                           "intent_template": "I want {gesture}"},
                 "HELP": {"phrase": "help", "matches": ["help"], "verified": True,
                          "intent_template": "I want {gesture}"}}
        d = fuse(LipResult(["I want"], 0), [sign("WATER"), sign("HELP")], vocab, 2000)
        self.assertEqual((d.text, d.reason, d.status), ("I want", "conflict", "confirmation_needed"))
        vocab["WATER"]["intent_template"] = ""
        d = fuse(LipResult(["I want"], 0), [sign("WATER")], vocab, 2000)
        self.assertEqual((d.text, d.reason), ("I want", "conflict"))

    def test_runtime_keeps_original_template_candidate_as_alternative(self):
        vocab = {"WATER": {"phrase": "water", "matches": ["water"], "verified": True,
                           "intent_template": "I want {gesture}"}}
        runtime = Runtime(rehearsal=True, vocabulary=vocab)
        try:
            runtime._publish(LipResult(["I want"], 0), [sign("WATER")], 2000)
            state = runtime.snapshot()
            self.assertEqual(state["decision"]["text"], "I want water")
            self.assertEqual(state["alternatives"], ["I want water", "I want", "water"])
            self.assertEqual(state["decision"]["status"], "confirmation_needed")
        finally:
            runtime.close()

    def test_no_is_not_found_inside_know(self):
        d = fuse(LipResult(["I know"], 0), [sign("NO")], VOCAB, 2000)
        self.assertEqual(d.reason, "conflict")

    def test_supported_gesture_sequence_composes_in_time_order(self):
        later_water = sign("WATER", start=1600, end=1800)
        d = fuse(None, [later_water, sign()], VOCAB, 2000)
        self.assertEqual((d.text, d.status, d.matched_signs),
                         ("I need help. I want water.", "accepted", ["HELP", "WATER"]))

    def test_multiple_gesture_agreement_and_partial_agreement_remain_reviewed(self):
        events = [sign(), sign("WATER", start=1600, end=1800)]
        both = fuse(LipResult(["I need help and I want water"], 0), events, VOCAB, 2000)
        self.assertEqual((both.reason, both.status, both.matched_signs),
                         ("agreement", "confirmation_needed", ["HELP", "WATER"]))
        partial = fuse(LipResult(["I need help"], 0), events, VOCAB, 2000)
        self.assertEqual((partial.reason, partial.status, partial.matched_signs),
                         ("partial_agreement", "confirmation_needed", ["HELP"]))

    def test_explicit_long_window_keeps_early_gesture(self):
        early = sign(start=1000, end=1500)
        d = fuse(None, [early], VOCAB, 30000, since_ms=1000)
        self.assertEqual((d.text, d.reason), ("I need help.", "sign_only"))

    def test_lip_only_has_no_claim_of_confidence(self):
        self.assertEqual(fuse(LipResult(["Hello"], 9), [], VOCAB, 2000).status, "confirmation_needed")

    def test_enabled_unverified_gestures_require_confirmation(self):
        vocabulary = load_vocabulary()
        for label, entry in vocabulary.items():
            with self.subTest(label=label):
                self.assertTrue(entry["enabled"])
                self.assertFalse(entry["verified"])
                decision = fuse(None, [sign(label)], vocabulary, 2000)
                self.assertEqual((decision.text, decision.status), (entry["phrase"], "confirmation_needed"))

    def test_disabled_mapping_overrides_verified_flag(self):
        vocabulary = {"HELP": {**VOCAB["HELP"], "enabled": False}}
        self.assertEqual(fuse(None, [sign()], vocabulary, 2000).status, "unavailable")

    def test_yes_and_no_do_not_agree_with_opposite_lip_candidates(self):
        vocabulary = load_vocabulary()
        for label, candidate in [("THUMB_UP", "No."), ("THUMB_DOWN", "Yes.")]:
            with self.subTest(label=label):
                decision = fuse(LipResult([candidate], 0), [sign(label)], vocabulary, 2000)
                self.assertEqual((decision.reason, decision.status), ("conflict", "confirmation_needed"))

    def test_shared_mapping_has_no_conflicting_phrase_aliases(self):
        vocabulary = load_vocabulary()
        aliases = {}
        for label, entry in vocabulary.items():
            for alias in entry["matches"]:
                normalized = alias.casefold().strip()
                self.assertNotIn(normalized, aliases, f"{label} overlaps {aliases.get(normalized)}")
                aliases[normalized] = label

    def test_enabled_flag_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps({"HELP": {**VOCAB["HELP"], "enabled": "yes"}}))
            with self.assertRaises(ValueError):
                load_vocabulary(path)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.r = Runtime(rehearsal=True, vocabulary=VOCAB)
        self.r.speaker = Mock(available=True)

    def tearDown(self):
        self.r.close()

    def test_latest_frame_queue_drops_old_frames(self):
        q = queue.Queue(maxsize=1)
        put_latest(q, 1); put_latest(q, 2); put_latest(q, 3)
        self.assertEqual(q.get_nowait(), 3)

    def test_late_record_release_does_not_change_a_finished_or_cancelled_recording(self):
        live = Runtime(vocabulary=VOCAB)
        live.state.update(camera="ready", lips="ready")
        try:
            live.action("start", {})
            live.action("stop", {})
            end = live.finish_at
            live.action("stop", {})
            self.assertEqual(live.finish_at, end)
            self.assertEqual(live.snapshot()["phase"], "tail")
            live.action("cancel", {})
            revision = live.revision
            live.action("stop", {})
            self.assertEqual((live.snapshot()["phase"], live.revision), ("idle", revision))
        finally:
            live.close()

    def test_uncertain_message_cannot_speak_until_confirmed(self):
        self.r.action("rehearse", {"text": "I want water", "label": "HELP"})
        rev = self.r.revision
        with self.assertRaises(ValueError):
            self.r.action("speak", {"revision": rev, "text": "I want water"})
        self.r.speaker.speak.assert_not_called()
        self.r.action("confirm", {"text": "I need help.", "revision": rev})
        self.r.action("speak", {"revision": rev, "text": "I need help."})
        self.r.speaker.speak.assert_called_once_with("I need help.")

    def test_another_tab_cannot_change_the_words_of_a_speak_request(self):
        self.r.action("rehearse", {"text": "I want water", "label": "HELP"})
        rev = self.r.revision
        self.r.action("confirm", {"text": "I want water", "revision": rev})
        # Another tab chooses the other alternative before the first tab speaks.
        self.r.action("confirm", {"text": "I need help.", "revision": rev})
        with self.assertRaises(ValueError):
            self.r.action("speak", {"revision": rev, "text": "I want water"})
        self.r.speaker.speak.assert_not_called()
        self.r.action("speak", {"revision": rev, "text": "I need help."})
        self.r.speaker.speak.assert_called_once_with("I need help.")

    def test_new_result_invalidates_old_confirmation_and_speak(self):
        self.r.action("rehearse", {"text": "Hello"})
        old = self.r.revision
        self.r.action("confirm", {"text": "Hello", "revision": old})
        self.r.action("rehearse", {"text": "Goodbye"})
        for action in ["confirm", "speak"]:
            with self.assertRaises(ValueError):
                self.r.action(action, {"text": "Hello", "revision": old})
        self.r.speaker.speak.assert_not_called()

    def test_cannot_confirm_invented_text(self):
        self.r.action("rehearse", {"text": "Hello"})
        with self.assertRaises(ValueError):
            self.r.action("confirm", {"text": "Different meaning", "revision": self.r.revision})

    def test_duplicate_event_does_not_republish(self):
        self.r.receive_signs([sign()], now=2000)
        rev = self.r.revision
        self.r.receive_signs([sign()], now=2001)
        self.assertEqual(self.r.revision, rev)

    def test_synthetic_endpoint_disabled_in_live_mode(self):
        live = Runtime(vocabulary=VOCAB)
        with self.assertRaises(ValueError):
            live.action("rehearse", {"text": "Made up"})
        live.close()


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Runtime(rehearsal=True, vocabulary=VOCAB)
        cls.server = make_server(cls.r, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.r.close()

    def post(self, path, data, token, **headers):
        request = Request(self.url + path, data=json.dumps(data).encode(),
                          headers={"Content-Type": "application/json", "X-Lipflow-Token": token, **headers})
        return urlopen(request, timeout=3)

    def test_full_http_confirmation_flow(self):
        with urlopen(self.url + "/api/state") as res:
            initial = json.load(res)
        with self.post("/api/action/rehearse", {"text": "I want water", "label": "HELP"}, initial["token"]) as res:
            current = json.load(res)
        self.assertEqual(current["decision"]["status"], "confirmation_needed")
        with self.post("/api/action/confirm", {"text": "I want water", "revision": current["revision"]}, initial["token"]) as res:
            self.assertTrue(json.load(res)["confirmed"])

    def test_other_origins_and_missing_token_are_rejected(self):
        with urlopen(self.url + "/api/state") as res:
            token = json.load(res)["token"]
        for value, headers in [("", {}), (token, {"Origin": "https://example.com"})]:
            with self.assertRaises(HTTPError) as err:
                self.post("/api/action/cancel", {}, value, **headers)
            self.assertEqual(err.exception.code, 403)

    def test_invalid_host_and_path_are_rejected(self):
        with self.assertRaises(HTTPError) as err:
            urlopen(Request(self.url + "/api/state", headers={"Host": "other.example"}))
        self.assertEqual(err.exception.code, 403)
        with self.assertRaises(HTTPError) as err:
            urlopen(self.url + "/../pyproject.toml")
        self.assertEqual(err.exception.code, 404)

    def test_communication_and_settings_pages_are_served(self):
        for path in ["/", "/settings"]:
            with self.subTest(path=path), urlopen(self.url + path) as res:
                self.assertIn("text/html", res.headers["Content-Type"])
                self.assertIn(b"QuietBridge", res.read())


class WorkerTests(unittest.TestCase):
    def wait_for(self, condition):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if condition():
                return
            time.sleep(.005)
        self.fail("Worker did not reach expected state")

    def run_worker_case(self, *, fail=False, cancel=False):
        r = Runtime(vocabulary=VOCAB)
        r.state["camera"] = "ready"
        decoding, proceed = threading.Event(), threading.Event()

        class Adapter:
            recording = None

            def start(self):
                self.recording = object()

            def process(self, *args):
                pass

            def finish(self):
                decoding.set()
                if not proceed.wait(3):
                    raise RuntimeError("Test timed out")
                if fail:
                    raise ValueError("No visible face")
                return LipResult(["I need help"], 7)

            def close(self):
                pass

        offset = [0]
        with patch("hackathon.lips_adapter.LipsAdapter", Adapter), patch("hackathon.runtime.clock_ms", side_effect=lambda: real_clock_ms() + offset[0]):
            thread = threading.Thread(target=r._lips, daemon=True)
            r.threads.append(thread)
            thread.start()
            try:
                self.wait_for(lambda: r.snapshot()["lips"] == "ready")
                r.action("start", {})
                now = real_clock_ms()
                r.receive_signs([SignEvent("HELP", .95, now, now, True)], now=now)
                r.action("stop", {})
                self.assertTrue(decoding.wait(2))
                if cancel:
                    r.action("cancel", {})
                    revision = r.revision
                # Simulate a long decode: capture-window events must remain aligned.
                offset[0] = 6000
                proceed.set()
                if cancel:
                    self.wait_for(lambda: r.snapshot()["phase"] == "idle")
                    r.stop.set(); thread.join(1)
                    self.assertEqual(r.revision, revision)
                    self.assertEqual(r.snapshot()["decision"]["status"], "unavailable")
                else:
                    self.wait_for(lambda: r.snapshot()["phase"] == "idle")
                    decision = r.snapshot()["decision"]
                    self.assertEqual(decision["reason"], "sign_only" if fail else "agreement")
                    self.assertEqual(decision["text"], "I need help." if fail else "I need help")
            finally:
                proceed.set()
                r.close()

    def test_slow_decode_retains_signs_from_capture_window(self):
        self.run_worker_case()

    def test_failed_lip_decode_preserves_supported_sign(self):
        self.run_worker_case(fail=True)

    def test_cancelled_decode_cannot_publish_late_result(self):
        self.run_worker_case(cancel=True)


if __name__ == "__main__":
    unittest.main()

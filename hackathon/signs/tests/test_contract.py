"""Real recognizer/stabilizer/fusion wiring with synthetic backend predictions."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from hackathon.contracts import SignEvent
from hackathon.fusion import fuse, load_vocabulary
from hackathon.signs import recognizer as module
from hackathon.signs.recognizer import SignRecognizer, SUPPORTED_LABELS


def test_handoff_vocabulary_matches_real_labels_but_stays_unverified():
    path = Path(__file__).parents[1] / "vocabulary.example.json"
    vocabulary = load_vocabulary(path)
    assert set(vocabulary) == set(SUPPORTED_LABELS)
    assert all(entry["verified"] is False for entry in vocabulary.values())


def test_default_stabilizer_returns_flat_contract_events(tmp_path, monkeypatch):
    model = tmp_path / "fake.task"
    model.write_bytes(b"fake model, backend is patched")
    result = SimpleNamespace(gestures=[[SimpleNamespace(category_name="Thumb_Up", score=.9)]])
    backend = SimpleNamespace(recognize_for_video=lambda image, ts: result, close=lambda: None)
    monkeypatch.setattr(module.vision.GestureRecognizer, "create_from_options", lambda options: backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with SignRecognizer(model_path=model) as recognizer:
        for ts in (1000, 1100, 1200):
            assert recognizer.process(frame, ts) == []
        events = recognizer.process(frame, 1300)
        assert events == [SignEvent("THUMB_UP", .9, 1000, 1300, True)]
        assert all(isinstance(event, SignEvent) for event in events)
        assert recognizer.process(frame, 1400) == []
    vocabulary = {"THUMB_UP": {"phrase": "Yes.", "matches": ["yes"], "verified": False}}
    assert fuse(None, events, vocabulary, 1300).status == "unavailable"
    # Explicitly verified synthetic mapping tests integration, not live QA.
    vocabulary["THUMB_UP"]["verified"] = True
    decision = fuse(None, events, vocabulary, 1300)
    assert (decision.text, decision.status, decision.reason) == ("Yes.", "accepted", "sign_only")


def test_sub_fusion_scores_do_not_lock_out_later_valid_gesture(tmp_path, monkeypatch):
    model = tmp_path / "fake.task"
    model.write_bytes(b"fake model, backend is patched")
    category = SimpleNamespace(category_name="Thumb_Up", score=.74)
    result = SimpleNamespace(gestures=[[category]])
    backend = SimpleNamespace(recognize_for_video=lambda image, ts: result, close=lambda: None)
    monkeypatch.setattr(module.vision.GestureRecognizer, "create_from_options", lambda options: backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with SignRecognizer(model_path=model) as recognizer:
        for ts in (1000, 1100, 1200, 1300):
            assert recognizer.process(frame, ts) == []
        category.score = .9
        for ts in (1400, 1500, 1600):
            assert recognizer.process(frame, ts) == []
        assert recognizer.process(frame, 1700) == [SignEvent("THUMB_UP", .9, 1400, 1700, True)]

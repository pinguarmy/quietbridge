"""Tests for hackathon.signs.recognizer.

The MediaPipe backend is faked at the ``vision.GestureRecognizer`` boundary and
the stabilizer is faked by injection; no real gesture inference is claimed or
tested here. RGB conversion and timestamp forwarding are exercised for real
through ``mp.Image``.
"""
import numpy as np
import pytest

from hackathon.contracts import SignEvent
from hackathon.signs import recognizer as rec_mod
from hackathon.signs.recognizer import (
    LABEL_MAP,
    MIN_SCORE,
    SUPPORTED_LABELS,
    SignRecognizer,
)


class FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class FakeResult:
    def __init__(self, gestures):
        self.gestures = gestures


class FakeBackend:
    """Stands in for vision.GestureRecognizer; records calls, replays results."""

    def __init__(self, results=()):
        self.calls = []  # (rgb_copy, timestamp_ms)
        self._results = list(results)
        self.closed = False

    def recognize_for_video(self, image, timestamp_ms):
        self.calls.append((image.numpy_view().copy(), timestamp_ms))
        if self._results:
            return self._results.pop(0)
        return FakeResult([])

    def close(self):
        self.closed = True


class FakeStabilizer:
    """Records update() calls and replays queued return values (event lists)."""

    def __init__(self, events=()):
        self.updates = []  # (label, score, timestamp_ms)
        self._events = list(events)

    def update(self, label, score, timestamp_ms):
        self.updates.append((label, score, timestamp_ms))
        if self._events:
            return self._events.pop(0)
        return []


def hand(name, score):
    return [FakeCategory(name, score)]


def make_recognizer(tmp_path, monkeypatch, *, results=(), events=()):
    """Build a SignRecognizer over a fake backend and fake stabilizer."""
    model = tmp_path / "gesture.task"
    model.write_bytes(b"not a real model")
    backend = FakeBackend(results)
    stabilizer = FakeStabilizer(events)
    captured = {}

    def fake_create_from_options(options):
        captured["options"] = options
        return backend

    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(fake_create_from_options),
    )
    recognizer = SignRecognizer(model_path=model, stabilizer=stabilizer)
    return recognizer, backend, stabilizer, captured


@pytest.fixture(autouse=True)
def no_implicit_personal_model(monkeypatch):
    monkeypatch.delenv("QUIETBRIDGE_PERSONAL_GESTURES", raising=False)


def frame(bgr=(10, 20, 30)):
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_public_label_contract():
    assert LABEL_MAP == {
        "Thumb_Up": "THUMB_UP",
        "Thumb_Down": "THUMB_DOWN",
        "Open_Palm": "OPEN_PALM",
        "Pointing_Up": "POINTING_UP",
    }
    assert set(SUPPORTED_LABELS) == set(LABEL_MAP.values())


def test_backend_constructed_once_with_video_mode_two_hands_no_allowlist(
    tmp_path, monkeypatch
):
    _, _, _, captured = make_recognizer(tmp_path, monkeypatch)
    options = captured["options"]
    assert options.running_mode is rec_mod.vision.RunningMode.VIDEO
    assert options.num_hands == 2
    canned = getattr(options, "canned_gesture_classifier_options", None)
    allowlist = getattr(canned, "category_allowlist", None) if canned else None
    assert not allowlist


def test_missing_model_raises_with_download_hint(tmp_path, monkeypatch):
    def fail_create(options):  # pragma: no cover - must not be reached
        raise AssertionError("backend must not be constructed for a missing model")

    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(fail_create),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        SignRecognizer(model_path=tmp_path / "absent.task")
    assert "python -m hackathon.signs.assets --download" in str(excinfo.value)


def test_default_construction_uses_real_assets_and_stabilizer(tmp_path, monkeypatch):
    """Default ctor wires .assets.default_model_path and .stabilizer.EventStabilizer."""
    from hackathon.signs.stabilizer import EventStabilizer

    model = tmp_path / "gesture.task"
    model.write_bytes(b"not a real model")
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", str(model))
    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(lambda options: FakeBackend()),
    )

    recognizer = SignRecognizer()
    try:
        assert isinstance(recognizer._stabilizer, EventStabilizer)
        assert recognizer._stabilizer.min_score == MIN_SCORE
    finally:
        recognizer.close()


def test_default_construction_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", str(tmp_path / "absent.task"))
    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(lambda options: FakeBackend()),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        SignRecognizer()
    assert "python -m hackathon.signs.assets --download" in str(excinfo.value)


def test_frame_is_converted_bgr_to_rgb(tmp_path, monkeypatch):
    recognizer, backend, _, _ = make_recognizer(tmp_path, monkeypatch)
    recognizer.process(frame(bgr=(10, 20, 30)), 1000)
    rgb, _ = backend.calls[0]
    assert tuple(rgb[0, 0]) == (30, 20, 10)


def test_external_timestamp_forwarded_unchanged(tmp_path, monkeypatch):
    recognizer, backend, stabilizer, _ = make_recognizer(tmp_path, monkeypatch)
    recognizer.process(frame(), 123456)
    assert backend.calls[0][1] == 123456
    assert stabilizer.updates[0][2] == 123456


def test_single_supported_hand_reaches_stabilizer(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path, monkeypatch, results=[FakeResult([hand("Thumb_Up", 0.9)])]
    )
    assert recognizer.process(frame(), 1) == []
    assert stabilizer.updates == [("THUMB_UP", pytest.approx(0.9), 1)]


def test_low_confidence_hand_is_dropped(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[FakeResult([hand("Thumb_Up", MIN_SCORE - 0.01)])],
    )
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [(None, 0.0, 1)]


def test_unsupported_top1_is_dropped(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path, monkeypatch, results=[FakeResult([hand("Victory", 0.99)])]
    )
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [(None, 0.0, 1)]


def test_empty_result_reports_no_candidate(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(tmp_path, monkeypatch)
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [(None, 0.0, 1)]


def test_two_hands_conflict_produces_no_candidate(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[
            FakeResult([hand("Thumb_Up", 0.9), hand("Open_Palm", 0.9)]),
        ],
    )
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [(None, 0.0, 1)]


def test_two_hands_one_unsupported_produces_no_candidate(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[FakeResult([hand("Thumb_Up", 0.9), hand("Victory", 0.95)])],
    )
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [(None, 0.0, 1)]


def test_two_hands_same_label_uses_min_score(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[FakeResult([hand("Thumb_Up", 0.95), hand("Thumb_Up", 0.8)])],
    )
    recognizer.process(frame(), 1)
    assert stabilizer.updates == [("THUMB_UP", pytest.approx(0.8), 1)]


def test_stabilizer_event_is_returned_otherwise_empty(tmp_path, monkeypatch):
    event = SignEvent(
        label="OPEN_PALM", score=0.88, start_ms=1, end_ms=5, is_final=True
    )
    recognizer, _, _, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[
            FakeResult([hand("Open_Palm", 0.88)]),
            FakeResult([hand("Open_Palm", 0.88)]),
        ],
        events=[[], [event]],
    )
    assert recognizer.process(frame(), 1) == []
    assert recognizer.process(frame(), 2) == [event]


def test_repeated_frames_do_not_emit_without_stabilizer_event(tmp_path, monkeypatch):
    recognizer, _, stabilizer, _ = make_recognizer(
        tmp_path,
        monkeypatch,
        results=[FakeResult([hand("Thumb_Up", 0.9)])] * 3,
    )
    assert recognizer.process(frame(), 1) == []
    assert recognizer.process(frame(), 2) == []
    assert recognizer.process(frame(), 3) == []
    # Debounce is fully delegated: every frame is forwarded exactly once.
    assert len(stabilizer.updates) == 3


def test_non_increasing_timestamp_raises_before_backend(tmp_path, monkeypatch):
    recognizer, backend, _, _ = make_recognizer(tmp_path, monkeypatch)
    recognizer.process(frame(), 100)
    with pytest.raises(ValueError):
        recognizer.process(frame(), 100)  # equal is non-increasing
    with pytest.raises(ValueError):
        recognizer.process(frame(), 50)
    # The backend was not touched again and can recover with a later timestamp.
    assert len(backend.calls) == 1
    assert recognizer.process(frame(), 101) == []


def test_close_releases_backend_and_is_idempotent(tmp_path, monkeypatch):
    recognizer, backend, _, _ = make_recognizer(tmp_path, monkeypatch)
    recognizer.close()
    recognizer.close()
    assert backend.closed
    with pytest.raises(RuntimeError):
        recognizer.process(frame(), 1)


def test_context_manager_closes(tmp_path, monkeypatch):
    model = tmp_path / "gesture.task"
    model.write_bytes(b"not a real model")
    backend = FakeBackend()
    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(lambda options: backend),
    )
    with SignRecognizer(model_path=model, stabilizer=FakeStabilizer()):
        pass
    assert backend.closed


def test_integration_with_real_stabilizer(tmp_path, monkeypatch):
    """End-to-end over the real EventStabilizer with a faked backend.

    Verifies the wiring contract only; no real gesture inference is performed.
    """
    from hackathon.signs.stabilizer import EventStabilizer

    model = tmp_path / "gesture.task"
    model.write_bytes(b"not a real model")
    backend = FakeBackend([FakeResult([hand("Thumb_Up", 0.9)])] * 6)
    monkeypatch.setattr(
        rec_mod.vision.GestureRecognizer,
        "create_from_options",
        staticmethod(lambda options: backend),
    )
    recognizer = SignRecognizer(
        model_path=model, stabilizer=EventStabilizer(min_score=MIN_SCORE)
    )
    outputs = [recognizer.process(frame(), ts) for ts in range(0, 600, 100)]

    events = [event for output in outputs for event in output]
    assert len(events) == 1
    assert events[0].label == "THUMB_UP"
    assert events[0].is_final
    assert events[0].score == pytest.approx(0.9)
    assert events[0].start_ms == 0
    assert events[0].end_ms <= 500
    # No duplicate emission while the same gesture is held.
    assert outputs[-1] == []

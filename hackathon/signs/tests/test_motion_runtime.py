"""Explicit motion windows using a fake MediaPipe backend and a real NPZ model."""
from types import SimpleNamespace

import numpy as np
import pytest

from hackathon.signs import recognizer as rec
from hackathon.signs.motion import MotionModel, N_FRAMES


class Backend:
    def __init__(self):
        self.closed = False

    def recognize_for_video(self, image, timestamp):
        points = [SimpleNamespace(x=0.1, y=0.2 + i * 0.01, z=0.0) for i in range(21)]
        return SimpleNamespace(hand_landmarks=[points], gestures=[[]],
                               handedness=[[SimpleNamespace(category_name="Left")]])

    def close(self):
        self.closed = True


def make_model(path, label="CUSTOM"):
    model = MotionModel()
    model.templates_ = np.zeros((1, N_FRAMES, 2, 21, 3), dtype=np.float32)
    model.masks_ = np.ones((1, N_FRAMES, 2), dtype=bool)
    model.labels_ = np.array([label])
    model.reject_threshold_ = 100.0
    model.margin_threshold_ = 0.0
    model.save(path)


def test_window_emits_once_below_point75_and_ignores_personal(tmp_path, monkeypatch):
    task = tmp_path / "model.task"
    task.touch()
    motion = tmp_path / "motion.npz"
    make_model(motion)
    backend = Backend()
    monkeypatch.setattr(rec.vision.GestureRecognizer, "create_from_options", lambda _: backend)
    monkeypatch.setenv("QUIETBRIDGE_PERSONAL_GESTURES", str(tmp_path / "missing.npz"))
    recognizer = rec.SignRecognizer(task, motion_model_path=motion)
    # Isolate runtime policy from score calibration; the model accepted this match.
    recognizer._motion.predict = lambda trial: ("CUSTOM", 0.6)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert recognizer.process(image, 10) == []
    assert recognizer.finish(10) == []
    recognizer.start(11)
    assert recognizer.process(image, 12) == []
    assert recognizer.process(image, 13) == []
    events = recognizer.finish(14)
    assert len(events) == 1
    assert (events[0].label, events[0].score, events[0].start_ms, events[0].end_ms, events[0].is_final) == ("CUSTOM", 0.6, 11, 14, True)
    assert recognizer.finish(15) == []
    recognizer.close()
    assert backend.closed
    with pytest.raises(RuntimeError):
        recognizer.start(16)


def test_real_model_rest_and_invalid_windows_are_silent(tmp_path, monkeypatch):
    task = tmp_path / "model.task"
    task.touch()
    motion = tmp_path / "motion.npz"
    make_model(motion, "REST")
    monkeypatch.setattr(rec.vision.GestureRecognizer, "create_from_options", lambda _: Backend())
    recognizer = rec.SignRecognizer(task, motion_model_path=motion)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    recognizer.start(10)
    recognizer.process(image, 11)
    assert recognizer.finish(12) == []
    recognizer.start(13)
    recognizer.process(image, 14)
    recognizer.process(image, 15)
    with pytest.raises(ValueError):
        recognizer.finish(14)
    assert recognizer.finish(16) == []
    recognizer.close()


def test_cancel_discards_window_and_preserves_monotonic_boundary(tmp_path, monkeypatch):
    task = tmp_path / "model.task"
    task.touch()
    motion = tmp_path / "motion.npz"
    make_model(motion)
    monkeypatch.setattr(rec.vision.GestureRecognizer, "create_from_options", lambda _: Backend())
    recognizer = rec.SignRecognizer(task, motion_model_path=motion)
    calls = []
    recognizer._motion.predict = lambda trial: (calls.append(trial) or ("CUSTOM", 0.6))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    recognizer.start(10)
    recognizer.process(image, 11)
    recognizer.process(image, 12)
    assert recognizer.cancel() is None
    assert recognizer.finish(13) == []
    assert calls == []
    with pytest.raises(ValueError):
        recognizer.start(11)
    recognizer.start(12)
    recognizer.process(image, 13)
    recognizer.process(image, 14)
    assert recognizer.finish(15)[0].label == "CUSTOM"
    assert len(calls) == 1
    recognizer.cancel()
    recognizer.close()


def test_cancel_is_noop_for_stock_mode(tmp_path, monkeypatch):
    task = tmp_path / "model.task"
    task.touch()
    monkeypatch.delenv("QUIETBRIDGE_MOTION_MODEL", raising=False)
    monkeypatch.delenv("QUIETBRIDGE_PERSONAL_GESTURES", raising=False)
    monkeypatch.setattr(rec.vision.GestureRecognizer, "create_from_options", lambda _: Backend())
    recognizer = rec.SignRecognizer(task)
    assert recognizer.cancel() is None
    recognizer.close()


def test_motion_environment_selection(tmp_path, monkeypatch):
    task = tmp_path / "model.task"
    task.touch()
    motion = tmp_path / "motion.npz"
    make_model(motion)
    monkeypatch.setenv("QUIETBRIDGE_MOTION_MODEL", str(motion))
    monkeypatch.setattr(rec.vision.GestureRecognizer, "create_from_options", lambda _: Backend())
    recognizer = rec.SignRecognizer(task)
    assert recognizer._motion is not None
    recognizer.close()

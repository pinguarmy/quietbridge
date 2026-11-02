"""Deterministic wrist streams through the recognizer and its fake backend."""
from types import SimpleNamespace

import numpy as np
import pytest

from hackathon.signs import recognizer as rec
from hackathon.signs.motion_auto import MotionAutoConfig
from hackathon.signs.tests.test_motion_runtime import make_model


class Backend:
    def __init__(self):
        self.x = 0.1
        self.present = True
        self.closed = False

    def recognize_for_video(self, image, timestamp):
        points = [SimpleNamespace(x=self.x, y=0.2 + i * 0.01, z=0.0) for i in range(21)]
        return SimpleNamespace(hand_landmarks=[points] if self.present else [],
                               gestures=[[]] if self.present else [],
                               handedness=[[SimpleNamespace(category_name='Left')]] if self.present else [])

    def close(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    task = tmp_path / 'model.task'
    task.touch()
    model = tmp_path / 'model.npz'
    make_model(model)
    backend = Backend()
    monkeypatch.setattr(rec.vision.GestureRecognizer, 'create_from_options', lambda _: backend)
    config = MotionAutoConfig(rest_ms=200, pre_roll_ms=100, post_roll_ms=100,
                              min_motion_frames=3, max_window_ms=1000, duplicate_ms=1200)
    recognizer = rec.SignRecognizer(task, motion_model_path=model, motion_auto=True, motion_auto_config=config)
    calls = []
    recognizer._motion.predict = lambda trial: (calls.append(trial) or ('CUSTOM', 0.6))
    def feed(t, x=None):
        backend.present = x is not None
        if x is not None:
            backend.x = x
        return recognizer.process(np.zeros((2, 2, 3), dtype=np.uint8), t)
    yield recognizer, feed, calls, backend
    recognizer.close()


def gesture(feed, start, x=0.1):
    events = []
    for i, pos in enumerate([x, x, x+.04, x+.08, x+.12, x+.12, x+.12, x+.12, x+.12]):
        events += feed(start + i*100, pos)
    return events


def test_idle_and_one_gesture_with_duplicate_suppression(rig):
    recognizer, feed, calls, _ = rig
    for t in range(0, 500, 100):
        assert feed(t, .1) == []
    events = gesture(feed, 500)
    assert len(events) == 1 and (events[0].label, events[0].score) == ('CUSTOM', .6)
    assert len(calls) == 1
    assert np.all(np.diff(calls[0]['timestamps_ms']) > 0)
    assert gesture(feed, 1400, .12) == []
    assert len(calls) == 2
    assert recognizer.motion_auto
    with pytest.raises(ValueError):
        feed(2200, .24)
    with pytest.raises(RuntimeError):
        recognizer.start(2300)


def test_continuous_motion_times_out_and_needs_rest(rig):
    _, feed, calls, _ = rig
    for i in range(30):
        assert feed(i*100, .01*i) == []
    for i in range(30, 35):
        assert feed(i*100, .29) == []
    assert not calls


def test_short_and_lost_hand_are_rejected(rig):
    _, feed, calls, _ = rig
    for i, x in enumerate([.1, .1, .14, .14, .14, .14, .14, .18, .22, None, None, .22, .22, .22, .22]):
        assert feed(i*100, x) == []
    assert not calls


def test_rejected_model_cancel_and_close(rig):
    recognizer, feed, calls, backend = rig
    recognizer._motion.predict = lambda trial: (None, 0.0)
    assert gesture(feed, 0) == []
    for i, x in enumerate([.22, .26, .30]):
        feed(900+i*100, x)
    recognizer.cancel()
    for i in range(12, 16):
        assert feed(i*100, .30) == []
    assert not calls
    recognizer.close()
    assert backend.closed


def test_auto_environment_and_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv('QUIETBRIDGE_MOTION_AUTO', '1')
    monkeypatch.delenv('QUIETBRIDGE_MOTION_MODEL', raising=False)
    with pytest.raises(ValueError, match='requires a motion model'):
        rec.SignRecognizer(tmp_path / 'unused.task')
    task = tmp_path / 'model.task'
    task.touch()
    model = tmp_path / 'model.npz'
    make_model(model)
    monkeypatch.setattr(rec.vision.GestureRecognizer, 'create_from_options', lambda _: Backend())
    recognizer = rec.SignRecognizer(task, motion_model_path=model)
    assert recognizer.motion_auto
    recognizer.close()

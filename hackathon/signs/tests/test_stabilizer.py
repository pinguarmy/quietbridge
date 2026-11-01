import pytest

from hackathon.contracts import SignEvent
from hackathon.signs.stabilizer import EventStabilizer


def test_emits_once_after_observation_and_duration_thresholds():
    stabilizer = EventStabilizer()
    assert stabilizer.update("THUMB_UP", 0.91, 100) == []
    assert stabilizer.update("THUMB_UP", 0.82, 180) == []
    assert stabilizer.update("THUMB_UP", 0.88, 260) == []
    event = stabilizer.update("THUMB_UP", 0.95, 350)

    assert event == [SignEvent("THUMB_UP", 0.82, 100, 350, True)]
    assert stabilizer.update("THUMB_UP", 0.9, 400) == []


def test_requires_both_observation_count_and_duration():
    too_few = EventStabilizer(min_observations=4, min_duration_ms=250)
    for timestamp in (0, 100, 200):
        assert too_few.update("OPEN_PALM", 0.8, timestamp) == []

    too_short = EventStabilizer(min_observations=4, min_duration_ms=250)
    for timestamp in (0, 50, 100, 200):
        assert too_short.update("OPEN_PALM", 0.8, timestamp) == []


def test_low_score_unknown_and_unsupported_labels_are_neutral():
    stabilizer = EventStabilizer(min_observations=2, min_duration_ms=10)
    assert stabilizer.update("THUMB_UP", 0.9, 0) == []
    assert stabilizer.update("THUMB_UP", 0.74, 10) == []
    assert stabilizer.update("WAVE", 0.99, 20) == []
    assert stabilizer.update(None, 1.0, 30) == []
    assert stabilizer.update("THUMB_UP", 0.9, 40) == []
    assert stabilizer.update("THUMB_UP", 0.9, 50) == [
        SignEvent("THUMB_UP", 0.9, 40, 50, True)
    ]


def test_switching_labels_restarts_candidate():
    stabilizer = EventStabilizer(min_observations=3, min_duration_ms=100)
    for label, timestamp in (("THUMB_UP", 0), ("THUMB_UP", 100), ("THUMB_DOWN", 150), ("THUMB_DOWN", 200)):
        assert stabilizer.update(label, 0.8, timestamp) == []
    assert stabilizer.update("THUMB_DOWN", 0.8, 250) == [
        SignEvent("THUMB_DOWN", 0.8, 150, 250, True)
    ]


def test_gap_breaks_candidate_and_sparse_time_does_not_count_as_hold():
    stabilizer = EventStabilizer(
        min_observations=2, min_duration_ms=10, max_gap_ms=20
    )
    assert stabilizer.update("POINTING_UP", 0.9, 0) == []
    # Long time elapsed, but only two observations are present and the gap
    # means the first one cannot contribute to this candidate.
    assert stabilizer.update("POINTING_UP", 0.9, 100) == []
    assert stabilizer.update("POINTING_UP", 0.9, 110) == [
        SignEvent("POINTING_UP", 0.9, 100, 110, True)
    ]


def test_lock_requires_contiguous_neutral_for_rearm_and_blocks_label_switch():
    stabilizer = EventStabilizer(
        min_observations=2, min_duration_ms=10, max_gap_ms=50, rearm_ms=40
    )
    assert stabilizer.update("THUMB_UP", 0.8, 0) == []
    assert stabilizer.update("THUMB_UP", 0.8, 10) == [
        SignEvent("THUMB_UP", 0.8, 0, 10, True)
    ]
    assert stabilizer.update("THUMB_DOWN", 0.9, 20) == []
    assert stabilizer.update(None, 0, 30) == []
    assert stabilizer.update("OPEN_PALM", 0.9, 40) == []
    assert stabilizer.update(None, 0, 50) == []
    assert stabilizer.update(None, 0, 70) == []
    # Neutral evidence spans only 20ms after the observable non-neutral reset.
    assert stabilizer.update(None, 0, 90) == []
    assert stabilizer.update("OPEN_PALM", 0.9, 100) == []
    assert stabilizer.update(None, 0, 110) == []
    assert stabilizer.update(None, 0, 150) == []
    assert stabilizer.update("OPEN_PALM", 0.9, 160) == []
    assert stabilizer.update("OPEN_PALM", 0.9, 170) == [
        SignEvent("OPEN_PALM", 0.9, 160, 170, True)
    ]


def test_gap_breaks_neutral_rearm_evidence():
    stabilizer = EventStabilizer(
        min_observations=2, min_duration_ms=10, max_gap_ms=20, rearm_ms=30
    )
    stabilizer.update("THUMB_UP", 0.8, 0)
    assert stabilizer.update("THUMB_UP", 0.8, 10)
    assert stabilizer.update(None, 0, 20) == []
    # The 50ms gap discards neutral evidence from 20ms; starts fresh at 70.
    assert stabilizer.update(None, 0, 70) == []
    assert stabilizer.update(None, 0, 90) == []
    assert stabilizer.update("THUMB_DOWN", 0.8, 100) == []
    assert stabilizer.update(None, 0, 110) == []
    assert stabilizer.update(None, 0, 120) == []
    assert stabilizer.update(None, 0, 130) == []
    assert stabilizer.update(None, 0, 140) == []
    assert stabilizer.update("THUMB_DOWN", 0.8, 150) == []
    assert stabilizer.update("THUMB_DOWN", 0.8, 160) == [
        SignEvent("THUMB_DOWN", 0.8, 150, 160, True)
    ]


def test_timestamp_must_be_strictly_increasing():
    stabilizer = EventStabilizer()
    stabilizer.update(None, 0, 10)
    with pytest.raises(ValueError):
        stabilizer.update(None, 0, 10)
    with pytest.raises(ValueError):
        stabilizer.update(None, 0, 9)


def test_event_uses_first_and_last_observation_times_and_minimum_score():
    stabilizer = EventStabilizer(min_observations=3, min_duration_ms=20)
    assert stabilizer.update("THUMB_DOWN", 0.99, 1000) == []
    assert stabilizer.update("THUMB_DOWN", 0.77, 1010) == []
    event = stabilizer.update("THUMB_DOWN", 0.88, 1030)
    assert event == [SignEvent("THUMB_DOWN", 0.77, 1000, 1030, True)]

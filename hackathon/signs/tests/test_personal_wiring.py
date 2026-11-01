"""Personal classification mode must not silently fall back to canned classes."""
from types import SimpleNamespace

import numpy as np

from hackathon.signs.recognizer import SignRecognizer


def make_result():
    landmarks = [SimpleNamespace(x=i * .01, y=i * .02, z=i * .001) for i in range(21)]
    return SimpleNamespace(hand_landmarks=[landmarks], handedness=[[SimpleNamespace(category_name="Right")]])


def test_personal_result_uses_same_label_and_score_floor():
    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer._personal = SimpleNamespace(predict=lambda feature: ("OPEN_PALM", .9))
    assert recognizer._pick_personal_label(make_result()) == ("OPEN_PALM", .9)
    recognizer._personal = SimpleNamespace(predict=lambda feature: ("OPEN_PALM", .74))
    assert recognizer._pick_personal_label(make_result()) == (None, 0.0)


def test_personal_rejection_and_no_hand_are_not_canned_fallbacks():
    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer._personal = SimpleNamespace(predict=lambda feature: (None, 0.0))
    assert recognizer._pick_personal_label(make_result()) == (None, 0.0)
    assert recognizer._pick_personal_label(SimpleNamespace(hand_landmarks=[], handedness=[])) == (None, 0.0)


def test_personal_mode_rejects_two_hands_as_outside_collection_contract():
    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer._personal = SimpleNamespace(predict=lambda feature: ("OPEN_PALM", .9))
    result = make_result()
    result.hand_landmarks *= 2
    result.handedness *= 2
    assert recognizer._pick_personal_label(result) == (None, 0.0)


def test_landmark_normalization_does_not_mutate_caller_array():
    from hackathon.signs.features import hand_features

    points = np.arange(63, dtype=np.float64).reshape(21, 3)
    before = points.copy()
    hand_features(points, "Left")
    np.testing.assert_array_equal(points, before)

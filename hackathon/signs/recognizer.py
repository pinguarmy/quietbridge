"""Track-B sign recognizer built on MediaPipe's pretrained GestureRecognizer.

Synchronous VIDEO-mode recognizer; Sam's runtime calls :meth:`SignRecognizer.process`
from an external worker thread, one camera frame at a time. The recognizer never
opens a camera and never touches the network; the model file must already exist
on disk (assets are managed by the lead, see ``hackathon.signs.assets``).

Label contract
--------------
``LABEL_MAP`` maps the raw MediaPipe canned-classifier category names to the
sign labels shared with fusion via ``hackathon.contracts.SignEvent``.
``SUPPORTED_LABELS`` is the resulting public vocabulary. Only these four labels
can ever be emitted; any other top-1 category (including ``None``) is treated
as "no supported gesture".

Handedness rules
----------------
The model is built once with ``num_hands=2`` and no category allowlist, so the
reported top-1 is the raw classifier top-1. A hand only counts when its top-1
label is supported and its score is >= ``MIN_SCORE``. Two detected hands with
any unsupported/low-confidence hand or conflicting labels produce no output;
a single valid hand, or two hands with the same label, use the conservative
``min`` of the hand scores.

Debounce is delegated to ``hackathon.signs.stabilizer.EventStabilizer`` (Luna):
``update(label, score, timestamp_ms) -> list[SignEvent]``. The stabilizer owns
all temporal smoothing; this class only feeds it one candidate per frame and
returns whatever events the stabilizer confirms.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from hackathon.contracts import SignEvent

#: Raw MediaPipe canned-gesture category -> shared sign label.
LABEL_MAP = {
    "Thumb_Up": "THUMB_UP",
    "Thumb_Down": "THUMB_DOWN",
    "Open_Palm": "OPEN_PALM",
    "Pointing_Up": "POINTING_UP",
}

#: Public vocabulary this recognizer can emit, in stable order.
SUPPORTED_LABELS = tuple(LABEL_MAP.values())

#: Match Sam's fusion floor so a rejected event cannot lock out a later valid hold.
MIN_SCORE = 0.75

#: The backend is always built to track at most this many hands.
NUM_HANDS = 2

_DOWNLOAD_HINT = "python -m hackathon.signs.assets --download"


class SignRecognizer:
    """Synchronous sign recognizer over BGR frames with external timestamps.

    Parameters
    ----------
    model_path:
        Path to the ``.task`` gesture model. Defaults to
        ``hackathon.signs.assets.default_model_path()``. If the file is
        missing, :class:`FileNotFoundError` is raised with a download hint;
        nothing is downloaded implicitly.
    stabilizer:
        Optional ``EventStabilizer`` instance (or compatible object whose
        ``update`` returns an iterable of ``SignEvent``). Defaults to
        ``EventStabilizer(min_score=MIN_SCORE)`` from
        ``hackathon.signs.stabilizer``.
    """

    def __init__(self, model_path=None, *, stabilizer=None, personal_model_path=None):
        # Personal classification is opt-in: an old local model must not silently
        # change a teammate's stock demo. No fallback to stock after rejection.
        personal_model_path = personal_model_path or os.environ.get("QUIETBRIDGE_PERSONAL_GESTURES")
        self._personal = None
        if personal_model_path:
            from .personal import PersonalGestureModel

            self._personal = PersonalGestureModel.load(Path(personal_model_path).expanduser())
        if model_path is None:
            from .assets import default_model_path

            model_path = default_model_path()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Gesture recognizer model not found at {model_path}. "
                f"Download it first with: {_DOWNLOAD_HINT}"
            )

        if stabilizer is None:
            from .stabilizer import EventStabilizer

            stabilizer = EventStabilizer(min_score=MIN_SCORE)
        self._stabilizer = stabilizer

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            # No category allowlist: keep the raw classifier top-1 so that
            # unsupported gestures stay visible instead of being forced into
            # one of our four labels.
        )
        self._backend = vision.GestureRecognizer.create_from_options(options)
        self._last_timestamp_ms: Optional[int] = None

    def process(self, frame_bgr, timestamp_ms: int) -> list[SignEvent]:
        """Classify one frame and return newly confirmed events (usually ``[]``).

        ``frame_bgr`` is an HxWx3 uint8 BGR image (as captured by OpenCV).
        ``timestamp_ms`` is the caller's monotonic millisecond clock; it is
        forwarded to MediaPipe unchanged and must be strictly increasing
        across calls, otherwise :class:`ValueError` is raised before the
        backend is touched.
        """
        if self._backend is None:
            raise RuntimeError("SignRecognizer is closed")

        # Validate before calling the backend: feeding MediaPipe a
        # non-increasing timestamp would put it into an error state.
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError(
                "timestamp_ms must be strictly increasing across calls; "
                f"got {timestamp_ms} after {self._last_timestamp_ms}"
            )
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._backend.recognize_for_video(image, timestamp_ms)

        label, score = self._pick_personal_label(result) if self._personal is not None else self._pick_label(result)
        return list(self._stabilizer.update(label, score, timestamp_ms))

    def _pick_personal_label(self, result) -> tuple[Optional[str], float]:
        from .features import hand_features

        # Collection records exactly one hand per feature trial. Reject two-hand
        # input rather than pretending that combination was part of training.
        if len(result.hand_landmarks) != 1:
            return None, 0.0
        feature = hand_features(result.hand_landmarks[0], result.handedness[0][0].category_name)
        label, score = self._personal.predict(feature)
        if label not in SUPPORTED_LABELS or score < MIN_SCORE:
            return None, 0.0
        return label, score

    @staticmethod
    def _pick_label(result) -> tuple[Optional[str], float]:
        """Reduce raw per-hand top-1 results to one conservative candidate."""
        hands = []
        for hand_gestures in result.gestures:
            if not hand_gestures:
                hands.append(None)
                continue
            top = hand_gestures[0]
            if top.category_name in LABEL_MAP and top.score >= MIN_SCORE:
                hands.append((LABEL_MAP[top.category_name], float(top.score)))
            else:
                # A detected hand with an unsupported or low-confidence top-1.
                hands.append(None)

        if not hands:
            return None, 0.0
        if len(hands) == 1:
            hand = hands[0]
            return hand if hand is not None else (None, 0.0)
        # Two hands: both must be valid and agree, otherwise stay quiet.
        if hands[0] is None or hands[1] is None:
            return None, 0.0
        if hands[0][0] != hands[1][0]:
            return None, 0.0
        return hands[0][0], min(hands[0][1], hands[1][1])

    def close(self) -> None:
        """Release the MediaPipe backend. Safe to call more than once."""
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def __enter__(self) -> "SignRecognizer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

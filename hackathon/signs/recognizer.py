"""Track-B sign recognizer built on MediaPipe's pretrained GestureRecognizer.

Synchronous VIDEO-mode recognizer; Sam's runtime calls :meth:`SignRecognizer.process`
from an external worker thread, one camera frame at a time. The recognizer never
opens a camera and never touches the network; the model file must already exist
on disk (assets are managed by the lead, see ``hackathon.signs.assets``).

Label contract
--------------
``LABEL_MAP`` maps the raw MediaPipe canned-classifier category names to the
sign labels shared with fusion via ``hackathon.contracts.SignEvent``.
``SUPPORTED_LABELS`` is the stock vocabulary. In the default stock mode, only
these four labels can be emitted; unsupported top-1 categories stay silent.
The opt-in motion mode emits labels learned by its selected model instead.

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
import time
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
    motion_model_path:
        Optional motion NPZ path; alternatively set ``QUIETBRIDGE_MOTION_MODEL``.
        Explicit ``start``/``finish`` windows replace framewise classification;
        personal static models and the stabilizer are not used in motion mode.
    stabilizer:
        Optional ``EventStabilizer`` instance (or compatible object whose
        ``update`` returns an iterable of ``SignEvent``). Defaults to
        ``EventStabilizer(min_score=MIN_SCORE)`` from
        ``hackathon.signs.stabilizer``.
    """

    def __init__(self, model_path=None, *, stabilizer=None, personal_model_path=None, motion_model_path=None):
        motion_model_path = motion_model_path or os.environ.get("QUIETBRIDGE_MOTION_MODEL")
        self._motion = None
        self._motion_frames = None
        self._motion_start_ms = None
        self._motion_last_boundary_ms = None
        if motion_model_path:
            from .motion import MotionModel

            self._motion = MotionModel.load(Path(motion_model_path).expanduser())
        # Personal classification is opt-in: an old local model must not silently
        # change a teammate's stock demo. No fallback to stock after rejection.
        personal_model_path = personal_model_path or os.environ.get("QUIETBRIDGE_PERSONAL_GESTURES")
        self._personal = None
        if personal_model_path and self._motion is None:
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
        self.last_tracking = []

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
        if self._motion_frames is not None and timestamp_ms < self._motion_start_ms:
            raise ValueError("frame timestamp precedes motion start")
        if self._motion_last_boundary_ms is not None and timestamp_ms < self._motion_last_boundary_ms:
            raise ValueError("frame timestamp precedes the last motion boundary")
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._backend.recognize_for_video(image, timestamp_ms)
        self.last_tracking = []
        hand_landmarks = getattr(result, "hand_landmarks", None) or []
        gestures = getattr(result, "gestures", None) or []
        handedness = getattr(result, "handedness", None) or []
        for index, landmarks in enumerate(hand_landmarks):
            hand_gestures = gestures[index] if index < len(gestures) else []
            top = hand_gestures[0] if hand_gestures else None
            categories = handedness[index] if index < len(handedness) else []
            self.last_tracking.append({
                "handedness": categories[0].category_name if categories else "",
                "landmarks": [[float(p.x), float(p.y), float(p.z)] for p in landmarks],
                "label": LABEL_MAP.get(top.category_name) if top else None,
                "score": float(top.score) if top else 0.0,
            })

        if self._motion is not None:
            if self._motion_frames is not None:
                self._motion_frames.append((timestamp_ms, self.last_tracking))
            return []

        label, score = self._pick_personal_label(result) if self._personal is not None else self._pick_label(result)
        return list(self._stabilizer.update(label, score, timestamp_ms))

    def start(self, timestamp_ms=None) -> None:
        """Start or replace an explicitly segmented motion window.

        Times use the same monotonic millisecond clock as ``process``.
        """
        if self._backend is None:
            raise RuntimeError("SignRecognizer is closed")
        if self._motion is None:
            raise RuntimeError("motion model is not configured")
        timestamp_ms = int(time.perf_counter() * 1000) if timestamp_ms is None else timestamp_ms
        if self._last_timestamp_ms is not None and timestamp_ms < self._last_timestamp_ms:
            raise ValueError("start timestamp precedes the last frame")
        if self._motion_last_boundary_ms is not None and timestamp_ms < self._motion_last_boundary_ms:
            raise ValueError("start timestamp precedes the last motion boundary")
        self._motion_last_boundary_ms = timestamp_ms
        self._motion_start_ms = timestamp_ms
        self._motion_frames = []

    def cancel(self) -> None:
        """Discard an active motion window without classifying it.

        The last observed frame or start remains the monotonic boundary.
        With no active window, including in stock mode, this is a no-op.
        """
        if self._motion_frames is None:
            return
        if self._motion_frames:
            self._motion_last_boundary_ms = self._motion_frames[-1][0]
        else:
            self._motion_last_boundary_ms = self._motion_start_ms
        self._motion_frames = None
        self._motion_start_ms = None

    def finish(self, timestamp_ms: int) -> list[SignEvent]:
        """Classify the complete window once; rejected and REST trials stay silent."""
        if self._backend is None:
            raise RuntimeError("SignRecognizer is closed")
        if self._motion_frames is None:
            return []
        if timestamp_ms < self._motion_start_ms or (
            self._motion_frames and timestamp_ms < self._motion_frames[-1][0]
        ):
            raise ValueError("end timestamp precedes the motion window")
        self._motion_last_boundary_ms = timestamp_ms
        frames = self._motion_frames
        start_ms = self._motion_start_ms
        self._motion_frames = None
        self._motion_start_ms = None
        if len(frames) < 2:
            return []
        import numpy as np
        from .motion import MotionModelError

        landmarks = np.full((len(frames), 2, 21, 3), np.nan, dtype=np.float64)
        present = np.zeros((len(frames), 2), dtype=bool)
        handedness = np.full((len(frames), 2), "", dtype="<U16")
        for i, (_, hands) in enumerate(frames):
            for j, hand in enumerate(hands[:2]):
                points = np.asarray(hand["landmarks"], dtype=np.float64)
                if points.shape != (21, 3) or not np.all(np.isfinite(points)):
                    continue
                landmarks[i, j] = points
                present[i, j] = True
                handedness[i, j] = hand["handedness"]
        if not present.any():
            return []
        trial = {"landmarks": landmarks, "hand_present": present,
                 "handedness": handedness,
                 "timestamps_ms": np.asarray([t for t, _ in frames])}
        try:
            label, score = self._motion.predict(trial)
        except MotionModelError:
            return []
        if label is None:
            return []
        return [SignEvent(label=label, score=score, start_ms=start_ms,
                          end_ms=timestamp_ms, is_final=True)]

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
        self._motion_frames = None
        if backend is not None:
            backend.close()

    def __enter__(self) -> "SignRecognizer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

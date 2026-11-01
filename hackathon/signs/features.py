"""Hand landmark features for the personal static-gesture template model.

One shared contract, used by collection, training and runtime:

    hand_features(landmarks, handedness) -> np.ndarray  # (63,) float32

`landmarks` is either a sequence of 21 MediaPipe-style landmark objects with
``.x``/``.y``/``.z`` attributes, or a NumPy array of shape ``(21, 3)``.
``handedness`` is ``"Left"`` or ``"Right"`` (case-insensitive).

Normalisation:
- translate so the wrist (landmark 0) is the origin;
- scale by the 2D (x, y) wrist -> middle-finger-MCP (landmark 9) distance;
- mirror ``Left`` hands on the x axis so both hands share one template space.

Mirroring flips x only. There is deliberately **no rotation**: rotating the
hand upright would erase the up/down direction that distinguishes THUMB_UP
from THUMB_DOWN. Degenerate input (wrong shape, non-finite values, or a
zero-length wrist->MCP reference) raises ``ValueError``.
"""
from __future__ import annotations

import numpy as np

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

# Below this wrist->MCP distance the hand geometry carries no usable shape.
_MIN_REFERENCE_SCALE = 1e-6

_VALID_HANDEDNESS = ("left", "right")


def _to_array(landmarks) -> np.ndarray:
    """Normalise supported landmark inputs to a finite (21, 3) float64 array."""
    if isinstance(landmarks, np.ndarray):
        pts = landmarks.astype(np.float64, copy=True)  # never mutate the caller
    else:
        items = list(landmarks)
        rows = []
        for lm in items:
            if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                rows.append((float(lm.x), float(lm.y), float(lm.z)))
            else:
                x, y, z = lm  # sequence of 3 numbers
                rows.append((float(x), float(y), float(z)))
        pts = np.asarray(rows, dtype=np.float64)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"expected {NUM_LANDMARKS} landmarks with x/y/z, got shape {pts.shape}"
        )
    if not np.all(np.isfinite(pts)):
        raise ValueError("landmarks contain non-finite coordinates")
    return pts


def hand_features(landmarks, handedness: str) -> np.ndarray:
    """Return the normalised (63,) float32 feature vector for one hand frame."""
    if not isinstance(handedness, str) or handedness.lower() not in _VALID_HANDEDNESS:
        raise ValueError(f"handedness must be 'Left' or 'Right', got {handedness!r}")

    pts = _to_array(landmarks)

    # Unify left/right hands by mirroring the x axis of left hands. x only:
    # y keeps its sign so up vs down stays meaningful.
    if handedness.lower() == "left":
        pts[:, 0] = -pts[:, 0]

    # Wrist translation.
    pts = pts - pts[WRIST_INDEX]

    # Uniform scale from the 2D wrist -> middle-MCP distance. A collapsed
    # reference length means the landmarks carry no shape information.
    scale = float(np.linalg.norm(pts[MIDDLE_MCP_INDEX, :2]))
    if scale < _MIN_REFERENCE_SCALE:
        raise ValueError(
            "degenerate hand: wrist->middle-MCP 2D distance is ~0; "
            "cannot normalise scale"
        )
    pts = pts / scale

    return pts.astype(np.float32).ravel()

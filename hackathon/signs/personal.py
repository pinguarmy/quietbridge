"""Personal static-gesture template model (small-sample, no deep training).

One user records a handful of independent trials per gesture. Each trial is a
short (~2 s) held pose; every trial collapses to a single median template.
Classification is nearest-3-templates majority vote with distance rejection,
plus NEUTRAL as an explicit negative class.

Honesty constraints baked into the contract:
- ``predict`` returns ``(label_or_None, score)``. The score is a heuristic
  similarity in [0, 1], **not a probability**. Rejection returns
  ``(None, 0.0)``. NEUTRAL is never returned as a label; it maps to ``None``.
- The rejection threshold is estimated on the training trials only, by
  leave-one-trial-out. Test data is never used to tune it.
- Fitting requires at least 5 independent trials per positive label and at
  least 5 NEUTRAL trials that contain a hand; anything less fails loudly.
  Duplicate ``trial_id`` values are refused, because repeats of one recording
  would fake independence.
- ``evaluate_heldout`` applies the same ``score >= 0.75`` gate the runtime
  recognizer uses and reports label-only vs runtime-gated results
  separately; a label under the floor counts as rejected, not correct.

Trial dicts use the shared collection format::

    {"features": (N, 63) float32 ndarray,   # N may be 0 for a no-hand trial
     "label": "THUMB_UP" | ... | "NEUTRAL",
     "session_id": str, "trial_id": str}

``save``/``load`` use a plain ``.npz`` with numeric and fixed-width string
arrays only — no pickling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hackathon.signs.features import FEATURE_DIM

logger = logging.getLogger(__name__)

POSITIVE_LABELS = ("THUMB_UP", "THUMB_DOWN", "OPEN_PALM", "POINTING_UP")
NEUTRAL_LABEL = "NEUTRAL"
ALL_LABELS = POSITIVE_LABELS + (NEUTRAL_LABEL,)

MIN_TRIALS_PER_LABEL = 5
K_NEIGHBOURS = 3

#: Same acceptance floor as ``hackathon.signs.recognizer.MIN_SCORE``: a label
#: whose score falls below this is rejected at runtime, so held-out evaluation
#: must apply the identical gate instead of counting the raw label a success.
#: Kept as a local constant because importing the recognizer would pull in
#: MediaPipe; the wiring tests pin the two values to the same behaviour.
RUNTIME_SCORE_FLOOR = 0.75


def _validate_trial(trial: dict) -> tuple[np.ndarray, str, str, str]:
    try:
        features = np.asarray(trial["features"], dtype=np.float32)
        label = str(trial["label"])
        session_id = str(trial["session_id"])
        trial_id = str(trial["trial_id"])
    except KeyError as exc:
        raise ValueError(f"trial is missing field {exc}") from exc
    if label not in ALL_LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {ALL_LABELS}")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"trial {trial_id!r} features must have shape (N, {FEATURE_DIM}), "
            f"got {features.shape}"
        )
    return features, label, session_id, trial_id


def _trial_template(features: np.ndarray) -> np.ndarray:
    """Aggregate one trial to a single template (median over frames)."""
    return np.median(features, axis=0).astype(np.float32)


def assert_disjoint_sessions(train_trials: list[dict], test_trials: list[dict]) -> None:
    """Refuse evaluation when a session leaks across the train/test split."""
    train_sessions = {str(t["session_id"]) for t in train_trials}
    test_sessions = {str(t["session_id"]) for t in test_trials}
    overlap = sorted(train_sessions & test_sessions)
    if overlap:
        raise ValueError(
            "test sessions overlap training sessions, which would leak the "
            f"person's recording conditions into the held-out estimate: {overlap}"
        )


class PersonalGestureModel:
    """Nearest-template personal gesture model with distance rejection."""

    def __init__(self) -> None:
        self.templates_: np.ndarray | None = None  # (M, 63) float32
        self.template_labels_: np.ndarray | None = None  # (M,) '<U12' strings
        self.threshold_: float | None = None
        # Trial ids of NEUTRAL trials with no detected hand (N=0). They are
        # valid "emit nothing" examples for evaluation, never feature rows.
        self.no_hand_trial_ids_: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, trials: list[dict]) -> "PersonalGestureModel":
        validated = [_validate_trial(t) for t in trials]

        trial_ids = [trial_id for _f, _l, _s, trial_id in validated]
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        if duplicates:
            raise ValueError(
                "duplicate trial_id values cannot count as independent "
                f"trials: {duplicates}. Record separate trials with distinct "
                "ids instead of reusing one."
            )

        templates: list[np.ndarray] = []
        labels: list[str] = []
        validated_with_hand: list[tuple[np.ndarray, str, str, str]] = []
        self.no_hand_trial_ids_ = []
        per_label = {label: 0 for label in ALL_LABELS}
        neutral_with_hand = 0

        for features, label, _session_id, trial_id in validated:
            if features.shape[0] == 0:
                if label == NEUTRAL_LABEL:
                    logger.info(
                        "trial %s is NEUTRAL with no detected hand; kept as a "
                        "no-hand example, excluded from templates",
                        trial_id,
                    )
                    self.no_hand_trial_ids_.append(trial_id)
                    continue
                raise ValueError(
                    f"trial {trial_id!r} labelled {label} has no hand frames; "
                    "a positive-gesture trial needs at least one frame"
                )
            per_label[label] += 1
            if label == NEUTRAL_LABEL:
                neutral_with_hand += 1
            templates.append(_trial_template(features))
            labels.append(label)
            validated_with_hand.append((features, label, _session_id, trial_id))

        shortages = [
            f"{label} has {per_label[label]} (<{MIN_TRIALS_PER_LABEL})"
            for label in POSITIVE_LABELS
            if per_label[label] < MIN_TRIALS_PER_LABEL
        ]
        if neutral_with_hand < MIN_TRIALS_PER_LABEL:
            shortages.append(
                f"NEUTRAL has {neutral_with_hand} with-hand trials "
                f"(<{MIN_TRIALS_PER_LABEL})"
            )
        if shortages:
            raise ValueError(
                "not enough independent trials to fit a personal model: "
                + "; ".join(shortages)
                + ". Record more separate trials; extra frames of the same "
                "trial do not count."
            )

        self.templates_ = np.stack(templates).astype(np.float32)
        self.template_labels_ = np.asarray(labels, dtype="<U12")
        self.threshold_ = self._estimate_threshold_loo(
            [t[0] for t in validated_with_hand], labels
        )
        return self

    def _estimate_threshold_loo(
        self, trial_features: list[np.ndarray], labels: list[str]
    ) -> float:
        """Leave-one-trial-out distance threshold, train data only.

        ``predict`` runs on single frames at runtime, so the threshold is
        estimated per frame: for every frame of every trial, the distance to
        the nearest template of the *other* trials, keeping only frames whose
        left-out 3-NN majority vote is correct. The threshold is the largest
        such distance — the widest radius at which training frames were still
        classified correctly. If no frame is classified correctly left-out,
        the data cannot support rejection: fail loudly.
        """
        n = self.templates_.shape[0]
        correct_distances: list[float] = []
        misclassified_frames = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_t = self.templates_[mask]
            others_l = self.template_labels_[mask]
            dists = np.linalg.norm(
                trial_features[i][:, None, :] - others_t[None, :, :], axis=-1
            )
            for frame_dists in dists:
                order = np.argsort(frame_dists)
                prediction = _majority(others_l[order[:K_NEIGHBOURS]])
                if prediction == labels[i]:
                    correct_distances.append(float(frame_dists[order[0]]))
                else:
                    misclassified_frames += 1

        if not correct_distances:
            raise ValueError(
                "leave-one-trial-out on the training trials classifies every "
                "frame incorrectly; the recordings do not form usable "
                "templates. Re-record with more distinct, consistent poses."
            )
        if misclassified_frames:
            logger.warning(
                "%d training frames are misclassified under "
                "leave-one-trial-out; the threshold still accepts correctly "
                "classified frames at their widest distance. "
                "Expect held-out errors.",
                misclassified_frames,
            )
        return max(correct_distances)

    # -------------------------------------------------------------- predict
    def predict(self, feature: np.ndarray) -> tuple[str | None, float]:
        """Classify one (63,) frame feature.

        Returns ``(label, score)`` for a supported positive gesture, or
        ``(None, 0.0)`` when the frame is rejected by distance, when the
        neighbourhood votes NEUTRAL, or when the vote is ambiguous. The score
        is a heuristic similarity, not a calibrated probability.
        """
        if self.templates_ is None or self.threshold_ is None:
            raise RuntimeError("model is not fitted")
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,):
            raise ValueError(
                f"feature must have shape ({FEATURE_DIM},), got {feature.shape}"
            )

        dists = np.linalg.norm(self.templates_ - feature, axis=-1)
        order = np.argsort(dists)
        nearest = float(dists[order[0]])
        if nearest > self.threshold_:
            return None, 0.0

        neighbours = self.template_labels_[order[:K_NEIGHBOURS]]
        prediction = _majority(neighbours)
        if prediction is None or prediction == NEUTRAL_LABEL:
            return None, 0.0

        # Heuristic similarity: 1.0 at an exact template match, 0.0 at the
        # rejection boundary. Monotonic in distance, nothing more.
        score = max(0.0, 1.0 - nearest / self.threshold_) if self.threshold_ > 0 else 0.0
        return str(prediction), float(score)

    # ----------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        if self.templates_ is None or self.threshold_ is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez stores plain ndarrays (no pickling); np.load enforces
        # allow_pickle=False on the read side.
        np.savez(
            path,
            templates=self.templates_.astype(np.float32),
            template_labels=self.template_labels_.astype("<U12"),
            threshold=np.asarray(self.threshold_, dtype=np.float64),
            no_hand_trial_ids=np.asarray(self.no_hand_trial_ids_, dtype="<U64"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PersonalGestureModel":
        with np.load(path, allow_pickle=False) as data:
            model = cls()
            model.templates_ = np.asarray(data["templates"], dtype=np.float32)
            model.template_labels_ = np.asarray(data["template_labels"], dtype="<U12")
            model.threshold_ = float(data["threshold"])
            model.no_hand_trial_ids_ = [str(t) for t in data["no_hand_trial_ids"]]
        return model


def _majority(labels: np.ndarray) -> str | None:
    """Strict majority of the 3 neighbours; None on NEUTRAL-free 3-way tie."""
    labels = [str(label) for label in labels]
    counts = {label: labels.count(label) for label in set(labels)}
    best_label = max(counts, key=counts.get)
    if counts[best_label] >= 2:
        return best_label
    return None  # 1-1-1 split: no defensible answer


# ---------------------------------------------------------------- evaluation
def _runtime_accept(label: str | None, score: float) -> str | None:
    """The exact gate SignRecognizer applies before a label may reach the UI."""
    if label is None or score < RUNTIME_SCORE_FLOOR:
        return None
    return label


def evaluate_heldout(
    model: PersonalGestureModel,
    train_trials: list[dict],
    test_trials: list[dict],
) -> dict:
    """Held-out evaluation with the split and the runtime gate taken seriously.

    Refuses to run when sessions overlap between train and test. Every
    prediction is judged twice: the raw label match (``label_correct``) and
    the label after the runtime ``score >= RUNTIME_SCORE_FLOOR`` gate
    (``gated_correct``) — the second is what the live recognizer would
    actually accept, and correct labels lost to the floor are counted in
    ``gated_out``. Trial-level and frame-level counts are reported
    separately; neither is a claim of real-world accuracy with this few
    samples, and passing the gate on static features is not evidence that
    debounced live events fire.
    """
    assert_disjoint_sessions(train_trials, test_trials)

    trial_total = trial_label_correct = trial_gated_correct = 0
    gated_out = no_hand_trials = 0
    frame_total = frame_label_correct = frame_gated_correct = 0
    per_label: dict[str, dict[str, int]] = {
        label: {"trials": 0, "label_correct": 0, "gated_correct": 0}
        for label in ALL_LABELS
    }

    for trial in test_trials:
        features, label, _session_id, _trial_id = _validate_trial(trial)
        expected = None if label == NEUTRAL_LABEL else label

        if features.shape[0] == 0:
            # No hand was ever detected: the pipeline emits nothing. That is
            # correct only for NEUTRAL; a positive gesture with zero usable
            # frames is a miss, not a free success.
            trial_total += 1
            no_hand_trials += 1
            per_label[label]["trials"] += 1
            if label == NEUTRAL_LABEL:
                trial_label_correct += 1
                trial_gated_correct += 1
                per_label[label]["label_correct"] += 1
                per_label[label]["gated_correct"] += 1
            continue

        raw_label, score = model.predict(_trial_template(features))
        accepted = _runtime_accept(raw_label, score)
        trial_total += 1
        per_label[label]["trials"] += 1
        if raw_label == expected:
            trial_label_correct += 1
            per_label[label]["label_correct"] += 1
        if accepted == expected:
            trial_gated_correct += 1
            per_label[label]["gated_correct"] += 1
        elif raw_label == expected:
            gated_out += 1

        for frame in features:
            frame_label, frame_score = model.predict(frame)
            frame_total += 1
            if frame_label == expected:
                frame_label_correct += 1
            if _runtime_accept(frame_label, frame_score) == expected:
                frame_gated_correct += 1

    return {
        "runtime_score_floor": RUNTIME_SCORE_FLOOR,
        "trial_level": {
            "total": trial_total,
            "label_correct": trial_label_correct,
            "gated_correct": trial_gated_correct,
            "gated_out": gated_out,
            "no_hand_trials": no_hand_trials,
            "per_label": per_label,
        },
        "frame_level": {
            "total": frame_total,
            "label_correct": frame_label_correct,
            "gated_correct": frame_gated_correct,
        },
    }

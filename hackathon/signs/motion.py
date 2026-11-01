"""Personal dynamic-gesture model for QuietBridge motion-v2 sequences.

Template-based temporal classifier over dual-hand landmark sequences:

- Canonical left/right hand identity from MediaPipe handedness labels.
- Resample each trial to a fixed number of frames over its own time span,
  interpolating only across frames where the hand was present; dropout
  regions stay missing and are carried as an explicit mask.
- Scale-normalize by the per-sequence median hand size (wrist to middle
  MCP). Positions are NOT translated per frame and the inter-hand relation
  is preserved, so absolute movement in camera-normalized space is kept.
- Nearest-template matching with a masked distance, a train-calibrated
  reject threshold and a runner-up margin threshold (leave-one-trial-out
  on the training split only).
- REST is a first-class template label but never produces output:
  predict() returns (None, 0.0) for REST and for rejected input.

The score in [0, 1] is a heuristic matching score, not a probability:
score = clip(1 - d1 / (2 * reject_threshold), 0, 1), so 0.75 means the
match distance is at most half of the calibrated reject distance.

This model classifies complete, pre-segmented trials. It does not do
continuous/online recognition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SUPPORTED_LABELS = ("HELLO", "WELCOME", "OPENAI", "DDE", "HACKATHON", "REST")
REST_LABEL = "REST"
N_FRAMES = 32
N_HANDS = 2
N_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
_EPS = 1e-8


class MotionModelError(ValueError):
    """Raised for invalid trials, labels or data leakage."""


# ---------------------------------------------------------------------------
# Trial loading / validation
# ---------------------------------------------------------------------------

def load_trials(data_dir, split, session_prefix="peter-motion-v2-"):
    """Load trials of one split whose session_id starts with session_prefix."""
    split_dir = Path(data_dir).expanduser() / split
    trials = []
    for path in sorted(split_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            session_id = str(d["session_id"])
            if not session_id.startswith(session_prefix):
                continue  # never touch older pose/other collections
            trial = {
                "label": str(d["label"]),
                "landmarks": np.asarray(d["landmarks"], dtype=np.float64),
                "hand_present": np.asarray(d["hand_present"], dtype=bool),
                "timestamps_ms": np.asarray(d["timestamps_ms"], dtype=np.float64),
                "handedness": np.asarray(d["handedness"]).astype(str),
                "session_id": session_id,
                "trial_id": str(d["trial_id"]),
                "path": str(path),
            }
        trials.append(trial)
    return trials


def check_session_leakage(train_trials, test_trials):
    """Raise if any session_id appears in both splits (or trial ids collide)."""
    train_sessions = {t["session_id"] for t in train_trials}
    test_sessions = {t["session_id"] for t in test_trials}
    overlap = train_sessions & test_sessions
    if overlap:
        raise MotionModelError(
            f"session leakage between train and test: {sorted(overlap)}")
    train_ids = {t["trial_id"] for t in train_trials}
    test_ids = {t["trial_id"] for t in test_trials}
    dup = train_ids & test_ids
    if dup:
        raise MotionModelError(
            f"duplicated trial_id across splits: {sorted(dup)}")


def _validate_trial(trial, require_label=False):
    if require_label:
        label = str(trial.get("label", ""))
        if label not in SUPPORTED_LABELS:
            raise MotionModelError(
                f"unsupported label {label!r}; expected one of {SUPPORTED_LABELS}")
    lm = np.asarray(trial["landmarks"], dtype=np.float64)
    if lm.ndim != 4 or lm.shape[1:] != (N_HANDS, N_LANDMARKS, 3):
        raise MotionModelError(
            f"landmarks must have shape (N, {N_HANDS}, {N_LANDMARKS}, 3), got {lm.shape}")
    n = lm.shape[0]
    if n < 2:
        raise MotionModelError("trial needs at least 2 frames")
    hp = np.asarray(trial["hand_present"], dtype=bool)
    if hp.shape != (n, N_HANDS):
        raise MotionModelError(f"hand_present must have shape ({n}, {N_HANDS})")
    ts = np.asarray(trial["timestamps_ms"], dtype=np.float64)
    if ts.shape != (n,):
        raise MotionModelError(f"timestamps_ms must have shape ({n},)")
    return lm, hp, ts


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _canonical_hands(lm, hp, handedness):
    """Reorder slots to canonical [Left, Right] using handedness strings."""
    n = lm.shape[0]
    out = np.full((n, N_HANDS, N_LANDMARKS, 3), np.nan)
    present = np.zeros((n, N_HANDS), dtype=bool)
    for f in range(n):
        for slot in range(N_HANDS):
            if not hp[f, slot]:
                continue
            hand = str(handedness[f, slot]).strip().lower() if handedness is not None else ""
            if hand.startswith("left"):
                canon = 0
            elif hand.startswith("right"):
                canon = 1
            else:
                canon = slot  # unknown handedness: keep the original slot
            if present[f, canon]:
                canon = 1 - canon  # collision: put it in the other slot
                if present[f, canon]:
                    continue
            out[f, canon] = lm[f, slot]
            present[f, canon] = True
    return out, present


def preprocess_trial(trial):
    """Return (features, mask): (N_FRAMES, 2, 21, 3) and (N_FRAMES, 2) bool.

    Resampled on the trial's own time span; missing hands/frames stay
    masked out; divided by the per-sequence median hand scale; no
    per-frame translation.
    """
    lm, hp, ts = _validate_trial(trial)
    handedness = trial.get("handedness")
    lm, hp = _canonical_hands(lm, hp, handedness)

    if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        ts = np.arange(lm.shape[0], dtype=np.float64)  # fallback to frame index
    target_t = np.linspace(ts[0], ts[-1], N_FRAMES)
    median_dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0

    feat = np.zeros((N_FRAMES, N_HANDS, N_LANDMARKS, 3), dtype=np.float64)
    mask = np.zeros((N_FRAMES, N_HANDS), dtype=bool)
    for h in range(N_HANDS):
        idx = np.flatnonzero(hp[:, h])
        if idx.size == 0:
            continue
        t_src = ts[idx]
        tol = max(median_dt / 2.0, 1e-3)
        in_window = (target_t >= t_src[0] - tol) & (target_t <= t_src[-1] + tol)
        flat = lm[idx, h].reshape(idx.size, -1)  # interpolate present frames only
        interp = np.stack(
            [np.interp(target_t, t_src, flat[:, j]) for j in range(flat.shape[1])],
            axis=1,
        ).reshape(N_FRAMES, N_LANDMARKS, 3)
        feat[:, h][in_window] = interp[in_window]
        mask[:, h] = in_window

    # Per-sequence scale: median wrist->middle-MCP length across present
    # frames of both hands. Single scalar keeps inter-hand relations intact.
    scale_samples = []
    for h in range(N_HANDS):
        m = mask[:, h]
        if m.any():
            seg = feat[m, h]
            scale_samples.append(np.linalg.norm(seg[:, WRIST] - seg[:, MIDDLE_MCP], axis=1))
    scale = float(np.median(np.concatenate(scale_samples))) if scale_samples else 1.0
    if not np.isfinite(scale) or scale < _EPS:
        scale = 1.0
    feat = feat / scale
    return feat.astype(np.float32), mask


def masked_distance(feat_a, mask_a, feat_b, mask_b):
    """Masked RMS distance over jointly present (frame, hand) slots."""
    w = (mask_a & mask_b).astype(np.float64)  # (T, H)
    total = w.sum() * N_LANDMARKS * 3
    if total < _EPS:
        return math.inf
    diff = (feat_a.astype(np.float64) - feat_b.astype(np.float64)).reshape(N_FRAMES, N_HANDS, -1)
    sq = (diff ** 2).sum(axis=2)  # (T, H)
    return float(np.sqrt((w * sq).sum() / total))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MotionModel:
    """Nearest-sequence-template dynamic gesture model with rejection."""

    def __init__(self):
        self.templates_ = None       # (M, N_FRAMES, 2, 21, 3) float32
        self.masks_ = None           # (M, N_FRAMES, 2) bool
        self.labels_ = None          # (M,) str
        self.reject_threshold_ = None
        self.margin_threshold_ = None
        self.loo_report_ = {}

    # -- training ---------------------------------------------------------

    def fit(self, trials):
        if not trials:
            raise MotionModelError("no training trials")
        feats, masks, labels = [], [], []
        for t in trials:
            _validate_trial(t, require_label=True)
            f, m = preprocess_trial(t)
            feats.append(f)
            masks.append(m)
            labels.append(str(t["label"]))
        self.templates_ = np.stack(feats)
        self.masks_ = np.stack(masks)
        self.labels_ = np.array(labels)

        stats = self._leave_one_out_stats()
        self._calibrate_thresholds(stats)
        return self

    def _distances_to_templates(self, feat, mask, exclude=None):
        dists = np.empty(len(self.labels_), dtype=np.float64)
        for i in range(len(self.labels_)):
            if exclude is not None and i == exclude:
                dists[i] = math.inf
            else:
                dists[i] = masked_distance(feat, mask, self.templates_[i], self.masks_[i])
        return dists

    def _leave_one_out_stats(self):
        """For each train trial: nearest distance/label and nearest
        different-label distance, excluding itself."""
        stats = []
        for i in range(len(self.labels_)):
            dists = self._distances_to_templates(
                self.templates_[i], self.masks_[i], exclude=i)
            order = np.argsort(dists)
            d1 = dists[order[0]]
            pred = self.labels_[order[0]]
            d2 = math.inf
            for j in order:
                if self.labels_[j] != self.labels_[i]:
                    d2 = dists[j]
                    break
            stats.append({
                "true": self.labels_[i],
                "pred": pred,
                "d1": float(d1),
                "gap": float(d2 - d1),
                "correct": bool(pred == self.labels_[i]),
            })
        return stats

    def _calibrate_thresholds(self, stats):
        """Grid-search (reject, margin) on LOO stats only. Never sees test."""
        d1s = sorted({s["d1"] for s in stats})
        gaps = sorted({s["gap"] for s in stats})
        candidates_r = [0.0] + d1s + [d1s[-1] * 1.25 + _EPS]
        candidates_m = [0.0] + [g for g in gaps if g > 0]
        best = None
        for r in candidates_r:
            for m in candidates_m:
                n_ok = n_bad = n_miss = 0
                for s in stats:
                    accepted = s["d1"] <= r + _EPS and s["gap"] >= m - _EPS
                    if accepted:
                        if s["correct"]:
                            n_ok += 1
                        else:
                            n_bad += 1
                    elif s["correct"]:
                        n_miss += 1
                # prioritize zero wrong accepts, then coverage, then margin
                objective = (n_ok - 3.0 * n_bad - 0.25 * n_miss, -n_bad, n_ok, m)
                if best is None or objective > best[0]:
                    best = (objective, r, m, (n_ok, n_bad, n_miss))
        _, r, m, (n_ok, n_bad, n_miss) = best
        self.reject_threshold_ = float(max(r, _EPS))
        self.margin_threshold_ = float(max(m, 0.0))
        n = len(stats)
        self.loo_report_ = {
            "n_trials": n,
            "raw_accuracy": float(np.mean([s["correct"] for s in stats])),
            "accepted_correct": n_ok,
            "accepted_wrong": n_bad,
            "rejected_would_be_correct": n_miss,
            "reject_threshold": self.reject_threshold_,
            "margin_threshold": self.margin_threshold_,
        }

    # -- inference ----------------------------------------------------------

    def _match(self, trial):
        if self.templates_ is None:
            raise MotionModelError("model is not fitted")
        feat, mask = preprocess_trial(trial)
        dists = self._distances_to_templates(feat, mask)
        order = np.argsort(dists)
        d1 = float(dists[order[0]])
        label1 = str(self.labels_[order[0]])
        d2 = math.inf
        for j in order:
            if self.labels_[j] != label1:
                d2 = float(dists[j])
                break
        return label1, d1, float(d2 - d1)

    def _score(self, d1):
        # Heuristic matching score (not a probability). 0.75 <=> the match
        # distance is half of the calibrated reject distance.
        return float(np.clip(1.0 - d1 / (2.0 * self.reject_threshold_), 0.0, 1.0))

    def classify(self, trial):
        """Full classification including REST: (label, score, accepted).

        REST is returned as a normal label here; accepted is False when the
        reject distance or runner-up margin rule fails.
        """
        label, d1, gap = self._match(trial)
        accepted = (d1 <= self.reject_threshold_ + _EPS
                    and gap >= self.margin_threshold_ - _EPS)
        score = self._score(d1) if accepted else 0.0
        return label, score, bool(accepted)

    def predict(self, trial):
        """Runtime output: (label or None, score). REST and rejected input
        both yield (None, 0.0) — REST must never produce output."""
        label, score, accepted = self.classify(trial)
        if not accepted or label == REST_LABEL:
            return None, 0.0
        return label, score

    # -- persistence (NPZ only, no pickle) ---------------------------------

    def save(self, path):
        if self.templates_ is None:
            raise MotionModelError("model is not fitted")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({
            "format": "motion-template-v1",
            "n_frames": N_FRAMES,
            "labels": sorted(set(self.labels_.tolist())),
            "score_semantics": "score = clip(1 - d1/(2*reject_threshold), 0, 1); heuristic, not a probability",
            "continuous_recognition": False,
        })
        np.savez(
            path,
            templates=self.templates_,
            masks=self.masks_,
            labels=self.labels_,
            reject_threshold=np.float64(self.reject_threshold_),
            margin_threshold=np.float64(self.margin_threshold_),
            loo_report=np.array(json.dumps(self.loo_report_)),
            meta=np.array(meta),
        )
        return path

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser()
        with np.load(path, allow_pickle=False) as d:
            model = cls()
            model.templates_ = np.asarray(d["templates"], dtype=np.float32)
            model.masks_ = np.asarray(d["masks"], dtype=bool)
            model.labels_ = np.asarray(d["labels"]).astype(str)
            model.reject_threshold_ = float(d["reject_threshold"])
            model.margin_threshold_ = float(d["margin_threshold"])
            model.loo_report_ = json.loads(str(d["loo_report"]))
        return model

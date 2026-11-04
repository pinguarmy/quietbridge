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
# Fraction of the union of observed hand/frame slots that must overlap.
MIN_JOINT_COVERAGE = 0.5


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
            if "sign_language" in d.files or "bgr_frames" in d.files:
                continue  # Standard-sign checks are not personal gesture training.
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
        label = trial.get("label")
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise MotionModelError("label must be a nonempty string without surrounding whitespace")
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
    feat = np.zeros((N_FRAMES, N_HANDS, N_LANDMARKS, 3), dtype=np.float64)
    mask = np.zeros((N_FRAMES, N_HANDS), dtype=bool)
    for h in range(N_HANDS):
        idx = np.flatnonzero(hp[:, h])
        if idx.size == 0:
            continue
        # Each observed frame owns the interval to its adjacent midpoints.
        # Interpolate only inside contiguous runs of present frames.
        boundaries = (ts[:-1] + ts[1:]) / 2.0
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            start, end = run[0], run[-1]
            left = ts[0] if start == 0 else boundaries[start - 1]
            right = ts[-1] if end == len(ts) - 1 else boundaries[end]
            in_window = (target_t >= left) & (target_t <= right)
            flat = lm[run, h].reshape(run.size, -1)
            interp = np.stack(
                [np.interp(target_t[in_window], ts[run], flat[:, j])
                 for j in range(flat.shape[1])], axis=1,
            ).reshape(-1, N_LANDMARKS, 3)
            feat[in_window, h] = interp
            mask[in_window, h] = True

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


def joint_coverage(mask_a, mask_b):
    """Fraction of observed hand/frame slots present in both sequences."""
    union = np.count_nonzero(mask_a | mask_b)
    return float(np.count_nonzero(mask_a & mask_b) / union) if union else 0.0


def masked_distance(feat_a, mask_a, feat_b, mask_b):
    """Masked RMS distance; reject pairs with insufficient joint coverage."""
    if joint_coverage(mask_a, mask_b) < MIN_JOINT_COVERAGE:
        return math.inf
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
        self.label_thresholds_ = None
        self.calibration_report_ = {}

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
        self.label_thresholds_ = None
        self.calibration_report_ = {
            "calibrated": False,
            "status": "uncalibrated",
            "coverage_rate": 0.0,
            "calibrated_labels": [],
            "fallback_labels": sorted(set(self.labels_.tolist())),
            "reason": "fit only; call calibrate(validation_trials)",
        }
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
                "gap": float(d2 - d1) if math.isfinite(d1) else 0.0,
                "correct": bool(math.isfinite(d1) and pred == self.labels_[i]),
            })
        return stats

    def _calibrate_thresholds(self, stats):
        """Grid-search (reject, margin) on LOO stats only. Never sees test."""
        d1s = sorted({s["d1"] for s in stats if math.isfinite(s["d1"])}) or [0.0]
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

    # -- validation calibration ---------------------------------------------

    def get_reject_threshold(self, label: str | None = None) -> float:
        """Return the reject distance threshold for a given label or global."""
        if label is not None and self.label_thresholds_ and label in self.label_thresholds_:
            return float(self.label_thresholds_[label]["reject_threshold"])
        return float(self.reject_threshold_ if self.reject_threshold_ is not None else _EPS)

    def get_margin_threshold(self, label: str | None = None) -> float:
        """Return the runner-up margin threshold for a given label or global."""
        if label is not None and self.label_thresholds_ and label in self.label_thresholds_:
            return float(self.label_thresholds_[label]["margin_threshold"])
        return float(self.margin_threshold_ if self.margin_threshold_ is not None else 0.0)

    def get_thresholds(self, label: str | None = None) -> tuple[float, float]:
        """Return (reject_threshold, margin_threshold) for a given label or global."""
        return self.get_reject_threshold(label), self.get_margin_threshold(label)

    def _search_threshold(self, matched_trials, target_label, default_r, default_m):
        """Search candidate (r, m) on matched trials.

        Prioritizes zero wrong accepts (mislabels and REST false alarms),
        then maximizes positive recall. Tie-breaks by distance to target_r,
        then larger margin.
        """
        correct_trials = []
        intruder_trials = []
        for s in matched_trials:
            if target_label == "GLOBAL":
                if s["true"] != REST_LABEL and s["matched"] == s["true"]:
                    correct_trials.append(s)
                elif (s["true"] == REST_LABEL and s["matched"] != REST_LABEL) or \
                     (s["true"] != REST_LABEL and s["matched"] != s["true"] and s["matched"] != REST_LABEL):
                    intruder_trials.append(s)
            elif target_label == REST_LABEL:
                if s["true"] == REST_LABEL:
                    correct_trials.append(s)
            else:
                if s["true"] == target_label:
                    correct_trials.append(s)
                else:
                    intruder_trials.append(s)

        cand_r = {0.0, float(default_r)}
        for s in matched_trials:
            d = s["d1"]
            if math.isfinite(d) and d >= 0.0:
                cand_r.add(d)
                cand_r.add(d * 1.05)
                cand_r.add(d * 1.15)
                cand_r.add(d * 1.25)
        for s in intruder_trials:
            cand_r.add(max(0.0, s["d1"] - _EPS))
            for c in correct_trials:
                if c["d1"] < s["d1"]:
                    cand_r.add((c["d1"] + s["d1"]) / 2.0)

        cand_m = {0.0, float(default_m)}
        for s in matched_trials:
            g = s["gap"]
            if math.isfinite(g) and g >= 0.0:
                cand_m.add(g)
                cand_m.add(g * 0.9)
        for s in intruder_trials:
            cand_m.add(s["gap"] + _EPS)
            for c in correct_trials:
                if c["gap"] > s["gap"]:
                    cand_m.add((c["gap"] + s["gap"]) / 2.0)

        d_correct = [s["d1"] for s in correct_trials if math.isfinite(s["d1"])]
        d_intruder = [s["d1"] for s in intruder_trials if math.isfinite(s["d1"])]
        if d_intruder and d_correct:
            min_bad = min(d_intruder)
            max_good = max(d_correct)
            if max_good < min_bad:
                target_r = (max_good + min_bad) / 2.0
            else:
                target_r = max(0.0, min_bad - _EPS)
        elif d_correct:
            max_good = max(d_correct)
            target_r = min(default_r, max_good * 1.25) if default_r >= max_good else max_good * 1.25
        else:
            target_r = default_r

        best = None
        for r in sorted(cand_r):
            if not (math.isfinite(r) and r >= 0.0):
                continue
            for m in sorted(cand_m):
                if not (math.isfinite(m) and m >= 0.0):
                    continue
                n_ok = 0
                n_bad = 0
                for s in matched_trials:
                    accepted = (s["d1"] <= r + _EPS) and (s["gap"] >= m - _EPS)
                    if not accepted:
                        continue
                    if target_label == "GLOBAL":
                        if s["true"] != REST_LABEL and s["matched"] == s["true"]:
                            n_ok += 1
                        elif (s["true"] == REST_LABEL and s["matched"] != REST_LABEL) or \
                             (s["true"] != REST_LABEL and s["matched"] != s["true"] and s["matched"] != REST_LABEL):
                            n_bad += 1
                    elif target_label == REST_LABEL:
                        if s["true"] == REST_LABEL:
                            n_ok += 1
                    else:
                        if s["true"] == target_label:
                            n_ok += 1
                        else:
                            n_bad += 1
                score = (-n_bad, n_ok, -abs(r - target_r), m)
                if best is None or score > best[0]:
                    best = (score, r, m, n_ok, n_bad)

        if best is None:
            return float(default_r), float(default_m), 0, 0
        _, best_r, best_m, best_ok, best_bad = best
        return float(max(best_r, _EPS)), float(max(best_m, 0.0)), best_ok, best_bad

    def calibrate(self, validation_trials, min_samples=2, per_label=True):
        """Calibrate acceptance thresholds on an independent validation split.

        Never sees test trials. Prioritizes zero mislabels and zero REST false
        alarms, then maximizes positive recall. If a label has fewer than
        `min_samples` validation trials, it falls back to train LOO thresholds
        and is recorded as a fallback in the calibration report.
        """
        if self.templates_ is None:
            raise MotionModelError("model is not fitted")
        if min_samples < 1:
            raise MotionModelError("min_samples must be at least 1")

        unique_labels = sorted(set(self.labels_.tolist()))
        loo_r = float(self.loo_report_.get("reject_threshold", self.reject_threshold_ or _EPS))
        loo_m = float(self.loo_report_.get("margin_threshold", self.margin_threshold_ or 0.0))

        if not validation_trials:
            self.label_thresholds_ = {
                lbl: {
                    "reject_threshold": loo_r,
                    "margin_threshold": loo_m,
                    "source": "train_loo",
                    "status": "fallback_train_loo",
                    "reason": "no_validation_trials",
                    "validation_trials": 0,
                    "matched_trials": 0,
                    "positive_correct": 0,
                    "wrong_accepts": 0,
                }
                for lbl in unique_labels
            }
            self.calibration_report_ = {
                "calibrated": False,
                "status": "no_validation_trials",
                "coverage_rate": 0.0,
                "n_labels_total": len(unique_labels),
                "n_labels_calibrated": 0,
                "calibrated_labels": [],
                "fallback_labels": unique_labels,
                "min_samples": int(min_samples),
                "per_label": self.label_thresholds_,
                "global_threshold": {
                    "reject_threshold": loo_r,
                    "margin_threshold": loo_m,
                    "source": "train_loo",
                },
            }
            return self

        val_stats = []
        for t in validation_trials:
            _validate_trial(t, require_label=True)
            true_label = str(t["label"])
            feat, mask = preprocess_trial(t)
            dists = self._distances_to_templates(feat, mask)
            order = np.argsort(dists)
            nearest_idx = int(order[0])
            d1 = float(dists[nearest_idx])
            if not math.isfinite(d1):
                raw_label = None
                gap = 0.0
            else:
                raw_label = str(self.labels_[nearest_idx])
                d2 = next((float(dists[j]) for j in order if self.labels_[j] != raw_label), math.inf)
                gap = float(d2 - d1)
            val_stats.append({
                "trial_id": str(t.get("trial_id", "")),
                "true": true_label,
                "matched": raw_label,
                "d1": d1,
                "gap": gap,
                "finite": math.isfinite(d1),
            })

        label_thresholds = {}
        for L in unique_labels:
            n_val_true = sum(1 for s in val_stats if s["true"] == L)
            matched_L = [s for s in val_stats if s["matched"] == L and s["finite"]]
            correct_L = [s for s in matched_L if s["true"] == L]

            if n_val_true < min_samples:
                label_thresholds[L] = {
                    "reject_threshold": loo_r,
                    "margin_threshold": loo_m,
                    "source": "train_loo",
                    "status": "fallback_train_loo",
                    "reason": "insufficient_samples" if n_val_true > 0 else "no_samples",
                    "validation_trials": n_val_true,
                    "matched_trials": len(matched_L),
                    "positive_correct": sum(
                        1 for s in matched_L
                        if s["true"] == L and s["d1"] <= loo_r + _EPS and s["gap"] >= loo_m - _EPS
                    ),
                    "wrong_accepts": sum(
                        1 for s in matched_L
                        if s["true"] != L and L != REST_LABEL and s["d1"] <= loo_r + _EPS and s["gap"] >= loo_m - _EPS
                    ),
                }
            elif len(correct_L) == 0:
                label_thresholds[L] = {
                    "reject_threshold": loo_r,
                    "margin_threshold": loo_m,
                    "source": "train_loo",
                    "status": "fallback_train_loo",
                    "reason": "no_correct_validation_matches",
                    "validation_trials": n_val_true,
                    "matched_trials": len(matched_L),
                    "positive_correct": 0,
                    "wrong_accepts": sum(
                        1 for s in matched_L
                        if s["true"] != L and L != REST_LABEL and s["d1"] <= loo_r + _EPS and s["gap"] >= loo_m - _EPS
                    ),
                }
            else:
                r_cal, m_cal, n_ok, n_bad = self._search_threshold(
                    matched_L, L, loo_r, loo_m
                )
                label_thresholds[L] = {
                    "reject_threshold": r_cal,
                    "margin_threshold": m_cal,
                    "source": "validation",
                    "status": "calibrated",
                    "reason": None,
                    "validation_trials": n_val_true,
                    "matched_trials": len(matched_L),
                    "positive_correct": n_ok,
                    "wrong_accepts": n_bad,
                }

        total_val_positive = sum(1 for s in val_stats if s["true"] != REST_LABEL)
        if total_val_positive < min_samples:
            global_r, global_m = loo_r, loo_m
            global_source = "train_loo"
        else:
            global_r, global_m, g_ok, g_bad = self._search_threshold(
                [s for s in val_stats if s["finite"]], "GLOBAL", loo_r, loo_m
            )
            global_source = "validation"

        self.reject_threshold_ = global_r
        self.margin_threshold_ = global_m

        calibrated_labels = [l for l, d in label_thresholds.items() if d["status"] == "calibrated"]
        fallback_labels = [l for l, d in label_thresholds.items() if d["status"] != "calibrated"]
        coverage_rate = len(calibrated_labels) / len(unique_labels) if unique_labels else 0.0

        if per_label:
            self.label_thresholds_ = label_thresholds
        else:
            self.label_thresholds_ = None

        self.calibration_report_ = {
            "calibrated": len(calibrated_labels) > 0,
            "status": "calibrated" if len(calibrated_labels) > 0 else "all_fallback",
            "coverage_rate": coverage_rate,
            "n_labels_total": len(unique_labels),
            "n_labels_calibrated": len(calibrated_labels),
            "calibrated_labels": calibrated_labels,
            "fallback_labels": fallback_labels,
            "min_samples": int(min_samples),
            "per_label": label_thresholds,
            "global_threshold": {
                "reject_threshold": global_r,
                "margin_threshold": global_m,
                "source": global_source,
            },
        }
        return self

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
        return label1, d1, float(d2 - d1) if math.isfinite(d1) else 0.0

    def diagnose(self, trial):
        """Return match details without changing the model or runtime output.

        Coverage is the intersection divided by the union of present
        hand/frame slots for the nearest template. A zero-overlap trial has
        no raw label; score is the ungated heuristic score.
        """
        if self.templates_ is None:
            raise MotionModelError("model is not fitted")
        feat, mask = preprocess_trial(trial)
        dists = self._distances_to_templates(feat, mask)
        order = np.argsort(dists)
        index = int(order[0])
        d1 = float(dists[index])
        if not math.isfinite(d1):
            coverage = max(joint_coverage(mask, m) for m in self.masks_)
            return {"raw_label": None, "d1": d1, "gap": 0.0, "score": 0.0,
                    "accepted": False, "reject_reason": "insufficient_coverage",
                    "joint_coverage": coverage}
        label = str(self.labels_[index])
        d2 = next((float(dists[j]) for j in order if self.labels_[j] != label), math.inf)
        gap = d2 - d1
        r = self.get_reject_threshold(label)
        m = self.get_margin_threshold(label)
        if d1 > r + _EPS:
            reason = "distance"
        elif gap < m - _EPS:
            reason = "margin"
        else:
            reason = None
        return {"raw_label": label, "d1": d1, "gap": gap,
                "score": self._score(d1, label=label), "accepted": reason is None,
                "reject_reason": reason,
                "joint_coverage": joint_coverage(mask, self.masks_[index])}

    def _score(self, d1, label=None):
        # Heuristic matching score (not a probability). 0.75 <=> the match
        # distance is half of the calibrated reject distance.
        r = self.get_reject_threshold(label)
        if r is None or r < _EPS:
            r = self.reject_threshold_ or 1.0
        return float(np.clip(1.0 - d1 / (2.0 * r), 0.0, 1.0))

    def classify(self, trial):
        """Full classification including REST: (label, score, accepted).

        REST is returned as a normal label here; accepted is False when the
        reject distance or runner-up margin rule fails.
        """
        result = self.diagnose(trial)
        return (result["raw_label"], result["score"] if result["accepted"] else 0.0,
                result["accepted"])

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
        cal_str = json.dumps(self.calibration_report_) if self.calibration_report_ else ""
        lt_str = json.dumps(self.label_thresholds_) if self.label_thresholds_ is not None else ""
        np.savez(
            path,
            templates=self.templates_,
            masks=self.masks_,
            labels=self.labels_,
            reject_threshold=np.float64(self.reject_threshold_),
            margin_threshold=np.float64(self.margin_threshold_),
            loo_report=np.array(json.dumps(self.loo_report_)),
            calibration_report=np.array(cal_str),
            label_thresholds=np.array(lt_str),
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
            if "label_thresholds" in d:
                raw_lt = str(d["label_thresholds"])
                model.label_thresholds_ = json.loads(raw_lt) if raw_lt else None
            else:
                model.label_thresholds_ = None
            if "calibration_report" in d:
                raw_cal = str(d["calibration_report"])
                model.calibration_report_ = json.loads(raw_cal) if raw_cal else {}
            else:
                model.calibration_report_ = {}
        return model

"""Record mouth-crop training clips from frames the browser uploads.

The browser already holds the user's camera for the demo and posts each frame here, so
this module never opens a camera and never loads the ~1 GB VSR model. Only aligned
96x96 grayscale mouth crops plus the prompt text are written to disk (the user has
consented to recording training data) — no raw frames, video or audio. Everything stays
local; there is no network or cloud access.

Threading: the lead lab serialises every call under its global lock; this class starts
no threads and no workers.
"""
from __future__ import annotations

import os
import re
import time
import uuid

import numpy as np

DEFAULT_ROOT = os.path.expanduser("~/.local/share/quietbridge/mouths")
SPLITS = ("train", "validation", "test")
MAX_TEXT = 300
LIMIT_SECONDS = 30.0
MIN_ROIS = 12
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class MouthCapture:
    def __init__(self, data_root=None, tracker_factory=None):
        self.root = os.path.expanduser(data_root or DEFAULT_ROOT)
        self._tracker_factory = tracker_factory
        self._tracker = None  # built lazily on the first start()
        self._rec = None
        self._meta = None  # (text, split, session) of the open/failed recording
        self._recording = False
        self._limit = False
        self._mouth_box = None
        self._outer_lips = []
        self._inner_lips = []
        self._face = False
        self._preview = False
        self.error = None
        self.last_saved = None
        self.saved_count = 0
        self._last_quality = 0.0

    def set_root(self, data_root):
        if self._recording:
            raise ValueError("Cannot change data root while recording is in progress")
        self.root = os.path.expanduser(str(data_root))

    # -- lifecycle ---------------------------------------------------
    def _ensure_tracker(self):
        if self._tracker is None:
            factory = self._tracker_factory
            if factory is None:
                from lipflow.face import FaceTracker
                factory = FaceTracker
            self._tracker = factory()
        return self._tracker

    def preview(self):
        """Explicit local face tracking, without collecting a training clip."""
        self._ensure_tracker()
        self._preview = True
        return self.snapshot()

    def start(self, text: str, split: str, session: str) -> dict:
        text = (text or "").strip()
        if not text or len(text) > MAX_TEXT:
            raise ValueError(f"text must be 1-{MAX_TEXT} characters")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not _SESSION_RE.match(session or ""):
            raise ValueError("session must be 1-64 chars of [A-Za-z0-9_-]")
        if self._rec is not None:
            raise ValueError("a recording is still open; stop or discard it first")
        self._check_session_split(session, split)
        from lipflow.camera import Recording
        self._ensure_tracker()
        self._rec = Recording(started=time.perf_counter())
        self._meta = (text, split, session)
        self._recording, self._limit, self.error = True, False, None
        return self.snapshot()

    def _check_session_split(self, session: str, split: str):
        """A session id belongs to one split; scan saved npz metadata to enforce it."""
        for other in SPLITS:
            folder = os.path.join(self.root, other)
            names = os.listdir(folder) if os.path.isdir(folder) else []
            for name in names:
                if not name.endswith(".npz"):
                    continue
                try:
                    with np.load(os.path.join(folder, name), allow_pickle=False) as z:
                        if str(z["session_id"]) == session and \
                           str(z["split"] if "split" in z.files else other) != split:
                            raise ValueError(
                                f"session {session!r} is already used in another split")
                except ValueError:
                    raise
                except Exception:
                    continue

    # -- frames ------------------------------------------------------
    def feed(self, frame_bgr, timestamp_ms: int):
        """Consume one browser frame. Returns the latest real mouth box, or None."""
        active = self._recording and self._rec is not None
        if not active and not self._preview:
            self._mouth_box = None
            self._outer_lips = self._inner_lips = []
            return None
        import cv2
        from lipflow.camera import face_crop
        rec = self._rec
        ts = timestamp_ms / 1000.0
        obs = self._tracker.detect(frame_bgr, timestamp_ms)
        if active:
            if rec.ts and ts - rec.ts[0] > LIMIT_SECONDS:
                self._limit = True  # tracking continues, but capture is capped
            else:
                rec.ts.append(ts)
                rec.grays.append(face_crop(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), obs, rec))
                rec.anchors.append(obs.anchors if obs else None)
                rec.mouth_open.append(obs.mouth_open if obs else 0.0)
        self._face = obs is not None
        self._mouth_box = None
        self._outer_lips = self._inner_lips = []
        if obs is not None:
            h, w = frame_bgr.shape[:2]
            scale = np.array([w, h], dtype=np.float32)
            self._outer_lips = (obs.outer_lips / scale).round(5).tolist()
            self._inner_lips = (obs.inner_lips / scale).round(5).tolist()
            x0, y0 = obs.outer_lips.min(0)
            x1, y1 = obs.outer_lips.max(0)
            self._mouth_box = [round(float(x0 / w), 4), round(float(y0 / h), 4),
                               round(float((x1 - x0) / w), 4), round(float((y1 - y0) / h), 4)]
        return self._mouth_box

    # -- finish ------------------------------------------------------
    def stop(self) -> dict:
        rec, meta = self._rec, self._meta
        if rec is None or meta is None:
            raise ValueError("no recording to stop")
        try:
            from lipflow.dictation import clip_problem, rois_for
            problem = clip_problem(rec)
            if problem:
                raise ValueError(": ".join(problem))
            rois = rois_for(rec)
            if rois is None or len(rois) < MIN_ROIS:
                raise ValueError("not enough visible mouth frames")
            path = self._save(rec, np.asarray(rois), meta)
        except Exception as e:
            self.error = str(e)  # keep the recording; call discard() to drop it
            raise
        self._last_quality = rec.face_ratio
        self._clear()
        self._mouth_box = None  # after a save the stale box must not look live
        self.last_saved, self.saved_count = path, self.saved_count + 1
        return self.snapshot()

    def _save(self, rec, rois, meta) -> str:
        text, split, session = meta
        trial = uuid.uuid4().hex
        folder = os.path.join(self.root, split)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{session}_{trial}.npz")
        ts_ms = np.asarray([int(round(t * 1000)) for t in rec.ts], dtype=np.int64)
        with open(path, "xb") as f:  # never overwrite an existing trial
            np.savez(f, text=np.array(text), rois=rois, timestamps_ms=ts_ms,
                     session_id=np.array(session), trial_id=np.array(trial),
                     split=np.array(split))
        return path

    def discard(self) -> dict:
        """Drop the open or failed recording without saving anything."""
        self._clear()
        self.error = None
        return self.snapshot()

    cancel = discard

    def _clear(self):
        self._outer_lips = self._inner_lips = []
        self._mouth_box = None
        self._rec = self._meta = None
        self._recording = self._limit = False
        self._face = False

    def snapshot(self) -> dict:
        rec = self._rec
        quality = rec.face_ratio if rec is not None and rec.anchors else self._last_quality
        return {"recording": bool(self._recording and rec is not None),
                "frames": len(rec.ts) if rec is not None else 0,
                "mouth_box": self._mouth_box,
                "outer_lips": self._outer_lips,
                "inner_lips": self._inner_lips,
                "face": bool(self._face and (self._recording or self._preview)),
                "preview": self._preview,
                "quality": round(float(quality), 3),
                "limit_reached": self._limit,
                "error": self.error,
                "last_saved": self.last_saved,
                "saved_count": self.saved_count}

    def close(self):
        self._preview = False
        self._clear()
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

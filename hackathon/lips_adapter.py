"""Import the existing engine without editing it or opening a second camera."""
from __future__ import annotations

import hashlib
import time

from .contracts import LipResult


def _file_sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LipsAdapter:
    def __init__(self, model_path=None, language_model_path=None, beam_size=4):
        from lipflow.vsr import LipReader
        from lipflow.face import FaceTracker
        self.reader = LipReader(beam_size=beam_size, personal=False,
                                personal_vsr_path=model_path,
                                personal_lm_path=language_model_path)
        self.model_info = {
            "mode": "personal" if self.reader.personal_vsr else "stock",
            "beam_size": beam_size,
            "vsr_path": self.reader.personal_vsr_source,
            "vsr_sha256": _file_sha256(self.reader.personal_vsr_source),
            "language_model_path": self.reader.personal_lm_source,
            "language_model_sha256": _file_sha256(self.reader.personal_lm_source),
        }
        self.reader.warmup()
        self.tracker = FaceTracker()
        self.recording = None
        self.last_tracking = None

    def start(self, timestamp_ms=None):
        from lipflow.camera import Recording
        self.recording = Recording(started=time.perf_counter())
        self.record_start_ms = timestamp_ms

    def process(self, frame_bgr, timestamp_ms: int):
        obs = self.tracker.detect(frame_bgr, timestamp_ms)
        self.last_tracking = None
        if obs is not None:
            height, width = frame_bgr.shape[:2]
            x0, y0 = obs.outer_lips.min(0)
            x1, y1 = obs.outer_lips.max(0)
            self.last_tracking = {
                "mouth_box": [float(x0 / width), float(y0 / height),
                              float((x1 - x0) / width), float((y1 - y0) / height)],
                "outer_lips": [[float(x / width), float(y / height)] for x, y in obs.outer_lips],
                "inner_lips": [[float(x / width), float(y / height)] for x, y in obs.inner_lips],
            }
        if self.recording is None:
            return
        import cv2
        from lipflow.camera import face_crop
        rec = self.recording
        rec.ts.append(timestamp_ms / 1000)
        rec.grays.append(face_crop(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), obs, rec))
        rec.anchors.append(obs.anchors if obs else None)
        rec.mouth_open.append(obs.mouth_open if obs else 0)

    def finish(self) -> LipResult:
        from lipflow.dictation import clip_problem, rois_for
        rec, self.recording = self.recording, None
        if rec is None:
            return LipResult([], 0)
        problem = clip_problem(rec)
        if problem:
            raise ValueError(": ".join(problem))
        started = time.perf_counter()
        rois = rois_for(rec)
        if rois is None or len(rois) < 12:
            raise ValueError("Not enough visible mouth frames; face the camera and try again")
        details = self.reader.beam_search_details(self.reader.encode(rois), nbest=3)
        duration = rec.ts[-1] - rec.ts[0] if len(rec.ts) > 1 else 0.0
        capture_fps = (len(rec.ts) - 1) / duration if duration > 0 else None
        max_gap = max((b - a for a, b in zip(rec.ts, rec.ts[1:])), default=0.0)
        return LipResult(
            [item["text"] for item in details],
            round((time.perf_counter() - started) * 1000),
            candidate_scores=[item["score"] for item in details],
            start_ms=round(rec.ts[0] * 1000) if rec.ts else None,
            end_ms=round(rec.ts[-1] * 1000) if rec.ts else None,
            raw_frames=len(rec.ts),
            capture_fps=round(capture_fps, 2) if capture_fps is not None else None,
            max_frame_gap_ms=round(max_gap * 1000),
        )

    def close(self):
        self.tracker.close()

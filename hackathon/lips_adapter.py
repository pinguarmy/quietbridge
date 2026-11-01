"""Import the existing engine without editing it or opening a second camera."""
from __future__ import annotations

import time

from .contracts import LipResult


class LipsAdapter:
    def __init__(self):
        from lipflow.vsr import LipReader
        from lipflow.face import FaceTracker
        self.reader = LipReader(beam_size=4, personal=False)
        self.reader.warmup()
        self.tracker = FaceTracker()
        self.recording = None
        self.last_tracking = None

    def start(self):
        from lipflow.camera import Recording
        self.recording = Recording(started=time.perf_counter())

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
        candidates = self.reader.beam_search(self.reader.encode(rois), nbest=3)
        return LipResult(candidates, round((time.perf_counter() - started) * 1000))

    def close(self):
        self.tracker.close()

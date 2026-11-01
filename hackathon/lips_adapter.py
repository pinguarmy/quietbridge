"""Import the existing engine without editing it or opening a second camera."""
from __future__ import annotations

import os
from pathlib import Path
import time

from .contracts import LipResult


PERSONAL_PREFIXES = ("encoder.frontend.", "encoder.embed.", "encoder.encoders.0.")


def load_personal_state(reader, path):
    """Validate an explicitly selected adaptation before patching the stock reader."""
    import torch

    state = torch.load(Path(path).expanduser(), weights_only=True, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError("Personal lip checkpoint must contain a nonempty tensor state dictionary")
    reference = reader.model.state_dict()
    for key, value in state.items():
        if not isinstance(key, str) or not key.startswith(PERSONAL_PREFIXES) or key not in reference:
            raise ValueError(f"Unsupported personal lip tensor: {key!r}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Personal lip tensor required: {key}")
        if value.shape != reference[key].shape or value.dtype != reference[key].dtype:
            raise ValueError(f"Personal lip tensor shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite personal lip tensor: {key}")
    # A partial adaptation intentionally omits frozen stock tensors. All provided
    # keys have already been checked, so strict=False cannot hide unexpected keys.
    result = reader.model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected personal lip tensors: {result.unexpected_keys}")
    reader.personal_vsr = True


def _sync_encoder(reader):
    """Measure completed MPS work rather than just asynchronous dispatch."""
    import torch

    device = getattr(reader, "enc_device", None)
    if device is not None and torch.device(device).type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


class LipsAdapter:
    def __init__(self, personal_model_path=None):
        from lipflow.vsr import LipReader
        from lipflow.face import FaceTracker
        self.reader = LipReader(beam_size=4, personal=False)
        selected = personal_model_path if personal_model_path is not None else os.environ.get("QUIETBRIDGE_LIP_MODEL")
        self.model_name = "stock"
        self.model_path = None
        if selected is not None:
            self.model_path = str(Path(selected).expanduser().resolve())
            load_personal_state(self.reader, self.model_path)
            self.model_name = "personal (welcome pilot)"
        self.last_metrics = {}
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
        self.last_metrics = {}
        if rec is None:
            return LipResult([], 0)
        problem = clip_problem(rec)
        if problem:
            raise ValueError(": ".join(problem))
        started = time.perf_counter()
        rois = rois_for(rec)
        if rois is None or len(rois) < 12:
            raise ValueError("Not enough visible mouth frames; face the camera and try again")
        roi_finished = time.perf_counter()
        _sync_encoder(self.reader)
        encode_started = time.perf_counter()
        encoded = self.reader.encode(rois)
        _sync_encoder(self.reader)
        encode_finished = time.perf_counter()
        candidates = self.reader.beam_search(encoded, nbest=3)
        # A repeated six-word phrase in one short clip is a known decoder failure,
        # not evidence of user intent. This guard does not validate other outputs.
        def repetitive(text):
            words = text.upper().split()
            grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
            return len(grams) != len(set(grams))
        candidates = [text for text in candidates if not repetitive(text)]
        if not candidates:
            raise ValueError("Unreliable repeated decoder output. No words added; please retry a short phrase.")
        _sync_encoder(self.reader)
        finished = time.perf_counter()
        self.last_metrics = {
            "roi_ms": (roi_finished - started) * 1000,
            "encode_ms": (encode_finished - encode_started) * 1000,
            "search_ms": (finished - encode_finished) * 1000,
        }
        return LipResult(candidates, round((finished - started) * 1000))

    def close(self):
        self.tracker.close()

"""Import the existing engine without editing it or opening a second camera."""
from __future__ import annotations

import hashlib
import os
from collections import deque
from pathlib import Path
import time

from .contracts import LipResult


PERSONAL_PREFIXES = ("encoder.frontend.", "encoder.embed.", "encoder.encoders.0.")


def _file_sha256(path):
    if path is None:
        return None
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return None
        digest = hashlib.sha256()
        with open(p, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


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
    def __init__(self, model_path=None, language_model_path=None, beam_size=4, *, personal_model_path=None):
        from lipflow.vsr import LipReader
        from lipflow.face import FaceTracker

        selected = personal_model_path if personal_model_path is not None else model_path
        if selected is None:
            selected = os.environ.get("QUIETBRIDGE_LIP_MODEL")

        reader_kwargs = {"beam_size": beam_size, "personal": False}
        if language_model_path is not None:
            reader_kwargs["personal_lm_path"] = language_model_path

        self.reader = LipReader(**reader_kwargs)
        self.model_name = "stock"
        self.model_path = None
        if selected is not None:
            self.model_path = str(Path(selected).expanduser().resolve())
            load_personal_state(self.reader, self.model_path)
            self.reader.personal_vsr_source = self.model_path
            self.model_name = "personal (welcome pilot)"

        vsr_source = getattr(self.reader, "personal_vsr_source", None) or self.model_path
        lm_source = getattr(self.reader, "personal_lm_source", None) or language_model_path
        self.model_info = {
            "mode": "personal" if getattr(self.reader, "personal_vsr", False) else "stock",
            "beam_size": beam_size,
            "vsr_path": vsr_source,
            "vsr_sha256": _file_sha256(vsr_source),
            "language_model_path": lm_source,
            "language_model_sha256": _file_sha256(lm_source),
        }
        self.last_metrics = {}
        self.reader.warmup()
        self.tracker = FaceTracker()
        self.recording = None
        self.last_tracking = None
        self.last_mouth_open = None
        self.record_start_ms = None
        self.pre_roll_ms = 300
        self._preroll = deque(maxlen=20)
        self._last_window_end_ms = None

    def start(self, timestamp_ms=None):
        from lipflow.camera import Recording, face_crop
        import cv2

        self.recording = Recording(started=time.perf_counter())
        self.record_start_ms = timestamp_ms

        preroll = getattr(self, "_preroll", None)
        if preroll:
            pre_ms = getattr(self, "pre_roll_ms", 300)
            cutoff = (timestamp_ms - pre_ms) if timestamp_ms is not None else None
            last_end = getattr(self, "_last_window_end_ms", None)
            rec = self.recording
            for f_bgr, t_ms, obs in list(preroll):
                if cutoff is not None and t_ms < cutoff:
                    continue
                if timestamp_ms is not None and t_ms > timestamp_ms:
                    continue
                if last_end is not None and t_ms <= last_end:
                    continue
                try:
                    gray = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY)
                    crop = face_crop(gray, obs, rec)
                except Exception:
                    crop = None
                rec.ts.append(t_ms / 1000)
                rec.grays.append(crop)
                rec.anchors.append(obs.anchors if obs else None)
                rec.mouth_open.append(getattr(obs, "mouth_open", 0.0) if obs else 0.0)
            preroll.clear()

    def process(self, frame_bgr, timestamp_ms: int):
        obs = self.tracker.detect(frame_bgr, timestamp_ms)
        self.last_tracking = None
        self.last_mouth_open = None
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
            self.last_mouth_open = getattr(obs, "mouth_open", None)

        if self.recording is None:
            if not hasattr(self, "_preroll") or self._preroll is None:
                self._preroll = deque(maxlen=20)
            self._preroll.append((frame_bgr, timestamp_ms, obs))
            pre_ms = getattr(self, "pre_roll_ms", 300)
            last_end = getattr(self, "_last_window_end_ms", None)
            while self._preroll and (
                timestamp_ms - self._preroll[0][1] > pre_ms + 100
                or (last_end is not None and self._preroll[0][1] <= last_end)
            ):
                self._preroll.popleft()
            return

        import cv2
        from lipflow.camera import face_crop
        rec = self.recording
        rec.ts.append(timestamp_ms / 1000)
        rec.grays.append(face_crop(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), obs, rec))
        rec.anchors.append(obs.anchors if obs else None)
        rec.mouth_open.append(getattr(obs, "mouth_open", 0.0) if obs else 0.0)

    def cancel(self, timestamp_ms=None):
        self.recording = None
        if getattr(self, "_preroll", None) is not None:
            self._preroll.clear()
        if timestamp_ms is not None:
            self._last_window_end_ms = timestamp_ms

    def reset(self):
        self.cancel()
        self._last_window_end_ms = None

    def finish(self) -> LipResult:
        from lipflow.dictation import clip_problem, rois_for
        rec, self.recording = self.recording, None
        self.last_metrics = {}
        if getattr(self, "_preroll", None) is not None:
            self._preroll.clear()
        if rec is None:
            return LipResult([], 0)
        ts = getattr(rec, "ts", None) or []
        if ts:
            self._last_window_end_ms = round(ts[-1] * 1000)
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

        if hasattr(self.reader, "beam_search_details"):
            details = self.reader.beam_search_details(encoded, nbest=3)
        else:
            search_res = self.reader.beam_search(encoded, nbest=3)
            if isinstance(search_res, str):
                search_res = [search_res]
            details = [{"text": t, "score": None} for t in search_res]

        # A repeated six-word phrase in one short clip is a known decoder failure,
        # not evidence of user intent. This guard does not validate other outputs.
        def repetitive(text):
            words = text.upper().split()
            grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
            return len(grams) != len(set(grams))

        details = [item for item in details if not repetitive(item["text"])]
        if not details:
            raise ValueError("Unreliable repeated decoder output. No words added; please retry a short phrase.")

        _sync_encoder(self.reader)
        finished = time.perf_counter()
        self.last_metrics = {
            "roi_ms": (roi_finished - started) * 1000,
            "encode_ms": (encode_finished - encode_started) * 1000,
            "search_ms": (finished - encode_finished) * 1000,
        }

        ts = getattr(rec, "ts", None) or []
        duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        capture_fps = (len(ts) - 1) / duration if duration > 0 else None
        max_gap = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
        candidate_scores = [item["score"] for item in details]

        return LipResult(
            candidates=[item["text"] for item in details],
            decode_ms=round((finished - started) * 1000),
            candidate_scores=candidate_scores if any(s is not None for s in candidate_scores) else [],
            start_ms=round(ts[0] * 1000) if ts else None,
            end_ms=round(ts[-1] * 1000) if ts else None,
            raw_frames=len(ts),
            capture_fps=round(capture_fps, 2) if capture_fps is not None else None,
            max_frame_gap_ms=round(max_gap * 1000) if ts else None,
        )

    def close(self):
        self.tracker.close()

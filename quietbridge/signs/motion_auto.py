"""Opt-in wrist-motion segmentation; defaults are uncalibrated starting points."""
from collections import deque
from dataclasses import dataclass
from math import dist, isfinite


@dataclass(frozen=True)
class MotionAutoConfig:
    # Distances are in MediaPipe normalized image coordinates per second.
    start_speed: float = 0.18
    rest_speed: float = 0.08
    rest_ms: int = 300
    pre_roll_ms: int = 200
    post_roll_ms: int = 120
    min_motion_frames: int = 3
    max_window_ms: int = 3000
    max_frames: int = 180
    duplicate_ms: int = 800


class MotionAutoSegmenter:
    """Return bounded complete windows; lost tracking and timeouts are discarded."""

    def __init__(self, config=None):
        self.config = config or MotionAutoConfig()
        self.reset()

    def reset(self):
        self._pre = deque(maxlen=self.config.max_frames)
        self._previous = None
        self._frames = None
        self._start = None
        self._rest_start = None
        self._moving_frames = 0
        self._blocked = False
        self._blocked_rest = None

    @staticmethod
    def _wrists(hands):
        wrists = {}
        for i, hand in enumerate(hands[:2]):
            points = hand['landmarks']
            if len(points) != 21 or not all(isfinite(v) for p in points for v in p):
                continue
            key = hand['handedness'].lower() or str(i)
            wrists[key] = points[0][:2]
        return wrists

    def update(self, timestamp_ms, hands):
        cfg = self.config
        wrists = self._wrists(hands)
        previous = self._previous
        self._previous = (timestamp_ms, wrists)
        if not wrists:
            self._frames = None
            self._pre.clear()
            self._blocked = True
            self._blocked_rest = None
            return None
        common = wrists.keys() & previous[1].keys() if previous else set()
        if self._frames is not None and previous and wrists.keys() != previous[1].keys():
            self._frames = None
            self._blocked = True
            self._blocked_rest = None
            self._pre.clear()
            return None
        if not common:
            if self._frames is not None:
                self._frames = None
                self._blocked = True
            self._pre.clear()
            self._pre.append((timestamp_ms, hands))
            return None
        dt = (timestamp_ms - previous[0]) / 1000
        speed = max(dist(wrists[k], previous[1][k]) / dt for k in common)
        if self._blocked:
            if speed <= cfg.rest_speed:
                if self._blocked_rest is None:
                    self._blocked_rest = timestamp_ms
                if timestamp_ms - self._blocked_rest >= cfg.rest_ms:
                    self._blocked = False
                    self._pre.clear()
            else:
                self._blocked_rest = None
            return None
        if self._frames is None:
            while self._pre and timestamp_ms - self._pre[0][0] > cfg.pre_roll_ms:
                self._pre.popleft()
            if speed >= cfg.start_speed:
                self._frames = list(self._pre)
                self._start = self._frames[0][0] if self._frames else timestamp_ms
                self._frames.append((timestamp_ms, hands))
                self._moving_frames = 1
                self._rest_start = None
            else:
                self._pre.append((timestamp_ms, hands))
            return None
        self._frames.append((timestamp_ms, hands))
        if timestamp_ms - self._start > cfg.max_window_ms or len(self._frames) > cfg.max_frames:
            self._frames = None
            self._blocked = True
            self._blocked_rest = None
            return None
        if speed > cfg.rest_speed:
            self._moving_frames += 1
            self._rest_start = None
            return None
        if self._rest_start is None:
            self._rest_start = timestamp_ms
        if timestamp_ms - self._rest_start < cfg.rest_ms:
            return None
        frames = [f for f in self._frames if f[0] <= self._rest_start + cfg.post_roll_ms]
        valid = self._moving_frames >= cfg.min_motion_frames
        self._frames = None
        self._pre.clear()
        self._pre.append((timestamp_ms, hands))
        return frames if valid else None

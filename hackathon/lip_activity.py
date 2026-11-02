"""Mouth activity endpoint state machine for continuous camera sessions.

Pure-Python state machine with zero external dependencies.
Tracks mouth opening and face visibility to start capture on lip activity
and stop capture after quiet pauses (~600-800ms) or maximum chunk duration.
Geometric segmentation hint only; not speech confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActivityState(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class ActivityAction(Enum):
    NONE = "none"
    START = "start"
    STOP = "stop"


@dataclass(frozen=True)
class LipActivityConfig:
    """Centralized, configurable parameters for lip activity endpointing."""
    open_threshold: float = 0.04       # Opening >= threshold marks lip activity; below is quiet/still.
    quiet_ms: int = 700               # Duration of quiet/stillness to trigger chunk stop (600-800ms).
    min_duration_ms: int = 1200       # Minimum duration of a chunk before quiet can trigger stop.
    max_chunk_ms: int = 6000          # Maximum duration cap for continuous activity.
    pre_roll_ms: int = 300            # Target pre-roll buffer duration before trigger timestamp.
    min_active_frames: int = 1        # Consecutive active frames to trigger start.
    face_loss_quiet: bool = False     # If True, face loss counts as stillness; if False, face loss waits.


class LipActivityDetector:
    """Finite state machine driving lip-activity endpoints.

    Evaluates:
    - Face visibility: opening is not None.
    - Mouth opening ratio vs open_threshold.
    - Activity onset and quiet stillness duration.

    Guarantees:
    - Missing face or quiet/still mouth does not start recording while idle.
    - Lip activity triggers START.
    - Quiet stillness (>= quiet_ms) after min_duration_ms triggers STOP.
    - Continuous activity reaching max_chunk_ms triggers STOP.
    - Reset/cancel restores IDLE state cleanly without leaking old timestamps.
    """

    def __init__(self, config: Optional[LipActivityConfig] = None):
        self.config = config or LipActivityConfig()
        self.state: ActivityState = ActivityState.IDLE
        self.start_ts: Optional[int] = None
        self.quiet_since: Optional[int] = None
        self.active_frames: int = 0
        self.last_ts: Optional[int] = None

    def reset(self):
        """Reset state machine to clean IDLE state."""
        self.state = ActivityState.IDLE
        self.start_ts = None
        self.quiet_since = None
        self.active_frames = 0
        self.last_ts = None

    def cancel(self):
        """Alias for reset."""
        self.reset()

    def sync_recording(self, start_ts: int):
        """Synchronize state when recording is started externally or by runtime."""
        self.state = ActivityState.RECORDING
        self.start_ts = start_ts
        self.quiet_since = None
        self.active_frames = 0

    def update(self, opening: Optional[float], timestamp_ms: int) -> ActivityAction:
        """Process one frame's mouth opening and timestamp.

        Args:
            opening: Normalized inner/outer lip ratio, or None if face/mouth is undetected.
            timestamp_ms: Timestamp of the frame in milliseconds.

        Returns:
            ActivityAction.START if activity begins,
            ActivityAction.STOP if quiet endpoint or max chunk is reached,
            ActivityAction.NONE otherwise.
        """
        if timestamp_ms is None:
            return ActivityAction.NONE
        self.last_ts = timestamp_ms

        if self.state == ActivityState.IDLE:
            is_active = opening is not None and opening >= self.config.open_threshold
            if is_active:
                self.active_frames += 1
                if self.active_frames >= self.config.min_active_frames:
                    self.state = ActivityState.RECORDING
                    self.start_ts = timestamp_ms
                    self.quiet_since = None
                    self.active_frames = 0
                    return ActivityAction.START
            else:
                self.active_frames = 0
            return ActivityAction.NONE

        if self.state == ActivityState.RECORDING:
            elapsed = timestamp_ms - (self.start_ts if self.start_ts is not None else timestamp_ms)
            if elapsed >= self.config.max_chunk_ms:
                self.reset()
                return ActivityAction.STOP

            if opening is not None and opening >= self.config.open_threshold:
                # Mouth is actively moving; reset any quiet timer.
                self.quiet_since = None
                return ActivityAction.NONE

            if opening is not None and opening < self.config.open_threshold:
                # Face visible and mouth is quiet/still.
                if self.quiet_since is None:
                    self.quiet_since = timestamp_ms
                quiet_elapsed = timestamp_ms - self.quiet_since
                if quiet_elapsed >= self.config.quiet_ms and elapsed >= self.config.min_duration_ms:
                    self.reset()
                    return ActivityAction.STOP
                return ActivityAction.NONE

            # Face is lost (opening is None).
            if self.config.face_loss_quiet:
                if self.quiet_since is None:
                    self.quiet_since = timestamp_ms
                quiet_elapsed = timestamp_ms - self.quiet_since
                if quiet_elapsed >= self.config.quiet_ms and elapsed >= self.config.min_duration_ms:
                    self.reset()
                    return ActivityAction.STOP
            return ActivityAction.NONE

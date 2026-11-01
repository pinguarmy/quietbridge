"""Temporal stabilization for supported sign-recognition labels."""

from hackathon.contracts import SignEvent


class EventStabilizer:
    """Emit one final event after a supported label remains stable."""

    _SUPPORTED_LABELS = {"THUMB_UP", "THUMB_DOWN", "OPEN_PALM", "POINTING_UP"}

    def __init__(
        self,
        min_observations: int = 4,
        min_duration_ms: int = 250,
        max_gap_ms: int = 250,
        min_score: float = 0.75,
        rearm_ms: int = 400,
    ) -> None:
        self.min_observations = min_observations
        self.min_duration_ms = min_duration_ms
        self.max_gap_ms = max_gap_ms
        self.min_score = min_score
        self.rearm_ms = rearm_ms

        self._last_timestamp_ms: int | None = None
        self._candidate_label: str | None = None
        self._candidate_start_ms: int | None = None
        self._candidate_end_ms: int | None = None
        self._candidate_count = 0
        self._candidate_min_score = 1.0
        self._locked = False
        self._neutral_start_ms: int | None = None
        self._neutral_end_ms: int | None = None

    def update(self, label: str | None, score: float, timestamp_ms: int) -> list[SignEvent]:
        """Consume one sampled prediction and return any newly stabilized event."""
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must be strictly increasing")

        gap = (
            self._last_timestamp_ms is not None
            and timestamp_ms - self._last_timestamp_ms > self.max_gap_ms
        )
        self._last_timestamp_ms = timestamp_ms

        if gap:
            self._reset_candidate()
            self._neutral_start_ms = None
            self._neutral_end_ms = None

        supported = label if label in self._SUPPORTED_LABELS and score >= self.min_score else None
        if self._locked:
            self._update_rearm(supported, timestamp_ms)
            return []

        if supported is None:
            self._reset_candidate()
            return []

        if supported != self._candidate_label:
            self._candidate_label = supported
            self._candidate_start_ms = timestamp_ms
            self._candidate_end_ms = timestamp_ms
            self._candidate_count = 1
            self._candidate_min_score = score
            return []

        self._candidate_end_ms = timestamp_ms
        self._candidate_count += 1
        self._candidate_min_score = min(self._candidate_min_score, score)
        assert self._candidate_start_ms is not None
        assert self._candidate_end_ms is not None
        if (
            self._candidate_count >= self.min_observations
            and self._candidate_end_ms - self._candidate_start_ms >= self.min_duration_ms
        ):
            event = SignEvent(
                label=self._candidate_label,
                score=self._candidate_min_score,
                start_ms=self._candidate_start_ms,
                end_ms=self._candidate_end_ms,
                is_final=True,
            )
            self._locked = True
            self._neutral_start_ms = None
            self._neutral_end_ms = None
            self._reset_candidate()
            return [event]
        return []

    def _update_rearm(self, supported: str | None, timestamp_ms: int) -> None:
        if supported is not None:
            self._neutral_start_ms = None
            self._neutral_end_ms = None
            return
        if self._neutral_start_ms is None:
            self._neutral_start_ms = timestamp_ms
        self._neutral_end_ms = timestamp_ms
        if self._neutral_end_ms - self._neutral_start_ms >= self.rearm_ms:
            self._locked = False
            self._neutral_start_ms = None
            self._neutral_end_ms = None

    def _reset_candidate(self) -> None:
        self._candidate_label = None
        self._candidate_start_ms = None
        self._candidate_end_ms = None
        self._candidate_count = 0
        self._candidate_min_score = 1.0

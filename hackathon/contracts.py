"""Shared interface between Sam's runtime and Peter's sign recognizer.

All event timestamps use int(time.perf_counter() * 1000), on the same machine.
Only final, supported sign events enter fusion. Scores are recognizer scores,
not a claim of calibrated probability or measured accuracy.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignEvent:
    label: str
    score: float
    start_ms: int
    end_ms: int
    is_final: bool


@dataclass(frozen=True)
class LipResult:
    candidates: list[str]
    decode_ms: int
    # ESPnet scores are decoder scores, not calibrated probabilities. They are
    # useful only for comparing candidates produced by the same model/config.
    candidate_scores: list[float | None] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None
    raw_frames: int = 0
    capture_fps: float | None = None
    max_frame_gap_ms: int | None = None


@dataclass(frozen=True)
class Decision:
    text: str
    status: str  # accepted | confirmation_needed | unavailable
    reason: str  # lip_only | sign_only | agreement | partial_agreement | conflict
    source: str  # local | suggested_by_api
    matched_signs: list[str] = field(default_factory=list)

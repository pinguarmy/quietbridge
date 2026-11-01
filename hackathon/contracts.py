"""Shared interface between Sam's runtime and Peter's sign recognizer.

All event timestamps use int(time.perf_counter() * 1000), on the same machine.
Only final, supported sign events enter fusion. Scores are recognizer scores,
not a claim of calibrated probability or measured accuracy.
"""
from dataclasses import dataclass


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


@dataclass(frozen=True)
class Decision:
    text: str
    status: str  # accepted | confirmation_needed | unavailable
    reason: str  # lip_only | sign_only | agreement | conflict
    source: str  # local | suggested_by_api

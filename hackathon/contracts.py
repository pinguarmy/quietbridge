"""Shared QuietBridge interfaces. Both tracks import these exact types."""

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

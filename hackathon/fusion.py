"""Conservative local fusion. No sentence generation or network access."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .contracts import Decision, LipResult, SignEvent

VOCABULARY = Path(__file__).with_name("vocabulary.json")
NEGATION = {"no", "not", "never", "don't", "dont", "cannot", "can't", "cant", "won't", "wont", "without"}
RISK = NEGATION | {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                   "hundred", "thousand", "million", "medicine", "medication", "dose", "emergency", "pain"}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'")))


def load_vocabulary(path=VOCABULARY) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Vocabulary must map labels to phrase/matches/verified entries")
    for label, entry in data.items():
        if (not isinstance(label, str) or not isinstance(entry, dict)
                or not isinstance(entry.get("phrase"), str) or not entry["phrase"].strip()
                or not isinstance(entry.get("matches"), list) or not entry["matches"]
                or not all(isinstance(m, str) and tokens(m) for m in entry["matches"])
                or not isinstance(entry.get("verified"), bool)
                or ("enabled" in entry and not isinstance(entry["enabled"], bool))):
            raise ValueError(f"Invalid vocabulary entry: {label}")
    return data


def usable_signs(events: list[SignEvent], vocabulary: dict, now_ms: int,
                 since_ms: int | None = None) -> list[SignEvent]:
    lower = max(now_ms - 4000, since_ms if since_ms is not None else now_ms - 4000)
    return [e for e in events if e.is_final and e.label in vocabulary
            and vocabulary[e.label].get("enabled", vocabulary[e.label].get("verified")) is True
            and math.isfinite(e.score) and 0.75 <= e.score <= 1
            and 0 <= e.start_ms <= e.end_ms <= now_ms and e.end_ms >= lower]


def fuse(lips: LipResult | None, events: list[SignEvent], vocabulary: dict,
         now_ms: int, since_ms: int | None = None) -> Decision:
    candidates = list(dict.fromkeys(c.strip() for c in (lips.candidates if lips else []) if c.strip()))
    signs = usable_signs(events, vocabulary, now_ms, since_ms)
    labels = list(dict.fromkeys(e.label for e in signs))
    if not candidates and not labels:
        return Decision("", "unavailable", "lip_only" if lips is not None else "sign_only", "local")
    if len(labels) > 1:
        return Decision(candidates[0] if candidates else "", "confirmation_needed", "conflict", "local")
    if not labels:
        # LipResult has no calibrated confidence: a tap always confirms intent.
        return Decision(candidates[0], "confirmation_needed", "lip_only", "local")
    entry = vocabulary[labels[0]]
    if not candidates:
        status = "accepted" if entry.get("verified") is True else "confirmation_needed"
        return Decision(entry["phrase"], status, "sign_only", "local")
    for candidate in candidates:
        words = tokens(candidate)
        matches = any(tokens(m) <= words for m in entry["matches"])
        # A concept match cannot establish the rest of a sentence, its negation or quantities.
        if matches and not (words & RISK) and not any(w.isdigit() for w in words):
            return Decision(candidate, "confirmation_needed", "agreement", "local")
        if matches and words == tokens(entry["phrase"]):
            return Decision(candidate, "confirmation_needed", "agreement", "local")
    return Decision(candidates[0], "confirmation_needed", "conflict", "local")

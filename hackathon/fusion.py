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
    # An explicit recording may last 30 seconds: all events inside that window
    # remain eligible. Only ambient, unwindowed events use the freshness limit.
    lower = since_ms if since_ms is not None else now_ms - 4000
    # Recognizers emit only events accepted by their own model-specific rules.
    # A MediaPipe classifier score and a MotionModel matching score do not share
    # a meaningful universal 0.75 probability threshold.
    return [e for e in events if e.is_final and e.label in vocabulary
            and vocabulary[e.label].get("enabled", vocabulary[e.label].get("verified")) is True
            and math.isfinite(e.score) and 0 <= e.score <= 1
            and 0 <= e.start_ms <= e.end_ms <= now_ms and e.end_ms >= lower]


def fuse(lips: LipResult | None, events: list[SignEvent], vocabulary: dict,
         now_ms: int, since_ms: int | None = None) -> Decision:
    # Beam search already orders candidates by model score. Preserve that order;
    # gesture evidence can improve it, but scores from different models are not
    # treated as comparable probabilities.
    candidates = list(dict.fromkeys(c.strip() for c in (lips.candidates if lips else []) if c.strip()))
    signs = sorted(usable_signs(events, vocabulary, now_ms, since_ms),
                   key=lambda event: (event.start_ms, event.end_ms))
    labels = list(dict.fromkeys(event.label for event in signs))
    if not candidates and not labels:
        return Decision("", "unavailable", "lip_only" if lips is not None else "sign_only", "local")
    if not labels:
        # Lip scores are not calibrated confidence: a tap always confirms intent.
        return Decision(candidates[0], "confirmation_needed", "lip_only", "local")
    if not candidates:
        # A sequence of explicit, supported gestures is composable. Unknown or
        # rejected motions never reach this function and cannot contribute text.
        text = " ".join(vocabulary[event.label]["phrase"].strip() for event in signs)
        verified = all(vocabulary[event.label].get("verified") is True for event in signs)
        return Decision(text, "accepted" if verified else "confirmation_needed",
                        "sign_only", "local", labels)

    def matched_labels(candidate):
        words = tokens(candidate)
        matched = []
        for label in labels:
            entry = vocabulary[label]
            alias_match = any(tokens(alias) <= words for alias in entry["matches"])
            safe = not (words & RISK) and not any(word.isdigit() for word in words)
            if alias_match and (safe or words == tokens(entry["phrase"])):
                matched.append(label)
        return matched

    ranked = [(candidate, matched_labels(candidate)) for candidate in candidates]
    best, matched = max(ranked, key=lambda item: len(item[1]))
    if matched:
        reason = "agreement" if len(matched) == len(labels) else "partial_agreement"
        return Decision(best, "confirmation_needed", reason, "local", matched)
    return Decision(candidates[0], "confirmation_needed", "conflict", "local")

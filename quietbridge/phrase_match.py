"""Offer the person's own saved sentences beside a lip n-best list.

Silent mouthing is ambiguous, but a person who uses a communication aid says
a small set of sentences often, and their profile already holds them: the saved
phrases and the corrected texts they explicitly chose to learn from. When the
lip reader's candidates are close to one of those sentences, this module
proposes that sentence as an *extra, selectable* alternative. It never edits,
reorders or replaces a lip candidate, never chooses for the person, and never
proposes anything without at least one non-empty lip candidate.

Similarity is an order-respecting word edit distance. A substitution between
two words that the coarse spelling-based viseme map of ``lipflow.visemes``
writes identically (p/b/m, f/v, t/d/n/s/z/l, k/g, ...) costs
``LOOKALIKE_COST`` instead of 1, because those words look alike on the lips.
Negation, number and safety words (``fusion.is_sensitive``) never get that
discount. A character-level distance only breaks ties.

All thresholds below are engineering defaults chosen on constructed examples.
They are NOT calibrated and have not been measured on any participant's
recordings; ``rescue_outcome``/``summarize_rescue`` exist so they can be
measured on paired data later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import os
import re
import unicodedata

from .fusion import _APOSTROPHES, is_sensitive

# --- Engineering defaults (uncalibrated; keep every threshold here) ----------
# Phrases with fewer words than this match only word-for-word through
# look-alike substitutions: "Yes" or "Thank you" would otherwise match far too
# many short lip readings.
MIN_FUZZY_WORDS = 3
# Longer phrases: normalised word edit distance (cost / longer length) must be
# at most this. 0.25 allows one full word error in four, or one look-alike
# substitution in a three-word phrase.
MAX_DISTANCE_RATIO = 0.25
# ...and at least this fraction of the phrase's words must align exactly.
MIN_EXACT_WORD_FRACTION = 0.5
# Substitution cost for two non-sensitive words with the same viseme string.
LOOKALIKE_COST = 0.5
# At most this many phrase suggestions are added to one result.
MAX_SUGGESTIONS = 2
# Bounded work under the runtime lock: phrases, candidates and words considered.
MAX_PHRASES = 100
MAX_CANDIDATES = 12
MAX_WORDS = 40
# Set to 0/false/off/no to disable suggestions for this process.
ENV_SWITCH = "QUIETBRIDGE_PHRASE_SUGGESTIONS"

# --- Model-scored offers (the lip model decides; text distance only prefilters) ----
# Measured on two public clips of one professional speaker talking aloud
# (docs/experiments/real-clip-experiments.md, Experiment 2b and the follow-up):
# scoring a saved sentence under the lip model separates the intended sentence
# from distinct sentences and from one-word variants far better than the text
# distance below. Not measured on silent mouthing, short phrases or real users.
#
# A profile with at most this many saved phrases has every phrase scored: for
# so few it is cheaper than prefiltering and cannot miss a phrase whose text the
# lip candidates got badly wrong.
SCORE_ALL_PHRASES = 20
# Larger profiles: at most this many phrases, the nearest by look-alike word
# edit distance to any lip candidate, are scored. Bounds the added latency
# (follow-up latency table: ~0.2-0.4 s for 20 phrases on 4 CPU threads).
SCORE_MAX_PHRASES = 20
# Prefilter ceiling: a phrase whose normalised distance to every candidate is
# above this is never scored. Deliberately much looser than MAX_DISTANCE_RATIO
# (at 12 fps no top-1 was exact, yet model scoring still found the sentence).
PREFILTER_MAX_RATIO = 0.75
# Offer a scored phrase only if its length-normalised model score (nats per
# token) is at least the score of the decoder's own top text, computed the same
# way, minus this margin. Chosen on session A only (follow-up addendum in the
# experiments document) by the product's costs: a lip-only result is always
# confirmed anyway, so an extra row beside a correct result is clutter, and an
# extra tap only if it creates a conflict; a missed rescue means retyping.
# Rule: largest rescue rate with offers for unsaved sentences <= 5 % and
# conflict offers beside a correct top-1 <= 10 % (one-word-variant scenarios).
# The earlier 0.2 (no offer at all beside a correct result) rescued far less.
# Measured on 9-35-word sentences, where one wrong word moves the score by
# ~0.3-0.6 nats/token; short phrases spread a word's cost over fewer tokens,
# so this margin is stricter for them (unmeasured).
MODEL_OFFER_MARGIN = 0.5

_DIGITS = {str(i): w for i, w in enumerate(
    ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
     "fifteen sixteen seventeen eighteen nineteen twenty").split())}
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def enabled_by_environment(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get(ENV_SWITCH, "").strip().lower()
    return value not in {"0", "false", "off", "no"}


def words(text: str) -> list[tuple[str, str]]:
    """Ordered ``(key, original)`` word pairs.

    Same NFKC/apostrophe folding as ``fusion.tokens``, but ordered. The key
    drops apostrophes ("DON'T" -> "dont") and spells 0-20 as words, because the
    lip reader writes words while a saved phrase may use digits.
    """
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    out = []
    for token in _WORD.findall(text):
        key = token.replace("'", "")
        out.append((_DIGITS.get(key, key), token))
    return out


def normalized(text: str) -> str:
    """Comparison form: lower-case words without punctuation."""
    return " ".join(key for key, _ in words(text))


def sensitive_words(text: str) -> frozenset[str]:
    return frozenset(key for key, original in words(text)
                     if is_sensitive(key) or is_sensitive(original))


@lru_cache(maxsize=4096)
def _viseme_string(word: str) -> str:
    try:
        from lipflow.visemes import visemes
    except Exception:  # pragma: no cover - lipflow is part of this repository
        return ""
    return visemes(word)


def lookalike(a: str, b: str) -> bool:
    """Two different, non-sensitive words with the same coarse viseme string.

    Unmeasured heuristic: the viseme map is derived from spelling, not
    pronunciation, so it is approximate in both directions.
    """
    if a == b or is_sensitive(a) or is_sensitive(b):
        return False
    va = _viseme_string(a)
    return bool(va) and va == _viseme_string(b)


def _pair(a: str, b: str) -> tuple[float, bool]:
    """(substitution cost, is a look-alike substitution). The flag is tracked
    separately from the cost so that a LOOKALIKE_COST of 1.0 (no discount)
    does not turn every ordinary substitution into a "look-alike"."""
    if a == b:
        return 0.0, False
    if lookalike(a, b):
        return LOOKALIKE_COST, True
    return 1.0, False


def word_cost(a: str, b: str) -> float:
    return _pair(a, b)[0]


def _align(phrase: list[str], candidate: list[str]):
    """Word edit distance with look-alike substitutions; returns
    (cost, exact_aligned, lookalike_aligned)."""
    n, m = len(phrase), len(candidate)
    pairs = [[_pair(p, c) for c in candidate] for p in phrase]
    dp = [[float(i + j) if i == 0 or j == 0 else 0.0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + pairs[i - 1][j - 1][0])
    i, j, exact, look = n, m, 0, 0
    while i > 0 and j > 0:
        cost, is_lookalike = pairs[i - 1][j - 1]
        if dp[i][j] == dp[i - 1][j - 1] + cost:  # costs are multiples of 0.5 by default: exact in binary
            exact += cost == 0
            look += is_lookalike
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return dp[n][m], exact, look


def _char_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1] / max(len(a), len(b))


@dataclass(frozen=True)
class PhraseMatch:
    """Evidence for one phrase. ``phrase`` is the person's own text."""
    phrase: str
    phrase_index: int        # position in the phrase list given
    candidate_index: int     # best-matching candidate (0 = lip top candidate)
    distance: float          # word edit cost / longer word count
    char_distance: float     # tie-break only
    phrase_words: int
    exact_words: int
    lookalike_words: int
    already_listed: bool     # normalised phrase equals a candidate or existing alternative
    sensitive_difference: bool = False  # negation/number/safety words differ from ``top``

    def evidence(self) -> dict:
        """Text-free diagnostics for state metrics and evaluation reports."""
        data = asdict(self)
        data.pop("phrase")
        data["distance"] = round(self.distance, 4)
        data["char_distance"] = round(self.char_distance, 4)
        return data


def _accept(cost, exact, look, n, m) -> bool:
    if n < MIN_FUZZY_WORDS:
        # Short phrases: word-for-word, every difference a look-alike word.
        return m == n and exact + look == n
    return (cost / max(n, m) <= MAX_DISTANCE_RATIO
            and exact >= math.ceil(n * MIN_EXACT_WORD_FRACTION))


def _match_one(phrase_keys, candidate_keys):
    n, m = len(phrase_keys), len(candidate_keys)
    if not n or not m or n > MAX_WORDS or m > MAX_WORDS:
        return None
    if n < MIN_FUZZY_WORDS:
        if m != n:
            return None
    else:
        if abs(n - m) / max(n, m) > MAX_DISTANCE_RATIO:
            return None
        # Cheap necessary condition before the quadratic alignment.
        shared = sum(min(phrase_keys.count(w), candidate_keys.count(w)) for w in set(phrase_keys))
        if shared < math.ceil(n * MIN_EXACT_WORD_FRACTION):
            return None
    cost, exact, look = _align(phrase_keys, candidate_keys)
    return (cost, exact, look) if _accept(cost, exact, look, n, m) else None


def match_phrases(candidates, phrases, *, top=None, existing=()) -> list[PhraseMatch]:
    """Every phrase that passes the thresholds against any candidate, ranked.

    ``candidates`` is the ordered n-best (only the first ``MAX_CANDIDATES``
    non-empty ones are compared; their indices are preserved). ``top`` is the
    text currently displayed for the person (defaults to the first candidate);
    ``sensitive_difference`` compares against it. ``existing`` lists texts
    already offered (they count as ``already_listed``). Nothing is returned
    when there is no non-empty candidate.
    """
    indexed = [(i, c) for i, c in enumerate(candidates or []) if isinstance(c, str) and c.strip()]
    indexed = indexed[:MAX_CANDIDATES]
    if not indexed:
        return []
    keyed = [(i, [k for k, _ in words(c)], c) for i, c in indexed]
    keyed = [(i, keys, c) for i, keys, c in keyed if keys]
    if not keyed:
        return []
    top_text = top if isinstance(top, str) and top.strip() else keyed[0][2]
    top_sensitive = sensitive_words(top_text)
    listed = {normalized(text) for text in [*(c for _, _, c in keyed), *existing] if isinstance(text, str)}
    matches, seen = [], set()
    for p_index, phrase in enumerate(list(phrases or [])[:MAX_PHRASES]):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        phrase = phrase.strip()
        keys = [k for k, _ in words(phrase)]
        norm = " ".join(keys)
        if not keys or norm in seen:
            continue
        best = None
        for c_index, c_keys, c_text in keyed:
            found = _match_one(keys, c_keys)
            if found is None:
                continue
            cost, exact, look = found
            rank = (cost / max(len(keys), len(c_keys)), _char_distance(norm, " ".join(c_keys)), c_index)
            if best is None or rank < best[0]:
                best = (rank, exact, look)
        if best is None:
            continue
        seen.add(norm)
        (distance, char_distance, c_index), exact, look = best
        matches.append(PhraseMatch(
            phrase=phrase, phrase_index=p_index, candidate_index=c_index, distance=distance,
            char_distance=char_distance, phrase_words=len(keys), exact_words=exact,
            lookalike_words=look, already_listed=norm in listed,
            sensitive_difference=sensitive_words(phrase) != top_sensitive))
    matches.sort(key=lambda m: (m.distance, m.char_distance, m.candidate_index, m.phrase_index))
    return matches


def suggest(candidates, phrases, *, top=None, existing=(), limit=MAX_SUGGESTIONS) -> list[PhraseMatch]:
    """The phrases to offer: not already listed, best first, at most ``limit``."""
    return [m for m in match_phrases(candidates, phrases, top=top, existing=existing)
            if not m.already_listed][:max(0, min(limit, MAX_SUGGESTIONS))]


# --- Offline evaluation hook (no accuracy claim without paired data) ---------

def rescue_outcome(candidates, reference, phrases, *, top=None, offered=None) -> dict:
    """Would a phrase suggestion have rescued this recording?

    Compares the lip top-1 text with "top-1 or any offered phrase suggestion"
    against the intended ``reference``, using ``normalized`` (lower-case words,
    no punctuation, 0-20 spelt out) for equality. The reference is only used
    for scoring, never for choosing suggestions. ``offered`` (``PhraseMatch``
    or ``ModelOffer`` objects) evaluates offers decided elsewhere, e.g. by
    ``model_offers``; by default the text matcher (``suggest``) decides.
    """
    texts = [c for c in (candidates or []) if isinstance(c, str) and c.strip()]
    ref = normalized(reference or "")
    offered = suggest(candidates, phrases, top=top) if offered is None else list(offered)
    top1_exact = bool(texts) and normalized(texts[0]) == ref
    suggestion_exact = any(normalized(m.phrase) == ref for m in offered)
    return {
        "top1_exact": top1_exact,
        "nbest_exact": any(normalized(t) == ref for t in texts),
        "reference_in_phrases": any(isinstance(p, str) and normalized(p) == ref for p in phrases or []),
        "offered": len(offered),
        "offered_evidence": [m.evidence() for m in offered],
        "suggestion_exact": suggestion_exact,
        "top1_or_suggestion_exact": top1_exact or suggestion_exact,
        "rescued": not top1_exact and suggestion_exact,
        "wrong_offer": bool(offered) and not suggestion_exact,
        "sensitive_offer": any(m.sensitive_difference for m in offered),
    }


def summarize_rescue(outcomes) -> dict:
    """Aggregate ``rescue_outcome`` rows; rates are None when undefined.

    Rows whose reference is one of the phrases bound how often a suggestion
    *could* help; rows whose reference is not a phrase measure how often a
    wrong phrase is offered. Report both: profile practice prompts are often
    the recording prompts, so the in-phrase share says how representative a
    split is of real, open use.
    """
    rows = list(outcomes)

    def rate(num, den):
        return num / den if den else None

    def group(subset):
        count = len(subset)
        top1 = sum(r["top1_exact"] for r in subset)
        either = sum(r["top1_or_suggestion_exact"] for r in subset)
        offered = sum(r["offered"] > 0 for r in subset)
        return {"count": count, "top1_exact": top1, "top1_exact_rate": rate(top1, count),
                "top1_or_suggestion_exact": either, "top1_or_suggestion_exact_rate": rate(either, count),
                "rescued": sum(r["rescued"] for r in subset),
                "rows_with_offer": offered,
                "wrong_offers": sum(r["wrong_offer"] for r in subset),
                "wrong_offer_rate": rate(sum(r["wrong_offer"] for r in subset), offered),
                "sensitive_offers": sum(r["sensitive_offer"] for r in subset)}

    return {**group(rows),
            "reference_in_phrases": group([r for r in rows if r["reference_in_phrases"]]),
            "reference_not_in_phrases": group([r for r in rows if not r["reference_in_phrases"]]),
            "limitations": [
                "Thresholds are uncalibrated engineering defaults; do not tune them on the final-test session.",
                "Exact match uses lower-case words without punctuation; it is not the WER normalisation.",
                "A suggestion only helps if the person notices and chooses it; this counts availability, not use."]}


# --- Model-scored offers -----------------------------------------------------

def _prefilter_distance(phrase_keys, candidate_keys) -> float | None:
    n, m = len(phrase_keys), len(candidate_keys)
    if not n or not m or n > MAX_WORDS or m > MAX_WORDS:
        return None
    if abs(n - m) / max(n, m) > PREFILTER_MAX_RATIO:  # the cost is at least |n - m|
        return None
    cost, _, _ = _align(phrase_keys, candidate_keys)
    ratio = cost / max(n, m)
    return ratio if ratio <= PREFILTER_MAX_RATIO else None


def phrases_to_score(candidates, phrases, *, existing=()) -> list[int]:
    """Indices of the saved phrases the lip model should score for this result.

    Every distinct, non-empty phrase (first ``MAX_PHRASES``) when there are at
    most ``SCORE_ALL_PHRASES``; otherwise the ``SCORE_MAX_PHRASES`` nearest by
    look-alike word edit distance to any of the first ``MAX_CANDIDATES``
    candidates, within ``PREFILTER_MAX_RATIO``. Phrases already shown as a
    candidate or alternative are skipped (they would never be offered). Nothing
    without a non-empty candidate.
    """
    keyed_candidates = [[k for k, _ in words(c)] for c in (candidates or [])[:MAX_CANDIDATES]
                        if isinstance(c, str) and c.strip()]
    keyed_candidates = [keys for keys in keyed_candidates if keys]
    if not keyed_candidates:
        return []
    listed = {" ".join(keys) for keys in keyed_candidates}
    listed |= {normalized(text) for text in existing if isinstance(text, str)}
    distinct, seen = [], set()
    for index, phrase in enumerate(list(phrases or [])[:MAX_PHRASES]):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        keys = [k for k, _ in words(phrase)]
        norm = " ".join(keys)
        if not keys or norm in seen or norm in listed:
            continue
        seen.add(norm)
        distinct.append((index, keys))
    if len(distinct) <= SCORE_ALL_PHRASES:
        return [index for index, _ in distinct]
    ranked = []
    for index, keys in distinct:
        found = [d for d in (_prefilter_distance(keys, c) for c in keyed_candidates) if d is not None]
        if found:
            ranked.append((min(found), index))
    ranked.sort()
    return [index for _, index in ranked[:SCORE_MAX_PHRASES]]


@dataclass(frozen=True)
class ModelOffer:
    """A saved phrase the lip model scored as competitive with its own top text.

    ``score_gap`` = phrase score minus top-text score, both length-normalised
    log-probabilities under the same lip model (nats per token; > 0: the model
    prefers the phrase). A decoder score, never a probability.
    """
    phrase: str
    phrase_index: int
    score_gap: float
    phrase_words: int
    already_listed: bool
    sensitive_difference: bool = False

    def evidence(self) -> dict:
        return {"phrase_index": self.phrase_index, "score_gap": round(self.score_gap, 4),
                "phrase_words": self.phrase_words, "already_listed": self.already_listed,
                "sensitive_difference": self.sensitive_difference, "path": "model"}


def model_offers(candidates, phrases, phrase_scores, top_score, *, top=None, existing=(),
                 margin=None, limit=MAX_SUGGESTIONS) -> list[ModelOffer]:
    """Phrases to offer, decided by the lip model's own scores.

    ``phrase_scores`` maps a phrase index (into ``phrases``) to its
    length-normalised model score; ``top_score`` is the decoder's top text
    scored the same way. A phrase is offered only when ``score >= top_score -
    margin`` (default ``MODEL_OFFER_MARGIN``), it is not already a candidate or
    alternative, and there is a non-empty candidate. Best score first, at most
    ``limit`` (never more than ``MAX_SUGGESTIONS``). ``sensitive_difference``
    compares negation/number/safety words with ``top`` (the displayed text,
    default: first candidate). Nothing is offered when ``top_score`` is missing.
    """
    texts = [c for c in (candidates or []) if isinstance(c, str) and c.strip()]
    if not texts or top_score is None or not math.isfinite(top_score):
        return []
    margin = MODEL_OFFER_MARGIN if margin is None else margin
    top_text = top if isinstance(top, str) and top.strip() else texts[0]
    top_sensitive = sensitive_words(top_text)
    listed = {normalized(t) for t in [*texts, *existing] if isinstance(t, str)}
    phrases = list(phrases or [])
    offers, seen = [], set()
    for index, score in sorted(((i, s) for i, s in (phrase_scores or {}).items()
                                if s is not None and math.isfinite(s)), key=lambda item: (-item[1], item[0])):
        if not (0 <= index < len(phrases)) or not isinstance(phrases[index], str):
            continue
        phrase = phrases[index].strip()
        norm = normalized(phrase)
        if not norm or norm in seen or score < top_score - margin:
            continue
        seen.add(norm)
        offers.append(ModelOffer(phrase=phrase, phrase_index=index, score_gap=score - top_score,
                                 phrase_words=len(norm.split()), already_listed=norm in listed,
                                 sensitive_difference=sensitive_words(phrase) != top_sensitive))
    return [o for o in offers if not o.already_listed][:max(0, min(limit, MAX_SUGGESTIONS))]

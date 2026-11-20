""""What to record next": plain-language guidance from data counts and validation reports.

``guidance()`` is pure: it takes a data inventory (``inventory()``), the
profile's gestures and phrases, and the newest candidate report of each model
type, and returns an ordered list of items. Each item carries the numbers it
is based on, including the denominator, so nothing is a bare percentage.

Honesty rules this module keeps:
- Only validation ("Check on another occasion") results and data counts are
  read. Final-check (test) results are never read, so guidance cannot steer
  anyone toward tuning on the final check.
- A count of recorded tries is never worded as a result. Results always say
  "recognised in X of Y check tries" and name the candidate they come from.
- All minimums are recommendations (engineering defaults), not guarantees.
  They live in ``THRESHOLDS`` and are returned with the guidance.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .capture_quality import from_npz, gesture_quality, lip_quality, needs_review
from .motion_variants import MIN_VALIDATION_REST_TRIES, MIN_VALIDATION_TRIES_PER_GESTURE
from .train_lips import MIN_FIT_CLIPS, MIN_VALIDATION_CLIPS

REST_LABEL = "REST"
SPLITS = ("train", "validation", "test")

#: Recommendations, not guarantees. Gesture check minimums match the point at
#: which the gesture trainer compares its variants (motion_variants); lip
#: minimums match train_lips' refusal limits.
THRESHOLDS: dict[str, float] = {
    "practice_tries_per_gesture": 5,
    "practice_occasions_per_gesture": 2,
    "practice_rest_tries": 5,
    "check_tries_per_gesture": MIN_VALIDATION_TRIES_PER_GESTURE,
    "check_rest_tries": MIN_VALIDATION_REST_TRIES,
    "final_check_tries_per_gesture": 3,
    # A gesture's check result is only judged with at least this many tries.
    "min_tries_to_judge": 3,
    # Below this share of check tries found, a gesture is called weak.
    "weak_recall": 0.8,
    # Confused at least this many times (or both ways) suggests the movements look alike.
    "lookalike_confusions": 2,
    "practice_tries_per_phrase": 3,
    "check_tries_per_phrase": 2,
    "lip_practice_clips": MIN_FIT_CLIPS,
    "lip_check_clips": MIN_VALIDATION_CLIPS,
    # A phrase with at least this word error rate on check tries is listed.
    "high_phrase_wer": 0.5,
    "max_phrases_listed": 3,
}

_SEVERITY_ORDER = {"action": 0, "warn": 1, "info": 2}
_SPLIT_WORD = {"train": "practice", "validation": "check", "test": "final-check"}


def _wer(hyp: str, ref: str) -> tuple[int, int]:
    """Word edit distance and reference length (same rule as lipflow.bench.wer)."""
    h, r = re.findall(r"[a-z0-9']+", (hyp or "").lower()), re.findall(r"[a-z0-9']+", (ref or "").lower())
    d = list(range(len(r) + 1))
    for i, a in enumerate(h, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(r, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1], len(r)


def _tries(n: int, kind: str) -> str:
    return f"{n} {kind} {'try' if n == 1 else 'tries'}"


def _more_tries(n: int, kind: str) -> str:
    return f"{n} more {kind} {'try' if n == 1 else 'tries'}"


def _times(n: int) -> str:
    return "1 time" if n == 1 else f"{n} times"


# ---------------------------------------------------------------------------
# Inventory (reads metadata only)
# ---------------------------------------------------------------------------

def _scalar(data, key):
    return str(data[key].item()) if key in data.files else None


def inventory(sequences_dir, mouths_dir) -> dict[str, Any]:
    """Count recorded tries per split by gesture label and by phrase.

    Standard-sign video captures are counted separately and never as personal
    gestures. Unreadable files are counted, not guessed. Flagged = a try whose
    capture-quality review asked for a decision (stored at save time, or
    measured now for older recordings).
    """
    out: dict[str, Any] = {"gestures": {s: {} for s in SPLITS}, "phrases": {s: {} for s in SPLITS},
                           "sign_video_tries": 0, "unreadable_files": 0}
    for split in SPLITS:
        for path in sorted((Path(sequences_dir) / split).glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as d:
                    if "sign_language" in d.files or "bgr_frames" in d.files:
                        out["sign_video_tries"] += 1
                        continue
                    label, session = _scalar(d, "label"), _scalar(d, "session_id")
                    review = from_npz(d)
                    if review is None:
                        review = gesture_quality(d["landmarks"], d["hand_present"], d["timestamps_ms"], label)
            except Exception:  # noqa: BLE001 - a broken file is reported, not fatal
                out["unreadable_files"] += 1
                continue
            row = out["gestures"][split].setdefault(label, {"tries": 0, "sessions": set(), "flagged": 0})
            row["tries"] += 1
            row["sessions"].add(session)
            row["flagged"] += bool(needs_review(review))
        for path in sorted((Path(mouths_dir) / split).glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as d:
                    text, session = (_scalar(d, "text") or "").strip(), _scalar(d, "session_id")
                    review = from_npz(d)
                    if review is None and "timestamps_ms" in d.files and "face_present" in d.files:
                        review = lip_quality(d["timestamps_ms"], d["face_present"])
            except Exception:  # noqa: BLE001
                out["unreadable_files"] += 1
                continue
            row = out["phrases"][split].setdefault(text, {"tries": 0, "sessions": set(), "flagged": 0})
            row["tries"] += 1
            row["sessions"].add(session)
            row["flagged"] += bool(needs_review(review))
    for kind in ("gestures", "phrases"):
        for split in SPLITS:
            for row in out[kind][split].values():
                row["sessions"] = len(row["sessions"])
    return out


# ---------------------------------------------------------------------------
# Guidance (pure)
# ---------------------------------------------------------------------------

def _item(id_, severity, area, stage, subject, finding, next_step, numbers):
    return {"id": id_, "severity": severity, "area": area, "stage": stage, "subject": subject,
            "finding": finding, "next_step": next_step, "numbers": numbers}


def _count(inv, kind, split, key):
    return inv.get(kind, {}).get(split, {}).get(key, {"tries": 0, "sessions": 0, "flagged": 0})


def _gesture_items(inv, gestures, report, source, t) -> list[dict[str, Any]]:
    items = []
    if not report and not any(inv.get("gestures", {}).get(s) for s in SPLITS):
        return [_item("gestures-not-started", "info", "gestures", "train", "Gestures",
                      "No gesture tries are recorded yet. Typing and standard gestures work without them.",
                      f"To train personal gestures, record {int(t['practice_tries_per_gesture'])} practice tries "
                      f"of each gesture and {int(t['practice_rest_tries'])} resting moments.",
                      {"tries": 0})]
    names = {g["label"]: (g.get("display_name") or g["label"]) for g in gestures}
    labels = [g["label"] for g in gestures]
    if REST_LABEL not in labels:
        labels.append(REST_LABEL)
        names[REST_LABEL] = "Resting"
    for label in labels:
        name = names[label]
        rest = label == REST_LABEL
        what = "resting moments" if rest else name
        practice = _count(inv, "gestures", "train", label)
        check = _count(inv, "gestures", "validation", label)
        final = _count(inv, "gestures", "test", label)
        need_p = int(t["practice_rest_tries"] if rest else t["practice_tries_per_gesture"])
        need_c = int(t["check_rest_tries"] if rest else t["check_tries_per_gesture"])
        if practice["tries"] == 0:
            items.append(_item(f"gesture-no-practice-{label}", "action", "gestures", "train", name,
                               f"{name} has no practice tries yet.",
                               f"Record {need_p} practice tries of {what}.",
                               {"tries": 0, "recommended": need_p, "split": "train"}))
        elif practice["tries"] < need_p:
            items.append(_item(f"gesture-few-practice-{label}", "warn", "gestures", "train", name,
                               f"{name} has {_tries(practice['tries'], 'practice')} "
                               f"({need_p} or more recommended).",
                               f"Record {_more_tries(need_p - practice['tries'], 'practice')} of {what}.",
                               {"tries": practice["tries"], "recommended": need_p, "split": "train"}))
        elif not rest and practice["sessions"] < t["practice_occasions_per_gesture"]:
            items.append(_item(f"gesture-one-occasion-{label}", "info", "gestures", "train", name,
                               f"All {_tries(practice['tries'], 'practice')} of {name} come from one occasion.",
                               f"Practise {name} again on a new occasion: tries from different days show the "
                               "model how your movement naturally varies.",
                               {"tries": practice["tries"], "occasions": practice["sessions"],
                                "recommended_occasions": int(t["practice_occasions_per_gesture"]),
                                "split": "train"}))
        if check["tries"] < need_c:
            items.append(_item(f"gesture-few-check-{label}", "warn", "gestures", "validation", name,
                               (f"{name} has no check tries yet." if check["tries"] == 0 else
                                f"{name} has {_tries(check['tries'], 'check')}.")
                               + f" The trainer needs {need_c} per gesture, resting included, to compare its "
                               "options; with fewer it keeps the plain model.",
                               f"Record {_more_tries(need_c - check['tries'], 'check')} of {what} "
                               "on an occasion different from practice.",
                               {"tries": check["tries"], "recommended": need_c, "split": "validation"}))
        if final["tries"] == 0 and practice["tries"] and not rest:
            items.append(_item(f"gesture-no-final-{label}", "info", "gestures", "test", name,
                               f"{name} has no final-check tries, so its final result will show as not measured.",
                               f"Before activating, record {int(t['final_check_tries_per_gesture'])} final-check "
                               f"tries of {name} in a separate sitting.",
                               {"tries": 0, "recommended": int(t["final_check_tries_per_gesture"]), "split": "test"}))
        for split in ("train", "validation"):
            flagged = _count(inv, "gestures", split, label)
            if flagged["flagged"]:
                word = _SPLIT_WORD[split]
                items.append(_item(f"gesture-flagged-{split}-{label}", "info", "gestures", split, name,
                                   f"{flagged['flagged']} of {_tries(flagged['tries'], word)} of {name} "
                                   f"{'was' if flagged['flagged'] == 1 else 'were'} flagged for capture quality "
                                   "when saved.",
                                   "They are still used. If the warning was right, record replacements and "
                                   "move the poor tries out of the profile folder.",
                                   {"flagged": flagged["flagged"], "tries": flagged["tries"], "split": split}))
    if report:
        items.extend(_gesture_result_items(report, names, source, t))
    return items


def _gesture_result_items(report, names, source, t) -> list[dict[str, Any]]:
    """Weakest gestures and confusions from validation only (never test)."""
    items = []
    honest = isinstance(report.get("validation_crossfit"), dict) and report["validation_crossfit"].get("per_label")
    metrics = report["validation_crossfit"] if honest else report.get("validation_ungated")
    if not isinstance(metrics, dict) or not metrics.get("per_label"):
        return items
    caveat = "" if honest else (" The acceptance limits were set on these same tries, so this may be "
                                "optimistic.")
    name = lambda label: names.get(label, label.title() if label != REST_LABEL else "Resting")
    trained = set((report.get("train_labels") or {}).keys())
    for label in sorted(set(names) - trained - {REST_LABEL}):
        if trained:
            items.append(_item(f"gesture-not-in-model-{label}", "info", "gestures", "models", name(label),
                               f"{name(label)} is not in the newest gesture candidate ({source}).",
                               "Train a new gesture candidate after recording it.",
                               {"candidate": source}))
    per_label = metrics["per_label"]
    for label, row in sorted(per_label.items()):
        if label == REST_LABEL or row.get("trials", 0) < t["min_tries_to_judge"]:
            continue
        found, total = row.get("correct", 0), row["trials"]
        if found / total < t["weak_recall"]:
            items.append(_item(f"gesture-weak-{label}", "warn", "gestures", "train", name(label),
                               f"{name(label)} was recognised in {found} of {total} check tries ({source})."
                               + caveat,
                               f"Record 5 more practice tries of {name(label)} on a new occasion, making the "
                               "movement the way you naturally would.",
                               {"found": found, "tries": total, "candidate": source, "split": "validation",
                                "basis": "crossfit" if honest else "in_sample"}))
    confusion = {}
    for key, n in (metrics.get("confusion") or {}).items():
        true, _, pred = key.partition(" -> ")
        if n and pred not in ("<none>", true):
            confusion[(true, pred)] = n
    rest_row = per_label.get(REST_LABEL)
    if rest_row and rest_row.get("trials"):
        fired = rest_row["trials"] - rest_row.get("correct", 0)
        if fired:
            which = sorted(pred for (true, pred) in confusion if true == REST_LABEL)
            items.append(_item("gesture-rest-fired", "warn", "gestures", "train", "Resting",
                               f"Hands at rest triggered a gesture {_times(fired)} in "
                               f"{_tries(rest_row['trials'], 'resting check')}"
                               + (f", taken for {', '.join(name(w) for w in which)}" if which else "")
                               + f" ({source})." + caveat,
                               "Record more ordinary resting moments in Practice, including small everyday "
                               "movements such as scratching or reaching, so the app learns to stay silent.",
                               {"fired": fired, "tries": rest_row["trials"], "candidate": source,
                                "split": "validation", "basis": "crossfit" if honest else "in_sample"}))
    for (true, pred), n in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0])):
        if REST_LABEL in (true, pred):
            continue
        total = per_label.get(true, {}).get("trials", 0)
        both_ways = (pred, true) in confusion
        if both_ways and true > pred:
            continue  # reported once, from the other side
        lookalike = n >= t["lookalike_confusions"] or both_ways
        items.append(_item(f"gesture-confused-{true}-{pred}", "warn", "gestures", "setup" if lookalike else "train",
                           f"{name(true)} / {name(pred)}",
                           f"{name(true)} was taken for {name(pred)} {_times(n)} in "
                           f"{_tries(total, 'check')} of {name(true)}"
                           + (f", and {name(pred)} for {name(true)} {_times(confusion[(pred, true)])}" if both_ways else "")
                           + f" ({source}).",
                           (f"{name(true)} and {name(pred)} look alike to the model. Consider changing one of "
                            "the movements, then record it again." if lookalike else
                            f"Record more practice tries of both {name(true)} and {name(pred)} on a new occasion."),
                           {"confused": n, "tries": total, "candidate": source, "split": "validation"}))
    selection = report.get("variant_selection") or {}
    if selection.get("status") == "baseline_fallback_validation_too_small":
        suff = selection.get("validation_sufficiency") or {}
        short = [f"{name(label)} ({n} of {int(t['check_tries_per_gesture'])})"
                 for label, n in sorted((suff.get("gesture_tries") or {}).items())
                 if n < t["check_tries_per_gesture"]]
        if suff.get("rest_tries", t["check_rest_tries"]) < t["check_rest_tries"]:
            short.append(f"resting ({suff['rest_tries']} of {int(t['check_rest_tries'])})")
        items.append(_item("gesture-variants-not-compared", "info", "gestures", "validation", "Gesture trainer",
                           "The trainer kept its plain model without comparing its options, because its check "
                           "session had too few tries" + (": " + ", ".join(short) if short else "") + ".",
                           f"Record at least {int(t['check_tries_per_gesture'])} check tries of every gesture and "
                           f"{int(t['check_rest_tries'])} resting check tries, then train again.",
                           {"candidate": source}))
    return items


def _lip_items(inv, prompts, report, source, t) -> list[dict[str, Any]]:
    items = []
    practice = inv.get("phrases", {}).get("train", {})
    check = inv.get("phrases", {}).get("validation", {})
    n_practice = sum(r["tries"] for r in practice.values())
    n_check = sum(r["tries"] for r in check.values())
    if not n_practice and not n_check and not report:
        # Lip reading works without personal training; do not nag someone who only uses gestures.
        return [_item("lips-not-started", "info", "lips", "train", "Lip reading",
                      "No lip tries are recorded yet. Lip reading works without them; personal lip training "
                      f"needs at least {int(t['lip_practice_clips'])} practice and "
                      f"{int(t['lip_check_clips'])} check tries.",
                      f"If you want a personal lip model, record {int(t['practice_tries_per_phrase'])} practice "
                      "tries of each phrase.",
                      {"tries": 0, "required_practice": int(t["lip_practice_clips"]),
                       "required_check": int(t["lip_check_clips"])})]
    if n_practice < t["lip_practice_clips"]:
        items.append(_item("lips-too-few-practice", "action", "lips", "train", "Lip practice",
                           f"There are {_tries(n_practice, 'lip practice')}; lip training needs at least "
                           f"{int(t['lip_practice_clips'])}.",
                           f"Record {_more_tries(int(t['lip_practice_clips']) - n_practice, 'lip practice')}, "
                           "spread over your phrases.",
                           {"tries": n_practice, "required": int(t["lip_practice_clips"]), "split": "train"}))
    if n_check < t["lip_check_clips"]:
        items.append(_item("lips-too-few-check", "action", "lips", "validation", "Lip check",
                           f"There are {_tries(n_check, 'lip check')}; lip training needs at least "
                           f"{int(t['lip_check_clips'])} to decide whether a candidate is better.",
                           f"Record {_more_tries(int(t['lip_check_clips']) - n_check, 'lip check')} on an "
                           "occasion different from practice.",
                           {"tries": n_check, "required": int(t["lip_check_clips"]), "split": "validation"}))
    never = [p for p in prompts if p.strip() and not practice.get(p.strip(), {}).get("tries")]
    if never and n_practice:
        shown = ", ".join(f"“{p}”" for p in never[: int(t["max_phrases_listed"])])
        more = f" and {len(never) - int(t['max_phrases_listed'])} more" if len(never) > t["max_phrases_listed"] else ""
        items.append(_item("lips-never-recorded", "warn", "lips", "train", "Phrases",
                           f"{len(never)} of {len(prompts)} phrases have no lip practice tries: {shown}{more}.",
                           f"Record {int(t['practice_tries_per_phrase'])} practice tries of each.",
                           {"phrases_without_tries": len(never), "phrases": len(prompts), "split": "train"}))
    few = [p for p in prompts if 0 < practice.get(p.strip(), {}).get("tries", 0) < t["practice_tries_per_phrase"]]
    if few:
        items.append(_item("lips-few-practice", "info", "lips", "train", "Phrases",
                           f"{len(few)} of {len(prompts)} phrases have fewer than "
                           f"{int(t['practice_tries_per_phrase'])} lip practice tries.",
                           "Record a few more practice tries of those phrases.",
                           {"phrases": len(few), "recommended": int(t["practice_tries_per_phrase"]), "split": "train"}))
    flagged = sum(r["flagged"] for split in ("train", "validation")
                  for r in inv.get("phrases", {}).get(split, {}).values())
    if flagged:
        total = sum(r["tries"] for split in ("train", "validation")
                    for r in inv.get("phrases", {}).get(split, {}).values())
        items.append(_item("lips-flagged", "info", "lips", "train", "Lip tries",
                           f"{flagged} of {_tries(total, 'lip practice and check')} were flagged for capture "
                           "quality when saved.",
                           "They are still used. If your mouth was hidden or the camera paused, record replacements.",
                           {"flagged": flagged, "tries": total}))
    if report:
        selected = (report.get("selection") or {}).get("selected")
        side = "candidate" if selected == "candidate" else "baseline"
        val = (report.get(side) or {}).get("val") or {}
        rows = val.get("transcripts") or []
        if rows:
            per_phrase: dict[str, list[int]] = {}
            for row in rows:
                errors, words = _wer(row.get("hypothesis", ""), row.get("text", ""))
                agg = per_phrase.setdefault(row.get("text", ""), [0, 0, 0])
                agg[0] += errors
                agg[1] += words
                agg[2] += 1
            worst = sorted(((e / w, p, e, w, n) for p, (e, w, n) in per_phrase.items() if w),
                           key=lambda x: (-x[0], x[1]))
            model = "your candidate" if side == "candidate" else "standard model"
            for rate, phrase, errors, words, n in worst[: int(t["max_phrases_listed"])]:
                if rate < t["high_phrase_wer"]:
                    break
                items.append(_item(f"lips-hard-phrase-{phrase}", "warn", "lips", "train", phrase,
                                   f"“{phrase}”: {errors} of {words} words wrong across {_tries(n, 'check')} "
                                   f"({model}, report of {source}).",
                                   f"Record {int(t['practice_tries_per_phrase'])} more practice tries of this "
                                   "phrase on a new occasion, facing the camera at a natural pace.",
                                   {"word_errors": errors, "words": words, "tries": n, "candidate": source,
                                    "model": side, "split": "validation"}))
    return items


def _language_items(report, source) -> list[dict[str, Any]]:
    overlap = (report or {}).get("text_overlap") or {}
    total, repeated = overlap.get("validation_phrases"), overlap.get("also_in_training")
    if not total or not repeated:
        return []
    return [_item("language-check-repeats-practice", "info", "language", "validation", "Word suggestions",
                  f"{repeated} of {total} check phrases also appear in practice, so the word-suggestion check "
                  f"shows how well these exact phrases were learned, not how it handles new wording ({source}).",
                  "This is expected when you use one phrase list. To check new wording, add a few phrases you "
                  "record only in “Check on another occasion”.",
                  {"repeated": repeated, "phrases": total, "candidate": source})]


def guidance(inv: dict[str, Any], gestures: list[dict[str, Any]], lip_prompts: list[str],
             reports: dict[str, tuple[str, dict[str, Any]] | None] | None = None,
             thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Ordered guidance items. ``reports`` maps kind -> (candidate id, report) or None."""
    t = {**THRESHOLDS, **(thresholds or {})}
    reports = reports or {}

    def pick(kind):
        entry = reports.get(kind)
        if not entry:
            return None, None
        cid, report = entry
        return (report if isinstance(report, dict) else None), f"candidate {str(cid)[:6]}"

    motion, motion_src = pick("motion")
    lips, lips_src = pick("lips")
    language, language_src = pick("language")
    items = (_gesture_items(inv, gestures, motion, motion_src, t)
             + _lip_items(inv, lip_prompts, lips, lips_src, t)
             + _language_items(language, language_src))
    items.sort(key=lambda i: (_SEVERITY_ORDER[i["severity"]], i["area"], i["id"]))
    return {
        "items": items,
        "thresholds": t,
        "sources": {k: (pick(k)[1] if pick(k)[0] else None) for k in ("motion", "lips", "language")},
        "data": {kind: {split: {k: dict(v) for k, v in rows.items()} for split, rows in inv.get(kind, {}).items()}
                 for kind in ("gestures", "phrases")},
        "notes": [
            "Counts of recorded tries are not results. Results come only from check (validation) tries.",
            "Final-check results are never used here.",
            "Recommended numbers are engineering defaults, not guarantees of accuracy.",
        ],
    }

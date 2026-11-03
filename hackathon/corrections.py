"""Explicit, local corrections that may inform later personal training."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def inspect_source_npz(
    path_val: Any,
    profile_dir: Path,
    *,
    heldout_sessions: set[str] | None = None,
    heldout_trials: set[str] | None = None,
) -> dict[str, Any]:
    """Inspect one NPZ source file inside profile_dir."""
    if not isinstance(path_val, (str, Path)):
        return {"status": "invalid", "relative_path": str(path_val), "session_id": None, "trial_id": None,
                "reason": f"invalid source path type: {type(path_val).__name__}"}
    try:
        raw_path = Path(path_val).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (profile_dir.resolve() / raw_path).resolve()
        relative = path.relative_to(profile_dir.resolve())
    except Exception as exc:
        return {"status": "invalid", "relative_path": str(path_val), "session_id": None, "trial_id": None,
                "reason": f"source escapes profile directory: {exc}"}

    rel_str = str(relative)
    if not path.is_file() or path.suffix != ".npz":
        return {"status": "invalid", "relative_path": rel_str, "session_id": None, "trial_id": None,
                "reason": f"source is not an existing NPZ file inside profile: {rel_str}"}

    path_split = None
    for part in relative.parts:
        if part in ("validation", "test"):
            return {"status": "heldout", "relative_path": rel_str, "session_id": None, "trial_id": None,
                    "reason": f"path contains held-out split {part!r}: {rel_str}"}
        if part == "train":
            path_split = "train"

    try:
        with np.load(path, allow_pickle=False) as z:
            z_split = None
            if "split" in z.files:
                val = z["split"]
                if val.ndim != 0:
                    return {"status": "contradiction", "relative_path": rel_str, "session_id": None, "trial_id": None,
                            "reason": f"{rel_str}: split metadata must be scalar"}
                z_split = str(val.item()).strip()
                if z_split in ("validation", "test"):
                    return {"status": "heldout", "relative_path": rel_str, "session_id": None, "trial_id": None,
                            "reason": f"NPZ metadata split is held-out ({z_split}): {rel_str}"}
                if path_split is not None and z_split != path_split:
                    return {"status": "contradiction", "relative_path": rel_str, "session_id": None, "trial_id": None,
                            "reason": f"split metadata {z_split!r} conflicts with path {rel_str}"}

            def _scalar_field(key: str) -> str | None:
                if key in z.files:
                    v = z[key]
                    if v.ndim != 0:
                        raise ValueError(f"{rel_str}: {key} must be scalar")
                    s = str(v.item()).strip()
                    return s or None
                return None

            try:
                session_id = _scalar_field("session_id")
                trial_id = _scalar_field("trial_id")
            except ValueError as e:
                return {"status": "invalid", "relative_path": rel_str, "session_id": None, "trial_id": None,
                        "reason": str(e)}

            if session_id and heldout_sessions and session_id in heldout_sessions:
                return {"status": "heldout", "relative_path": rel_str, "session_id": session_id, "trial_id": trial_id,
                        "reason": f"session {session_id!r} belongs to held-out split: {rel_str}"}
            if trial_id and heldout_trials and trial_id in heldout_trials:
                return {"status": "heldout", "relative_path": rel_str, "session_id": session_id, "trial_id": trial_id,
                        "reason": f"trial {trial_id!r} belongs to held-out split: {rel_str}"}

            is_train = (path_split == "train" or z_split == "train")
            if is_train and session_id and trial_id:
                return {"status": "verified_train", "relative_path": rel_str, "session_id": session_id, "trial_id": trial_id, "reason": None}

            missing = [k for k, ok in [("non-train split", is_train), ("missing session_id", bool(session_id)), ("missing trial_id", bool(trial_id))] if not ok]
            return {"status": "unknown", "relative_path": rel_str, "session_id": session_id, "trial_id": trial_id,
                    "reason": f"unverified provenance ({', '.join(missing)}): {rel_str}"}
    except Exception as exc:
        return {"status": "invalid", "relative_path": rel_str, "session_id": None, "trial_id": None,
                "reason": f"cannot read NPZ {rel_str}: {exc}"}


class CorrectionStore:
    """Append-only per-profile corrections; recording consent is never inferred."""

    def __init__(self, profile):
        self.profile = profile
        self.path = profile.profile_dir / "corrections.jsonl"

    def record(self, *, original, corrected, candidates=None, source_paths=None,
               learn=False) -> dict:
        if learn is not True:
            raise ValueError("Choose whether this correction may be used for learning")
        if not isinstance(original, str) or not isinstance(corrected, str):
            raise ValueError("Correction text must be strings")
        original = original.strip()
        corrected = corrected.strip()
        if not corrected or len(corrected) > 2000 or len(original) > 2000:
            raise ValueError("Corrected text must contain 1–2000 characters")
        if original == corrected:
            raise ValueError("The correction does not change the text")
        candidates = candidates or []
        if not isinstance(candidates, list) or any(not isinstance(c, str) for c in candidates):
            raise ValueError("Candidates must be text")
        entry = {
            "id": uuid.uuid4().hex,
            "created_at": time.time(),
            "profile_id": self.profile.id,
            "original": original,
            "corrected": corrected,
            "candidates": [c[:2000] for c in candidates[:10]],
            "learn": True,
            "sources": self._source_references(source_paths or []),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def _source_references(self, paths):
        references = []
        for value in paths:
            info = inspect_source_npz(value, self.profile.profile_dir)
            if info["status"] == "heldout":
                raise ValueError(f"Learning sources must not include held-out data: {info['reason']}")
            if info["status"] == "contradiction":
                raise ValueError(f"Learning source split contradiction: {info['reason']}")
            if info["status"] == "invalid":
                if "escapes profile directory" in info["reason"]:
                    raise ValueError(f"Learning sources must stay inside this profile: {info['reason']}")
                raise ValueError(f"Invalid learning source: {info['reason']}")
            references.append(info["relative_path"])
        return references

    def list(self):
        if not self.path.is_file():
            return []
        entries = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


def summarize_corrections(
    profile_dir: str | Path | Any,
    *,
    heldout_sessions: set[str] | None = None,
    heldout_trials: set[str] | None = None,
) -> dict[str, Any]:
    """Audit and summarize corrections for language training."""
    if hasattr(profile_dir, "profile_dir"):
        profile_dir = profile_dir.profile_dir
    profile_path = Path(profile_dir).expanduser().resolve()
    path = profile_path / "corrections.jsonl"

    total_records = 0
    valid_count = 0
    skipped_count = 0
    clean_texts: list[str] = []
    warnings: list[str] = []
    sessions: set[str] = set()
    trials: set[str] = set()

    verified_train = 0
    unknown_provenance = 0
    heldout_rejected = 0
    invalid_sources = 0

    if not path.is_file():
        return {
            "texts": [],
            "total_records": 0,
            "valid_count": 0,
            "skipped_count": 0,
            "source_coverage": {
                "verified_train": 0,
                "unknown_provenance": 0,
                "heldout_rejected": 0,
                "invalid_sources": 0,
                "total_records": 0,
            },
            "sessions": [],
            "trials": [],
            "warnings": [],
            "provenance_unknown": False,
            "correction_provenance_complete": False,
        }

    with path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                skipped_count += 1
                invalid_sources += 1
                warnings.append(f"Line {line_num}: malformed JSON in corrections.jsonl")
                continue

            if not (entry.get("learn") is True and entry.get("profile_id") == profile_path.name
                    and isinstance(entry.get("corrected"), str) and entry["corrected"].strip()):
                continue

            total_records += 1
            entry_id = entry.get("id", f"line_{line_num}")
            sources = entry.get("sources")

            if not sources:
                valid_count += 1
                unknown_provenance += 1
                text = entry["corrected"].strip()
                if text not in clean_texts:
                    clean_texts.append(text)
                continue

            if not isinstance(sources, list):
                skipped_count += 1
                invalid_sources += 1
                warnings.append(f"Correction {entry_id}: sources must be a list")
                continue

            entry_heldout = False
            entry_contradiction = False
            entry_invalid = False
            entry_unknown = False
            entry_sessions: list[str] = []
            entry_trials: list[str] = []

            for src in sources:
                info = inspect_source_npz(
                    src,
                    profile_path,
                    heldout_sessions=heldout_sessions,
                    heldout_trials=heldout_trials,
                )
                st = info["status"]
                if st == "heldout":
                    entry_heldout = True
                    warnings.append(f"Correction {entry_id} excluded: {info['reason']}")
                    break
                elif st == "contradiction":
                    entry_contradiction = True
                    warnings.append(f"Correction {entry_id} excluded: {info['reason']}")
                    break
                elif st == "invalid":
                    entry_invalid = True
                    warnings.append(f"Correction {entry_id} excluded: {info['reason']}")
                    break
                elif st == "unknown":
                    entry_unknown = True
                    warnings.append(f"Correction {entry_id}: {info['reason']}")
                    if info["session_id"]:
                        entry_sessions.append(info["session_id"])
                    if info["trial_id"]:
                        entry_trials.append(info["trial_id"])
                elif st == "verified_train":
                    entry_sessions.append(info["session_id"])
                    entry_trials.append(info["trial_id"])

            if entry_heldout:
                skipped_count += 1
                heldout_rejected += 1
            elif entry_contradiction or entry_invalid:
                skipped_count += 1
                invalid_sources += 1
            elif entry_unknown:
                valid_count += 1
                unknown_provenance += 1
                sessions.update(entry_sessions)
                trials.update(entry_trials)
                text = entry["corrected"].strip()
                if text not in clean_texts:
                    clean_texts.append(text)
            else:
                valid_count += 1
                verified_train += 1
                sessions.update(entry_sessions)
                trials.update(entry_trials)
                text = entry["corrected"].strip()
                if text not in clean_texts:
                    clean_texts.append(text)

    has_unknown = (unknown_provenance > 0)
    has_invalid = (invalid_sources > 0)
    correction_provenance_complete = (valid_count > 0 and not has_unknown and not has_invalid)
    provenance_unknown = has_unknown

    if has_unknown:
        warnings.append(
            f"{unknown_provenance} correction(s) learned with explicit consent but unverified/unknown provenance: "
            "evaluation cannot claim independent validation"
        )

    return {
        "texts": clean_texts,
        "total_records": total_records,
        "valid_count": valid_count,
        "skipped_count": skipped_count,
        "source_coverage": {
            "verified_train": verified_train,
            "unknown_provenance": unknown_provenance,
            "heldout_rejected": heldout_rejected,
            "invalid_sources": invalid_sources,
            "total_records": total_records,
        },
        "sessions": sorted(sessions),
        "trials": sorted(trials),
        "warnings": warnings,
        "provenance_unknown": provenance_unknown,
        "correction_provenance_complete": correction_provenance_complete,
    }


def learning_texts(profile_dir) -> list[str]:
    """Return unique corrected texts explicitly marked for learning."""
    return summarize_corrections(profile_dir)["texts"]

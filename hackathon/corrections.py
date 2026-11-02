"""Explicit, local corrections that may inform later personal training."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


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
            path = Path(value).expanduser().resolve()
            try:
                relative = path.relative_to(self.profile.profile_dir.resolve())
            except ValueError:
                raise ValueError("Learning sources must stay inside this profile") from None
            if not path.is_file() or path.suffix != ".npz":
                raise ValueError("Learning sources must be existing NPZ files inside this profile")
            references.append(str(relative))
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


def learning_texts(profile_dir) -> list[str]:
    """Return unique corrected texts explicitly marked for learning."""
    path = Path(profile_dir) / "corrections.jsonl"
    if not path.is_file():
        return []
    texts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            if (entry.get("learn") is True and entry.get("profile_id") == path.parent.name
                    and isinstance(entry.get("corrected"), str) and entry["corrected"].strip()):
                text = entry["corrected"].strip()
                if text not in texts:
                    texts.append(text)
    return texts

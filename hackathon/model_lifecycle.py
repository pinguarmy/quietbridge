"""Profile-scoped candidate training and explicit model promotion.

Training runs in a separate Python process without a shell. Candidates and their
reports remain immutable; promotion copies a candidate to the runtime path.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from hackathon.profiles import Profile

_KINDS = {
    "motion": ("hackathon.signs.train_motion", "motion.npz", "sequences_dir"),
    "lips": ("hackathon.signs.train_lips", "lip.pth", "mouths_dir"),
    "language": ("hackathon.signs.train_language", "lm.pth", "mouths_dir"),
}


def _kind(kind: str) -> tuple[str, str, str]:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError(f"Unknown model kind: {kind!r}")
    return _KINDS[kind]


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _stderr_summary(text: str) -> str:
    """Expose only a Python exception class or a generic trainer error marker."""
    lines = text.splitlines()
    for line in reversed(lines):
        match = re.match(r"^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception|Exit))(?::|$)", line.strip())
        if match:
            return f"{match.group(1)} (details withheld)"
    return "Trainer reported an error (details withheld)" if text.strip() else ""


class ModelLifecycle:
    """One shared manager tracks subprocesses for all profiles in this server."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[tuple[str, str], subprocess.Popen] = {}
        self._stderr: dict[tuple[str, str], list[str]] = {}
        self._readers: dict[tuple[str, str], threading.Thread] = {}

    @staticmethod
    def _root(profile: Profile) -> Path:
        return profile.profile_dir / "models"

    @staticmethod
    def _candidate_dir(profile: Profile, candidate_id: str) -> Path:
        if not isinstance(candidate_id, str) or len(candidate_id) != 32 or any(c not in "0123456789abcdef" for c in candidate_id):
            raise ValueError("Invalid candidate id")
        return ModelLifecycle._root(profile) / "candidates" / candidate_id

    def start_training(self, profile: Profile, kind: str) -> dict[str, Any]:
        module, filename, data_attr = _kind(kind)
        with self._lock:
            for candidate in self.status(profile)["jobs"]:
                if candidate["kind"] == kind:
                    raise ValueError(f"{kind} training is already running for this profile")
            data = getattr(profile, data_attr)
            if kind == "motion":
                from hackathon.signs.motion import check_session_leakage, load_trials
                splits = [load_trials(data, split, "") for split in ("train", "validation", "test")]
                if not splits[0] or not splits[1]:
                    raise ValueError("Independent train and validation recordings are required")
                for i in range(len(splits)):
                    for j in range(i + 1, len(splits)):
                        check_session_leakage(splits[i], splits[j])
            elif kind == "lips":
                from hackathon.signs.train_lips import independent_partition, load_dataset
                independent_partition(load_dataset(data))
            else:
                from hackathon.signs.train_language import load_text_metadata
                load_text_metadata(data)
            candidate_id = uuid.uuid4().hex
            directory = self._candidate_dir(profile, candidate_id)
            directory.mkdir(parents=True, exist_ok=False)
            output = directory / filename
            report = output.with_suffix(".report.json")
            record = {"id": candidate_id, "kind": kind, "status": "running",
                      "model_path": str(output), "report_path": str(report), "error": None}
            _atomic_json(directory / "candidate.json", record)
            args = [sys.executable, "-m", module, "--data", str(getattr(profile, data_attr)),
                    "--output", str(output)]
            if kind == "motion":
                args += ["--session-prefix", ""]
            try:
                process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                           shell=False, cwd=str(Path(__file__).resolve().parent.parent))
                key = (str(profile.profile_dir.resolve()), candidate_id)
                self._jobs[key] = process
                chunks: list[str] = []
                self._stderr[key] = chunks

                def drain() -> None:
                    assert process.stderr is not None
                    while True:
                        data = process.stderr.read(4096)
                        if not data:
                            break
                        chunks.append(data.decode("utf-8", errors="replace"))
                        # Bound memory even if the trainer emits an enormous traceback.
                        while sum(len(chunk) for chunk in chunks) > 16384:
                            chunks.pop(0)
                    process.stderr.close()

                reader = threading.Thread(target=drain, daemon=True)
                self._readers[key] = reader
                reader.start()
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__} (details withheld)")
                _atomic_json(directory / "candidate.json", record)
                raise RuntimeError(record["error"]) from None
            return self._describe(profile, record)

    def _describe(self, profile: Profile, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        key = (str(profile.profile_dir.resolve()), record["id"])
        if record["status"] == "running":
            process = self._jobs.get(key)
            if process is None:
                record.update(status="failed", error="Training process is no longer tracked")
            else:
                code = process.poll()
                if code is not None:
                    reader = self._readers.pop(key)
                    reader.join()
                    summary = _stderr_summary("".join(self._stderr.pop(key)))
                    error = f"Trainer exited with code {code}"
                    if summary:
                        error += f": {summary}"
                    record.update(status="completed" if code == 0 else "failed",
                                  error=None if code == 0 else error)
                    self._jobs.pop(key, None)
            if record["status"] != "running":
                _atomic_json(self._candidate_dir(profile, record["id"]) / "candidate.json", record)
        report = None
        try:
            with Path(record["report_path"]).open(encoding="utf-8") as f:
                report = json.load(f)
            if not isinstance(report, dict):
                report = None
        except (OSError, ValueError):
            pass
        selected = None
        if report and isinstance(report.get("selection"), dict):
            selected = report["selection"].get("selected") == "candidate"
        valid = False
        if report and not report.get("failures"):
            if record["kind"] == "motion":
                valid = bool(report.get("validation_trials")) and isinstance(report.get("validation_ungated"), dict)
            elif record["kind"] == "lips":
                valid = bool(report.get("counts", {}).get("validation")) and selected is True
            else:
                valid = bool(report.get("counts", {}).get("validation")) and selected is True and report.get("saved") is True
        record.update(report=report, selected=selected,
                      promotable=record["status"] == "completed" and valid and Path(record["model_path"]).is_file())
        return record

    def status(self, profile: Profile) -> dict[str, Any]:
        with self._lock:
            root = self._root(profile)
            candidates = []
            for meta in sorted((root / "candidates").glob("*/candidate.json")):
                try:
                    record = json.loads(meta.read_text(encoding="utf-8"))
                    candidates.append(self._describe(profile, record))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            active = {kind: str(root / filename) if (root / filename).is_file() else None
                      for kind, (_, filename, _) in _KINDS.items()}
            return {"candidates": candidates, "active": active,
                    "jobs": [c for c in candidates if c["status"] == "running"]}

    def promote(self, profile: Profile, kind: str, candidate_id: str) -> dict[str, Any]:
        _, filename, _ = _kind(kind)
        with self._lock:
            directory = self._candidate_dir(profile, candidate_id)
            meta = directory / "candidate.json"
            if not meta.is_file():
                raise ValueError("Candidate not found")
            record = self._describe(profile, json.loads(meta.read_text(encoding="utf-8")))
            if record["kind"] != kind or not record["promotable"]:
                raise ValueError("Candidate is not eligible for promotion")
            root = self._root(profile)
            target = root / filename
            history = root / "history" / kind
            history.mkdir(parents=True, exist_ok=True)
            # Record absence as well as existing models, so rollback can restore stock.
            entry = history / f"{time.time_ns()}-{uuid.uuid4().hex}"
            entry.mkdir()
            if target.is_file():
                shutil.copyfile(target, entry / filename)
            _atomic_copy(directory / filename, target)
            return self.status(profile)

    def rollback(self, profile: Profile, kind: str) -> dict[str, Any]:
        _, filename, _ = _kind(kind)
        with self._lock:
            root = self._root(profile)
            history = root / "history" / kind
            entries = sorted(p for p in history.glob("*") if p.is_dir())
            if not entries:
                raise ValueError("No previous model to roll back to")
            entry = entries[-1]
            target = root / filename
            previous = entry / filename
            if previous.is_file():
                _atomic_copy(previous, target)
            else:
                target.unlink(missing_ok=True)
            # Consume the rollback point, preserving its contents for audit.
            archived = root / "history" / "used" / kind / entry.name
            archived.parent.mkdir(parents=True, exist_ok=True)
            entry.rename(archived)
            return self.status(profile)

"""Per-user profile management and data isolation for QuietBridge.

Provides safe profile creation, switching, customization of gesture vocabulary and
lip prompts, and directory isolation for hand sequences and mouth crops.

Profiles are stored under ``~/.local/share/quietbridge/profiles/<profile_id>`` by default,
or redirected via ``QUIETBRIDGE_DATA_ROOT`` or an explicit ``root_dir``.
Existing data outside the profiles directory is never migrated or deleted.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".local" / "share" / "quietbridge"
SPLITS = ("train", "validation", "test")

_PROFILE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_GESTURE_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
MAX_PROMPT_LEN = 300

DEFAULT_GESTURES: list[dict[str, Any]] = [
    {
        "label": "HELLO",
        "display_name": "Hello",
        "concept": "Hello / Greeting",
        "description": "Raise one open hand. Gently curl your fingers in, then open them once.",
        "hint": "Start still for 0.5 seconds, do the small comfortable motion once, then return to REST for 0.5 seconds. Aim for 3–5 seconds; finish within 15 seconds.",
        "emoji": "👋",
    },
    {
        "label": "WELCOME",
        "display_name": "Welcome",
        "concept": "Welcome / Inward welcome",
        "description": "Bring both hands close together in front of your chest, then draw them back toward your chest.",
        "hint": "This is a fixed custom demo motion, not a standard sign. Start still for 0.5 seconds, do it once, return to REST for 0.5 seconds. Aim for 3–5 seconds.",
        "emoji": "🫶",
    },
    {
        "label": "WATER",
        "display_name": "Water",
        "concept": "water",
        "description": "Hold one relaxed hand in front of you, then move it toward yourself once.",
        "hint": "Start still for 0.5 seconds, make one comfortable inward motion, then return to REST. Aim for 3–5 seconds.",
        "emoji": "💧",
    },
    {
        "label": "YES",
        "display_name": "Yes",
        "concept": "yes",
        "description": "Hold one relaxed fist in front of you and move it down and up once.",
        "hint": "Start still for 0.5 seconds, make one small nodding motion, then return to REST. Aim for 3–5 seconds.",
        "emoji": "✅",
    },
    {
        "label": "NO",
        "display_name": "No",
        "concept": "no",
        "description": "Hold one open hand in front of you and move it a little to the side and back once.",
        "hint": "Start still for 0.5 seconds, make one small sideways motion, then return to REST. Aim for 3–5 seconds.",
        "emoji": "🙅",
    },
    {
        "label": "REST",
        "display_name": "Rest",
        "concept": "Rest / Relaxed negative class",
        "description": "Relax into a loose half-fist, or lower your hands. No gesture or command.",
        "hint": "Record ordinary relaxed moments as the negative class. REST is its own label; it is not NEUTRAL.",
        "emoji": "😌",
    },
]

DEFAULT_LIP_PROMPTS: list[str] = [
    "Hello, how are you?",
    "Please wait a moment.",
    "I need some help.",
    "I would like some water.",
    "Yes, that works for me.",
    "No, I do not agree.",
    "Can you repeat that?",
    "I want to take a break.",
    "Please write it down.",
    "I will be back this afternoon.",
]


def get_default_root() -> Path:
    override = os.environ.get("QUIETBRIDGE_DATA_ROOT") or os.environ.get("QUIETBRIDGE_ROOT")
    if override:
        return Path(override).expanduser()
    return DEFAULT_ROOT


def is_safe_profile_id(profile_id: str) -> bool:
    if not isinstance(profile_id, str) or not profile_id:
        return False
    if not _PROFILE_ID_RE.match(profile_id):
        return False
    if Path(profile_id).name != profile_id:
        return False
    if profile_id in (".", ".."):
        return False
    return True


def validate_gestures(gestures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(gestures, list) or not gestures:
        raise ValueError("gestures must be a non-empty list")
    validated = []
    seen_labels = set()
    for g in gestures:
        if not isinstance(g, dict):
            raise ValueError("Each gesture must be an object")
        label = str(g.get("label", "")).strip()
        if not _GESTURE_LABEL_RE.match(label):
            raise ValueError(f"Gesture label {label!r} must be a safe UPPER_SNAKE_CASE identifier")
        if label in seen_labels:
            raise ValueError(f"Duplicate gesture label {label!r}")
        seen_labels.add(label)
        display_name = str(g.get("display_name") or label).strip()
        concept = str(g.get("concept") or display_name).strip()
        description = str(g.get("description") or "").strip()
        hint = str(g.get("hint") or "").strip()
        emoji = str(g.get("emoji") or "✋").strip()
        validated.append({
            "label": label,
            "display_name": display_name,
            "concept": concept,
            "description": description,
            "hint": hint,
            "emoji": emoji,
        })
    return validated


def validate_lip_prompts(prompts: list[str]) -> list[str]:
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("lip_prompts must be a non-empty list of strings")
    validated = []
    for p in prompts:
        if not isinstance(p, str):
            raise ValueError("Each lip prompt must be a string")
        text = p.strip()
        if not text or len(text) > MAX_PROMPT_LEN:
            raise ValueError(f"Lip prompt must be 1-{MAX_PROMPT_LEN} characters")
        validated.append(text)
    return validated


def check_session_split(data_root: Path | str, split: str, session: str) -> None:
    """Refuse split leakage: session_id must not exist in any other split."""
    data_root = Path(data_root)
    for other in SPLITS:
        if other == split:
            continue
        other_dir = data_root / other
        if other_dir.is_dir() and any(other_dir.glob(f"{session}_*.npz")):
            raise ValueError(
                f"session {session!r} already has recordings in {other}; refusing split leakage"
            )


@dataclass
class Profile:
    id: str
    name: str
    root_dir: Path
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    gestures: list[dict[str, Any]] = field(default_factory=lambda: [dict(g) for g in DEFAULT_GESTURES])
    lip_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_LIP_PROMPTS))

    @property
    def profile_dir(self) -> Path:
        return self.root_dir / "profiles" / self.id

    @property
    def sequences_dir(self) -> Path:
        return self.profile_dir / "sequences"

    @property
    def mouths_dir(self) -> Path:
        return self.profile_dir / "mouths"

    @property
    def config_path(self) -> Path:
        return self.profile_dir / "profile.json"

    def ensure_dirs(self) -> None:
        for split in SPLITS:
            (self.sequences_dir / split).mkdir(parents=True, exist_ok=True)
            (self.mouths_dir / split).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.ensure_dirs()
        self.updated_at = time.time()
        payload = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "gestures": self.gestures,
            "lip_prompts": self.lip_prompts,
        }
        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.config_path)

    def to_dict(self, include_stats: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "gestures": [dict(g) for g in self.gestures],
            "lip_prompts": list(self.lip_prompts),
            "paths": {
                "profile": str(self.profile_dir),
                "sequences": str(self.sequences_dir),
                "mouths": str(self.mouths_dir),
            },
        }
        if include_stats:
            data["stats"] = self.get_stats()
        return data

    def get_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {
            "sequences": {},
            "mouths": {},
        }
        for split in SPLITS:
            s_dir = self.sequences_dir / split
            m_dir = self.mouths_dir / split
            stats["sequences"][split] = len(list(s_dir.glob("*.npz"))) if s_dir.is_dir() else 0
            stats["mouths"][split] = len(list(m_dir.glob("*.npz"))) if m_dir.is_dir() else 0
        return stats


class ProfileManager:
    def __init__(self, root_dir: Path | str | None = None):
        self.root_dir = Path(root_dir).expanduser() if root_dir else get_default_root()
        self.profiles_dir = self.root_dir / "profiles"

    @property
    def active_file(self) -> Path:
        return self.profiles_dir / "active_profile.txt"

    @property
    def active_profile_id(self) -> str:
        if self.active_file.is_file():
            content = self.active_file.read_text(encoding="utf-8").strip()
            if is_safe_profile_id(content) and (self.profiles_dir / content / "profile.json").is_file():
                return content
        return "default"

    def list_profiles(self, include_stats: bool = True) -> list[dict[str, Any]]:
        self.ensure_default_profile()
        profiles = []
        if self.profiles_dir.is_dir():
            for folder in sorted(self.profiles_dir.iterdir()):
                if folder.is_dir() and (folder / "profile.json").is_file():
                    try:
                        p = self.get_profile(folder.name)
                        profiles.append(p.to_dict(include_stats=include_stats))
                    except Exception:
                        continue
        return profiles

    def get_profile(self, profile_id: str) -> Profile:
        if not is_safe_profile_id(profile_id):
            raise ValueError(f"Invalid profile id: {profile_id!r}")
        cfg = self.profiles_dir / profile_id / "profile.json"
        if not cfg.is_file():
            if profile_id == "default":
                return self.ensure_default_profile()
            raise KeyError(f"Profile {profile_id!r} not found")
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return Profile(
            id=data["id"],
            name=data.get("name", data["id"]),
            root_dir=self.root_dir,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            gestures=data.get("gestures", [dict(g) for g in DEFAULT_GESTURES]),
            lip_prompts=data.get("lip_prompts", list(DEFAULT_LIP_PROMPTS)),
        )

    def create_profile(
        self,
        profile_id: str,
        name: str,
        gestures: list[dict[str, Any]] | None = None,
        lip_prompts: list[str] | None = None,
    ) -> Profile:
        profile_id = (profile_id or "").strip()
        if not is_safe_profile_id(profile_id):
            raise ValueError(
                "Profile id must be 1-64 characters matching [a-zA-Z0-9_-], and cannot be '.' or '..'"
            )
        target = self.profiles_dir / profile_id
        if target.exists():
            raise ValueError(f"Profile {profile_id!r} already exists")
        name = (name or profile_id).strip()
        if not name or len(name) > 64:
            raise ValueError("Profile name must be 1-64 characters")

        valid_gestures = validate_gestures(gestures) if gestures is not None else [dict(g) for g in DEFAULT_GESTURES]
        valid_prompts = validate_lip_prompts(lip_prompts) if lip_prompts is not None else list(DEFAULT_LIP_PROMPTS)

        profile = Profile(
            id=profile_id,
            name=name,
            root_dir=self.root_dir,
            gestures=valid_gestures,
            lip_prompts=valid_prompts,
        )
        profile.save()
        return profile

    def update_profile(
        self,
        profile_id: str,
        name: str | None = None,
        gestures: list[dict[str, Any]] | None = None,
        lip_prompts: list[str] | None = None,
    ) -> Profile:
        profile = self.get_profile(profile_id)
        if name is not None:
            name = name.strip()
            if not name or len(name) > 64:
                raise ValueError("Profile name must be 1-64 characters")
            profile.name = name
        if gestures is not None:
            profile.gestures = validate_gestures(gestures)
        if lip_prompts is not None:
            profile.lip_prompts = validate_lip_prompts(lip_prompts)
        profile.save()
        return profile

    def set_active_profile(self, profile_id: str) -> Profile:
        profile = self.get_profile(profile_id)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.active_file.write_text(profile.id + "\n", encoding="utf-8")
        return profile

    def get_active_profile(self) -> Profile:
        pid = self.active_profile_id
        try:
            return self.get_profile(pid)
        except KeyError:
            return self.ensure_default_profile()

    def ensure_default_profile(self) -> Profile:
        cfg = self.profiles_dir / "default" / "profile.json"
        if cfg.is_file():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                return Profile(
                    id="default",
                    name=data.get("name", "Default Profile"),
                    root_dir=self.root_dir,
                    created_at=data.get("created_at", time.time()),
                    updated_at=data.get("updated_at", time.time()),
                    gestures=data.get("gestures", [dict(g) for g in DEFAULT_GESTURES]),
                    lip_prompts=data.get("lip_prompts", list(DEFAULT_LIP_PROMPTS)),
                )
            except Exception:
                pass
        profile = Profile(
            id="default",
            name="Default Profile",
            root_dir=self.root_dir,
            gestures=[dict(g) for g in DEFAULT_GESTURES],
            lip_prompts=list(DEFAULT_LIP_PROMPTS),
        )
        profile.save()
        return profile


def profile_vocabulary(profile: Profile) -> dict[str, dict[str, Any]]:
    """Translate custom concepts into the shared, confirmation-first vocabulary."""
    vocabulary = {}
    for gesture in profile.gestures:
        if gesture["label"] == "REST":
            continue
        matches = list(dict.fromkeys(
            value.strip() for value in (gesture["concept"], gesture["display_name"])
            if value and value.strip()
        ))
        vocabulary[gesture["label"]] = {
            "display_name": gesture["display_name"],
            "phrase": gesture["concept"],
            "matches": matches,
            "enabled": True,
            "verified": False,
        }
    return vocabulary


def profile_motion_model(profile: Profile) -> Path:
    return profile.profile_dir / "models" / "motion.npz"

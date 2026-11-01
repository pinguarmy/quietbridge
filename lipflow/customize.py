"""Local starter vocabulary and phrase customization for Lipflow.

Allows importing plain-text phrases/words, merging with existing files non-destructively,
seeding generic starter vocabulary, and updating settings profile options.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from .dictation import load_settings, save_settings
from .paths import HOME
from .personal import PHRASES, words_of
from .vocab import PATH as WORDS_PATH

# Generic keyword / vocabulary placeholders - no personal fact invention
GENERIC_STARTER_WORDS = [
    "Lipflow",
    "LLM",
    "Python",
    "macOS",
    "Slack",
    "Zoom",
    "GitHub",
    "VSCode",
    "API",
    "PR",
]

# Practical everyday dictation starter phrases
GENERIC_STARTER_PHRASES = [
    "Sounds good to me, let us proceed.",
    "Can you share the link when you get a chance?",
    "I will look into this right now.",
    "Thanks for the update, talk soon.",
    "Let me know if you need anything else.",
    "I am running five minutes late to the meeting.",
    "Please review the pull request when you have time.",
    "Let us follow up on this tomorrow morning.",
]


def load_lines(path: str) -> list[str]:
    """Load non-empty, non-comment stripped lines from a text file."""
    if not os.path.exists(path):
        return []
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def merge_unique(existing: Iterable[str], incoming: Iterable[str], case_insensitive: bool = False) -> list[str]:
    """Merge incoming items into existing list preserving order and without duplicates.
    Existing items are always kept as-is.
    """
    res: list[str] = []
    seen = set()

    for item in existing:
        s = item.strip()
        if not s:
            continue
        key = s.lower() if case_insensitive else s
        if key not in seen:
            seen.add(key)
            res.append(s)

    for item in incoming:
        s = item.strip()
        if not s:
            continue
        key = s.lower() if case_insensitive else s
        if key not in seen:
            seen.add(key)
            res.append(s)

    return res


def init_starter_files(merge_if_exists: bool = True) -> dict[str, Any]:
    """Initialize words.txt and phrases.txt non-destructively.
    If files already exist, never overwrite them; merges generic starter placeholders if missing.
    """
    os.makedirs(HOME, exist_ok=True)
    created = {"words": False, "phrases": False}

    # 1. words.txt
    existing_words = load_lines(WORDS_PATH) if os.path.exists(WORDS_PATH) else []
    if not os.path.exists(WORDS_PATH):
        created["words"] = True
    if not os.path.exists(WORDS_PATH) or merge_if_exists:
        merged_words = merge_unique(existing_words, GENERIC_STARTER_WORDS, case_insensitive=True)
        content = (
            "# Lipflow custom words: one name or term per line, written how you want it typed.\n"
            "# When one of the model's top guesses contains a word from this list, that guess wins.\n"
            + "\n".join(merged_words)
            + "\n"
        )
        with open(WORDS_PATH, "w", encoding="utf-8") as f:
            f.write(content)

    # 2. phrases.txt
    existing_phrases = load_lines(PHRASES) if os.path.exists(PHRASES) else []
    if not os.path.exists(PHRASES):
        created["phrases"] = True
    if not os.path.exists(PHRASES) or merge_if_exists:
        merged_phrases = merge_unique(existing_phrases, GENERIC_STARTER_PHRASES, case_insensitive=True)
        with open(PHRASES, "w", encoding="utf-8") as f:
            f.write("\n".join(merged_phrases) + "\n")

    return {
        "words_path": WORDS_PATH,
        "phrases_path": PHRASES,
        "created": created,
        "total_words": len(load_lines(WORDS_PATH)),
        "total_phrases": len(load_lines(PHRASES)),
    }


def import_plain_text(
    phrases: Iterable[str] | None = None,
    words: Iterable[str] | None = None,
    phrases_file: str | None = None,
    words_file: str | None = None,
) -> dict[str, Any]:
    """Import and merge phrases and vocabulary non-destructively into existing files.
    Never overwrites existing phrases or words; appends new items preserving order.
    """
    os.makedirs(HOME, exist_ok=True)

    # Collect incoming phrases
    in_phrases: list[str] = []
    if phrases:
        in_phrases.extend(phrases)
    if phrases_file and os.path.exists(phrases_file):
        with open(phrases_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # split sentences if multiple
                    for s in re.split(r"[\r\n]+", line):
                        s = s.strip()
                        if len(words_of(s)) >= 2:
                            in_phrases.append(s)

    # Collect incoming words
    in_words: list[str] = []
    if words:
        in_words.extend(words)
    if words_file and os.path.exists(words_file):
        with open(words_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Avoid polluting vocab with whole sentences
                    tokens = words_of(line)
                    if len(tokens) <= 3 and not line.endswith((".", "!", "?")):
                        in_words.append(line)
                    else:
                        for t in re.findall(r"\b[A-Za-z0-9'-]+\b", line):
                            if len(t) > 1 and t.lower() not in {"the", "and", "that", "this", "with", "from"}:
                                in_words.append(t)

    # Existing phrases
    existing_phrases = load_lines(PHRASES) if os.path.exists(PHRASES) else []
    merged_phrases = merge_unique(existing_phrases, in_phrases, case_insensitive=True)
    with open(PHRASES, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_phrases) + ("\n" if merged_phrases else ""))

    # Existing words
    existing_words = load_lines(WORDS_PATH) if os.path.exists(WORDS_PATH) else []
    merged_words = merge_unique(existing_words, in_words, case_insensitive=True)
    with open(WORDS_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Lipflow custom words: one name or term per line, written how you want it typed.\n"
            "# When one of the model's top guesses contains a word from this list, that guess wins.\n"
            + "\n".join(merged_words)
            + ("\n" if merged_words else "")
        )

    return {
        "phrases_before": len(existing_phrases),
        "phrases_after": len(merged_phrases),
        "phrases_added": len(merged_phrases) - len(existing_phrases),
        "words_before": len(existing_words),
        "words_after": len(merged_words),
        "words_added": len(merged_words) - len(existing_words),
        "phrases_path": PHRASES,
        "words_path": WORDS_PATH,
    }


def update_custom_settings(
    cleanup_backend: str | None = None,
    cleanup_style: str | None = None,
    gemini_model: str | None = None,
    cpa_base_url: str | None = None,
    cleanup_timeout: float | None = None,
    key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Safely update configuration/settings in settings.json."""
    settings = load_settings()

    if cleanup_backend is not None:
        settings["cleanup_backend"] = cleanup_backend
    if cleanup_style is not None:
        settings["cleanup_style"] = cleanup_style
    if gemini_model is not None:
        settings["gemini_model"] = gemini_model
    if cpa_base_url is not None:
        settings["cpa_base_url"] = cpa_base_url
    if cleanup_timeout is not None:
        settings["cleanup_timeout"] = float(cleanup_timeout)
    if key is not None:
        settings["key"] = key

    for k, v in kwargs.items():
        if v is not None:
            settings[k] = v

    save_settings(settings)
    return settings


def get_custom_status() -> dict[str, Any]:
    """Return local customization status for words, phrases, and settings."""
    settings = load_settings()
    num_words = len(load_lines(WORDS_PATH)) if os.path.exists(WORDS_PATH) else 0
    num_phrases = len(load_lines(PHRASES)) if os.path.exists(PHRASES) else 0

    return {
        "words_path": WORDS_PATH,
        "words_count": num_words,
        "phrases_path": PHRASES,
        "phrases_count": num_phrases,
        "cleanup_backend": settings.get("cleanup_backend", "auto"),
        "cleanup_style": settings.get("cleanup_style", "polish"),
        "gemini_model": settings.get("gemini_model", "gemini-3.8-flash"),
        "cpa_base_url": settings.get("cpa_base_url", "https://cpa.peterhpy.dpdns.org/v1"),
        "cleanup_timeout": settings.get("cleanup_timeout", 8.0),
        "key": settings.get("key", "right_option"),
        "cloud_upload_disclosure": (
            "Video frames and facial recordings are strictly processed locally and never uploaded. "
            "When cloud cleanup (gemini/claude) is active, candidate text and active application context "
            "are transmitted to the configured cloud endpoint for LLM polishing."
        ),
    }


def customize_handler(args: Any = None) -> int:
    """CLI handler for customization."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="lipflow customize",
        description="Prepare starter files and import custom words/phrases non-destructively.",
    )
    parser.add_argument("--init", action="store_true", help="Initialize starter vocabulary and phrases if missing")
    parser.add_argument("--import-phrases", type=str, default=None, help="Plain text file with phrases (one per line)")
    parser.add_argument("--import-words", type=str, default=None, help="Plain text file with custom words/names")
    parser.add_argument("--backend", type=str, default=None, help="Cleanup backend (e.g. gemini, local, basic)")
    parser.add_argument("--style", type=str, default=None, help="Cleanup style (faithful, polish)")
    parser.add_argument("--gemini-model", type=str, default=None, help="Gemini model name")
    parser.add_argument("--cpa-url", type=str, default=None, help="CPA OpenAI-compatible base URL")
    parser.add_argument("--status", action="store_true", help="Print current customization status")

    parsed = parser.parse_args(args)

    if parsed.status or (not parsed.init and not parsed.import_phrases and not parsed.import_words and not parsed.backend and not parsed.style and not parsed.gemini_model and not parsed.cpa_url):
        st = get_custom_status()
        print("Lipflow Customization Status:")
        print(f"  Words:   {st['words_count']} entries ({st['words_path']})")
        print(f"  Phrases: {st['phrases_count']} entries ({st['phrases_path']})")
        print(f"  Backend: {st['cleanup_backend']} (style: {st['cleanup_style']})")
        print(f"  Model:   {st['gemini_model']}")
        print(f"  BaseURL: {st['cpa_base_url']}")
        print(f"  Key:     {st['key']}")
        print(f"  Disclosure: Video stays 100% local; cloud cleanup transmits candidate & context text to {st['cpa_base_url']}.")
        return 0

    if parsed.init:
        res = init_starter_files()
        print(f"Initialized starter files: {res['total_words']} words, {res['total_phrases']} phrases.")

    if parsed.import_phrases or parsed.import_words:
        res = import_plain_text(phrases_file=parsed.import_phrases, words_file=parsed.import_words)
        print(f"Imported: +{res['phrases_added']} phrases (total {res['phrases_after']}), +{res['words_added']} words (total {res['words_after']}).")

    if any([parsed.backend, parsed.style, parsed.gemini_model, parsed.cpa_url]):
        update_custom_settings(
            cleanup_backend=parsed.backend,
            cleanup_style=parsed.style,
            gemini_model=parsed.gemini_model,
            cpa_base_url=parsed.cpa_url,
        )
        print("Updated settings successfully.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(customize_handler(sys.argv[1:]))

"""Merge the user's single customization worksheet into Lipflow's local profile."""
from __future__ import annotations

import argparse
from pathlib import Path

from lipflow.customize import import_plain_text, update_custom_settings
from lipflow.paths import HOME, WINDOWS


def parse_sheet(text):
    sections, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return sections


def apply(path):
    sections = parse_sheet(Path(path).read_text(encoding="utf-8"))
    settings = {}
    for line in sections.get("操作习惯", []):
        key, _, value = line.partition("：")
        value = value.strip()
        if key == "整理风格" and value:
            settings["cleanup_style"] = "faithful" if value in ("忠实", "faithful") else "polish"
        elif key == "快捷键" and value:
            # Parsing a worksheet must not load an OS keyboard hook. Resolve it
            # only when applying a shortcut, using the current platform's keys.
            if WINDOWS:
                from lipflow.win.hotkey import KEYS
            else:
                from lipflow.hotkey import KEYS
            code = value.lower().replace(" ", "_")
            if code not in KEYS:
                raise ValueError(f"快捷键需先匹配支持的按键，未自动修改：{value}")
            settings["key"] = code
    stats = import_plain_text(words=sections.get("常用名词", []), phrases=sections.get("常用英文句子", []))
    if settings:
        update_custom_settings(**settings)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(Path(HOME) / "customization.txt"))
    args = parser.parse_args()
    stats = apply(args.path)
    print(f"Merged customization: +{stats['words_added']} words, +{stats['phrases_added']} phrases.")
    print("Open Lipflow → Practice & train more to record your customized practice round.")


if __name__ == "__main__":
    main()

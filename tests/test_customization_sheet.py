import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest

from scripts import apply_customization
from scripts.apply_customization import parse_sheet


def test_single_sheet_separates_profile_words_and_sentences():
    sections = parse_sheet('''# 留注释
[操作习惯]
整理风格：自动整理
[常用名词]
Example Name
# ignored
[常用英文句子]
Can you send the document tomorrow morning?
[试用场景]
Notes
''')
    assert sections['常用名词'] == ['Example Name']
    assert sections['常用英文句子'] == ['Can you send the document tomorrow morning?']
    assert sections['操作习惯'] == ['整理风格：自动整理']


def test_loading_and_applying_sheet_without_shortcut_needs_no_keyboard_hook(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def without_keyboard_hooks(name, *args, **kwargs):
        if name in {"lipflow.hotkey", "lipflow.win.hotkey"}:
            raise AssertionError(f"Unexpected platform keyboard-hook import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_keyboard_hooks)
    importlib.reload(apply_customization)
    monkeypatch.setattr(apply_customization, "WINDOWS", True)
    changes = []
    monkeypatch.setattr(apply_customization, "import_plain_text", lambda **kwargs: kwargs)
    monkeypatch.setattr(apply_customization, "update_custom_settings", lambda **kwargs: changes.append(kwargs))
    sheet = tmp_path / "customization.txt"
    sheet.write_text("[操作习惯]\n整理风格：faithful\n[常用名词]\nQuietBridge\n", encoding="utf-8")

    result = apply_customization.apply(sheet)

    assert result == {"words": ["QuietBridge"], "phrases": []}
    assert changes == [{"cleanup_style": "faithful"}]


@pytest.mark.parametrize("windows,module_name,key", [
    (True, "lipflow.win.hotkey", "right_control"),
    (False, "lipflow.hotkey", "right_option"),
])
def test_shortcut_uses_only_current_platform_keys(tmp_path, monkeypatch, windows, module_name, key):
    original_import = builtins.__import__
    wrong_module = "lipflow.hotkey" if windows else "lipflow.win.hotkey"

    def current_platform_only(name, *args, **kwargs):
        if name == wrong_module:
            raise AssertionError(f"Imported another platform's keyboard hook: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", current_platform_only)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(KEYS={key: object()}))
    monkeypatch.setattr(apply_customization, "WINDOWS", windows)
    monkeypatch.setattr(apply_customization, "import_plain_text", lambda **kwargs: kwargs)
    changes = []
    monkeypatch.setattr(apply_customization, "update_custom_settings", lambda **kwargs: changes.append(kwargs))
    sheet = tmp_path / "customization.txt"
    sheet.write_text(f"[操作习惯]\n快捷键：{key.replace('_', ' ')}\n", encoding="utf-8")

    apply_customization.apply(sheet)

    assert changes == [{"key": key}]

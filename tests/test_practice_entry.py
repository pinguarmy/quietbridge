import sys
from types import SimpleNamespace

import pytest

import lipflow.__main__ as cli


def test_cli_direct_practice_entry(monkeypatch):
    seen = {}
    # This checks CLI routing, not a native global keyboard listener.
    windows = sys.platform == "win32"
    key = "right_control" if windows else "right_option"
    module = "lipflow.win.hotkey" if windows else "lipflow.hotkey"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(KEYS={key: 1}, DEFAULT_KEY=key))
    monkeypatch.setattr(cli, '_app', lambda: (lambda **kw: seen.update(kw), lambda opts: None))
    cli.main(['onboard', '--start', 'practice'])
    assert seen == {'onboard': True, 'onboarding_start': 'practice'}


def test_practice_gate_preserves_permissions_and_shows_once():
    pytest.importorskip("AppKit", reason="Native practice setup checks require macOS AppKit")
    from lipflow.app import Lipflow
    for allowed in (True, False):
        calls = []
        setup = SimpleNamespace(_perm_state=lambda: [allowed] * 3,
                                goPractice_=lambda sender: calls.append('practice'),
                                goPermissions_=lambda sender: calls.append('permissions'),
                                show=lambda: calls.append('show'))
        controller = SimpleNamespace(setup=setup, camera=SimpleNamespace(track_always=False))
        Lipflow.show_setup(controller, 'practice')
        assert calls == ['practice' if allowed else 'permissions', 'show']
        assert controller.camera.track_always

from types import SimpleNamespace

import lipflow.__main__ as cli


def test_cli_direct_practice_entry(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, '_app', lambda: (lambda **kw: seen.update(kw), lambda opts: None))
    cli.main(['onboard', '--start', 'practice'])
    assert seen == {'onboard': True, 'onboarding_start': 'practice'}


def test_practice_gate_preserves_permissions_and_shows_once():
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

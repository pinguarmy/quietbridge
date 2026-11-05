"""Owned Personalize entry and explicit new-message regression tests."""
import json
import threading
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import pytest

from hackathon.contracts import LipResult
from hackathon.profiles import ProfileManager
from hackathon.runtime import Runtime
from hackathon.server import _LazyLabEngine, make_server


def post(url, token, body=b'{}', origin=None, content_type='application/json'):
    headers = {'X-QuietBridge-Token': token, 'Content-Type': content_type}
    if origin:
        headers['Origin'] = origin
    return urlopen(Request(url, data=body, headers=headers), timeout=3)


def test_owned_lab_http_profile_return_security_and_close(tmp_path, monkeypatch):
    manager = ProfileManager(root_dir=tmp_path)
    profile = manager.create_profile('test-profile', 'Test profile')
    manager.set_active_profile(profile.id)
    runtime = Runtime(rehearsal=True, vocabulary={}, profile_manager=manager)
    runtime.isolated_sign_model_path = tmp_path / 'sign.json'
    info = {'status': 'capture_only', 'vocabulary': {}, 'language': 'BSL'}
    from hackathon.signs import lab as lab_module
    monkeypatch.setattr(lab_module, '_sign_model_info', lambda path: info if path == runtime.isolated_sign_model_path else None)
    real_engine = Mock(side_effect=AssertionError('real engine must not start'))
    monkeypatch.setattr(lab_module, 'Engine', real_engine)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    main = f'http://127.0.0.1:{server.server_port}'
    try:
        assert server._lab is None
        with urlopen(main + '/record', timeout=3) as response:
            location = response.url
            page = response.read()
            assert response.status == 200 and b'Personal' in page
        lab_url = urlsplit(location)
        assert lab_url.hostname == '127.0.0.1' and lab_url.port == server._lab.server_port
        assert parse_qs(lab_url.query)['return_to'] == [main + '/']
        base = f'http://127.0.0.1:{lab_url.port}'
        with urlopen(base + '/api/profiles', timeout=3) as response:
            assert json.load(response)['active_profile'] == profile.id
        with urlopen(base + '/api/state', timeout=3) as response:
            state = json.load(response)
        assert state['sign_model'] == info
        assert not server._lab.engine.ready
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/state', headers={'Host': 'example.com'}), timeout=3)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post(base + '/api/frame', state['token'], b'jpeg', origin='https://example.com', content_type='image/jpeg')
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post(base + '/api/frame', state['token'], b'jpeg', origin=base, content_type='image/jpeg')
        assert error.value.code == 400
        for path in ['/api/mouth/preview', '/api/lips/enable', '/api/paired/start']:
            with pytest.raises(HTTPError) as error:
                post(base + path, state['token'], origin=base)
            assert error.value.code == 400
        real_engine.assert_not_called()
        with urlopen(main + '/record', timeout=3) as response:
            assert urlsplit(response.url).port == lab_url.port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.close()
    assert not server._lab_thread
    with pytest.raises(OSError):
        urlopen(base + '/api/state', timeout=1)


def test_lazy_engine_loads_only_on_infer_and_closes(monkeypatch):
    from hackathon.signs import lab
    actual = Mock()
    monkeypatch.setattr(lab, 'Engine', Mock(return_value=actual))
    engine = _LazyLabEngine()
    assert not engine.ready
    lab.Engine.assert_not_called()
    engine.infer('frame', 1)
    engine.infer('frame', 2)
    lab.Engine.assert_called_once_with()
    assert engine.ready
    engine.close()
    actual.close.assert_called_once_with()
    assert not engine.ready
    rehearsal = _LazyLabEngine(rehearsal=True)
    with pytest.raises(RuntimeError):
        rehearsal.infer('frame', 1)
    lab.Engine.assert_called_once()


def test_new_message_resets_session_and_invalidates_late_cleanup():
    runtime = Runtime(rehearsal=True, vocabulary={})
    runtime.speaker.speak = Mock()
    try:
        with runtime.lock:
            runtime._publish_session(LipResult(['old message'], 0))
            runtime.state['last_correction'] = 'old correction'
            runtime.session_signs.append('old sign')
            runtime._session_conflict = True
        old = runtime.snapshot()
        old_generation = runtime._cleanup_generation
        with pytest.raises(ValueError):
            runtime.action('new_message', {'revision': old['revision'] - 1})
        assert runtime.snapshot()['decision']['text'] == 'old message'
        runtime.action('new_message', {'revision': old['revision']})
        current = runtime.snapshot()
        assert current['revision'] > old['revision']
        assert current['decision']['text'] == ''
        assert current['alternatives'] == current['session_chunks'] == current['session_gestures'] == []
        assert current['session_text'] == current['last_correction'] == ''
        assert not runtime.session_signs and not runtime._session_windows and not runtime._session_conflict
        runtime._cleanup_result(old_generation, old['revision'], {'status': 'done', 'suggestion': 'old suggestion'})
        assert runtime.snapshot()['cleanup']['suggestion'] == ''
        with runtime.lock:
            runtime._publish_session(None)
        assert runtime.snapshot()['decision']['text'] == ''
        runtime.speaker.speak.assert_not_called()
    finally:
        runtime.close()


@pytest.mark.parametrize('field,value', [('session_active', True), ('session_finalizing', True), ('phase', 'recording'), ('phase', 'decoding')])
def test_new_message_rejects_active_input(field, value):
    runtime = Runtime(rehearsal=True, vocabulary={})
    try:
        runtime.action('rehearse', {'text': 'keep this', 'label': ''})
        runtime.state[field] = value
        revision = runtime.revision
        with pytest.raises(ValueError):
            runtime.action('new_message', {'revision': revision})
        assert runtime.snapshot()['decision']['text'] == 'keep this'
        assert runtime.revision == revision
    finally:
        runtime.close()


def test_new_message_rejects_speaking():
    runtime = Runtime(rehearsal=True, vocabulary={})
    try:
        runtime.speaker._process = Mock(poll=Mock(return_value=None))
        with pytest.raises(ValueError):
            runtime.action('new_message', {'revision': runtime.revision})
        runtime.speaker._process = None
    finally:
        runtime.close()

"""Loopback tests for optional paired sign video capture."""
import hashlib
import http.client
import json
import threading

import cv2
import numpy as np
import pytest

from hackathon.signs import lab
from hackathon.signs.tests.test_sequence_lab import FakeEngine, _json, _post_frame, _paired_start
from hackathon.signs.tests.test_mouth_capture import FakeTracker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / 'model.pt'
    model.write_bytes(b'capture does not load torchscript')
    def entry(label, enabled):
        return {'label': label, 'phrase': label, 'matches': [], 'enabled': enabled, 'verified': False}
    manifest = {'schema_version': 1, 'language': 'bsl', 'model_file': model.name,
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(), 'labels': ['one', 'two'],
        'input': {'frames': 16, 'size': 224, 'fps': 25, 'mean': [.5, .5, .5], 'std': [1, 1, 1]},
        'vocabulary': {'one': entry('BSL_0000', True), 'two': entry('BSL_0001', False)},
        'threshold': .8, 'margin': .2, 'source_url': '', 'license': ''}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    server = lab.create_server(engine=FakeEngine(), port=0, data_root=tmp_path / 'seq', sign_model=path)
    server.mouth._tracker_factory = FakeTracker
    monkeypatch.setattr('lipflow.dictation.clip_problem', lambda rec: None)
    monkeypatch.setattr('lipflow.dictation.rois_for', lambda rec: np.zeros((20, 96, 96), dtype=np.uint8))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1])
    yield server, conn, path, model
    conn.close(); server.shutdown(); server.server_close(); thread.join()


def start(conn, server, label='BSL_0000', **kw):
    data = {'profile': 'default', 'label': label, 'text': 'Hello, how are you?',
            'split': 'train', 'session': 'video', 'capture_video': True, **kw}
    return _json(conn, server, 'POST', '/api/paired/start', body=json.dumps(data))


def test_capture_metadata_and_alignment(setup):
    server, conn, path, model = setup
    _, state = _json(conn, server, 'GET', '/api/state', token=False)
    info = state['sign_model']
    assert info == server.sign_model_info
    assert info['sha256'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert info['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info['language'] == 'bsl' and info['status'] == 'capture_only'
    assert set(info['vocabulary']) == {'BSL_0000'}
    assert start(conn, server, video_consent=True)[0].status == 200
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :16] = (10, 30, 220); image[:, 16:] = (200, 40, 10)
    _, encoded = cv2.imencode('.jpg', image)
    expected = cv2.resize(cv2.imdecode(encoded, cv2.IMREAD_COLOR), (256, 256))
    for _ in range(3):
        assert _json(conn, server, 'POST', '/api/frame', body=encoded.tobytes(), headers={'Content-Type': 'image/jpeg'})[0].status == 200
    _, state = _json(conn, server, 'GET', '/api/state', token=False)
    assert state['paired_recording']['video_frames'] == 3
    assert 'bgr_frames' not in json.dumps(state)
    response, result = _json(conn, server, 'POST', '/api/paired/stop', body=b'{}')
    assert response.status == 200 and result['capture_video'] is True
    with np.load(result['saved']['sequence'], allow_pickle=False) as seq, np.load(result['saved']['mouth'], allow_pickle=False) as mouth:
        frames = seq['bgr_frames']
        assert frames.shape == (3, 256, 256, 3) and frames.dtype == np.uint8
        assert np.array_equal(frames[0], expected)
        assert np.array_equal(seq['timestamps_ms'], mouth['timestamps_ms'])
        assert np.array_equal(seq['timestamps_ms'], server.engine.timestamps)
        assert seq['video_consent'].dtype == np.bool_ and seq['video_consent'].item() is True
        assert seq['sign_language'].item() == 'bsl'
        assert seq['sign_model_sha256'].item() == info['sha256']
        assert seq['sign_manifest_sha256'].item() == info['manifest_sha256']
        assert seq['label'].item() == 'BSL_0000' and seq['pair_id'].item() == mouth['pair_id'].item()


def test_consent_and_labels(setup):
    server, conn, _, _ = setup
    for consent in ({}, {'video_consent': False}, {'video_consent': 1}):
        assert start(conn, server, **consent)[0].status == 400
    for label in ('BSL_0001', 'HELLO'):
        assert start(conn, server, label, video_consent=True)[0].status == 400
    for label in ('BSL_0000', 'REST', 'UNKNOWN'):
        assert start(conn, server, label, video_consent=True)[0].status == 200
        _json(conn, server, 'POST', '/api/paired/discard', body=b'{}')


def test_no_video_default_and_no_model(setup, tmp_path):
    server, conn, _, _ = setup
    assert _paired_start(conn, server)[0].status == 200
    _post_frame(conn, server)
    response, result = _json(conn, server, 'POST', '/api/paired/stop', body=b'{}')
    assert response.status == 200 and 'capture_video' not in result
    with np.load(result['saved']['sequence'], allow_pickle=False) as seq:
        assert 'bgr_frames' not in seq and 'video_consent' not in seq
    other = lab.create_server(engine=FakeEngine(), port=0, data_root=tmp_path / 'other')
    thread = threading.Thread(target=other.serve_forever, daemon=True); thread.start()
    connection = http.client.HTTPConnection('127.0.0.1', other.server_address[1])
    try:
        assert _json(connection, other, 'GET', '/api/state', token=False)[1]['sign_model'] is None
        assert start(connection, other, video_consent=True)[0].status == 400
    finally:
        connection.close(); other.shutdown(); other.server_close(); thread.join()


def test_overflow_rejects_before_mouth_save_and_discard(setup, tmp_path, monkeypatch):
    server, conn, _, _ = setup
    monkeypatch.setattr(lab, 'MAX_VIDEO_FRAMES', 2)
    assert start(conn, server, video_consent=True)[0].status == 200
    for _ in range(3):
        _, frame = _post_frame(conn, server)
    assert frame['limit_reached'] is True
    assert _json(conn, server, 'GET', '/api/state', token=False)[1]['limit_reached'] is True
    assert server.recording.video_overflow
    assert _json(conn, server, 'POST', '/api/paired/stop', body=b'{}')[0].status == 400
    assert not list(tmp_path.rglob('*.npz'))
    assert _json(conn, server, 'POST', '/api/paired/discard', body=b'{}')[0].status == 200
    assert server.recording is None and not list(tmp_path.rglob('*.npz'))


def test_bad_artifact_rejected(setup, tmp_path):
    _, _, path, model = setup
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        lab.create_server(engine=FakeEngine(), port=0, data_root=tmp_path / 'other', sign_model=path)
    model.unlink()
    with pytest.raises(ValueError, match='existing local file'):
        lab.create_server(engine=FakeEngine(), port=0, data_root=tmp_path / 'other', sign_model=path)

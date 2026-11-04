import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import hackathon.evaluate_fusion as evaluator
from hackathon.contracts import SignEvent
from hackathon.evaluate_fusion import evaluate, load_pairs, main
from hackathon.tests.test_evaluate_fusion import Motion, Reader, samples


class FakeSign:
    """Deterministic replay backend: same start/process/finish/cancel contract."""

    def __init__(self, label='WATER', accept=True, language='bsl'):
        self.label, self.accept = label, accept
        self.vocabulary = {'WATER': {'label': 'WATER', 'phrase': 'water', 'matches': ['water'],
                                     'enabled': True, 'verified': False}}
        self.model_info = {'language': language, 'sha256': 'd' * 64}
        self.last_diagnostics = {}
        self.windows = []

    def start(self, ts):
        self._start, self._last = ts, None

    def process(self, frame, ts):
        assert frame.dtype == np.uint8 and frame.shape == (256, 256, 3)
        assert ts >= self._start if self._last is None else ts > self._last  # adapter allows frame == start
        self._last = ts

    def finish(self, ts):
        assert self._last is None or ts >= self._last
        self.windows.append((self._start, ts))
        self.last_diagnostics = {'outcome': 'accepted' if self.accept else 'rejected',
                                 'reason': None if self.accept else 'below_threshold',
                                 'top_class': 'water_class', 'score': 0.9}
        return [SignEvent(self.label, 0.9, self._start, ts, True)] if self.accept else []

    def cancel(self):
        self._last = None


def sign_samples(tmp_path, **video):
    profile, seq, mouth = samples(tmp_path)
    data = dict(seq, bgr_frames=np.zeros((3, 256, 256, 3), dtype=np.uint8),
                video_consent=np.bool_(True), sign_language='bsl',
                sign_model_sha256='a' * 64, sign_manifest_sha256='b' * 64, **video)
    np.savez(profile.sequences_dir / 'test' / 'a.npz', **data)
    return profile, data, mouth


def test_sign_backend_fusion_and_mutual_exclusion(tmp_path):
    profile, _, _ = sign_samples(tmp_path)
    with pytest.raises(ValueError, match='exactly one'):
        evaluate(profile, 'test', Reader(), Motion(), sign_recognizer=FakeSign())
    with pytest.raises(ValueError, match='exactly one'):
        evaluate(profile, 'test', Reader())
    sign = FakeSign()
    report = evaluate(profile, 'test', Reader(), sign_recognizer=sign)
    row = report['pairs'][0]
    assert row['fusion']['text'] == 'water' and row['decision']['reason'] == 'agreement'
    assert row['sign_events'][0]['start_ms'] == 100 and row['sign_events'][0]['end_ms'] == 200  # original bounds
    assert sign.windows[0][0] == 0  # recognizer saw the normalized clock, not captured ms
    assert row['gesture']['raw_label'] == 'water_class' and row['gesture']['accepted_label'] == 'WATER'
    assert row['predicted_label'] == 'WATER' and report['summary']['gesture_classification']['recall'] == 1
    assert row['sign_decode_ms'] >= 0 and row['captured_sign_model_sha256'] == 'a' * 64
    # Captured model id differs from the current bundle: valid re-evaluation, reported not forbidden.
    assert report['summary']['sign']['model_drift_pairs'] == 1


def test_sign_rest_unknown_and_rejection(tmp_path):
    profile, seq, mouth = sign_samples(tmp_path)
    for pair, label in (('b', 'REST'), ('c', 'UNSEEN')):
        np.savez(profile.sequences_dir / 'test' / f'{pair}.npz',
                 **dict(seq, pair_id=pair, label=label, trial_id=f'trial-{pair}'))
        np.savez(profile.mouths_dir / 'test' / f'mouth-{pair}.npz', **dict(mouth, pair_id=pair, gesture_label=label))
    report = evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign())  # emits WATER on every window
    classification = report['summary']['gesture_classification']
    assert classification['correct'] == 1
    assert classification['rest_false_emissions'] == 1 and classification['unknown_false_emissions'] == 1
    rejected = evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign(accept=False))
    classification = rejected['summary']['gesture_classification']
    assert classification['rejected'] == 1  # only positives count; REST/UNSEEN are separate groups
    assert classification['rest_false_emissions'] == 0 and classification['unknown_false_emissions'] == 0
    assert rejected['pairs'][0]['gesture']['reject_reason'] == 'below_threshold'
    assert len(rejected['pairs']) == 3 and not rejected['pairs'][0]['sign_events']  # low-score clips still get rows


def test_sign_video_contract_validation(tmp_path):
    profile, seq, _ = sign_samples(tmp_path)
    target = profile.sequences_dir / 'test' / 'a.npz'
    cases = [(('bgr_frames',), {}, 'video consent given but bgr_frames missing'),
             ((), {'video_consent': 'true'}, 'video_consent'),  # string is not scalar bool True
             ((), {'video_consent': np.bool_(False)}, 'video_consent'),
             ((), {'sign_language': 'asl'}, 'sign language'),
             ((), {'bgr_frames': np.zeros((2, 256, 256, 3), np.uint8)}, 'count'),
             ((), {'bgr_frames': np.zeros((3, 256, 256, 3), np.float32)}, 'uint8')]
    for drop, overrides, match in cases:
        np.savez(target, **{k: v for k, v in dict(seq, **overrides).items() if k not in drop})
        with pytest.raises(ValueError, match=match):
            evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign())
    np.savez(target, **{k: v for k, v in seq.items() if k not in ('bgr_frames', 'video_consent')})
    with pytest.raises(ValueError, match='video_consent'):  # malformed capture is rejected, not dropped
        evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign())


def test_sign_cohorts_and_split_isolation(tmp_path):
    profile, seq, mouth = samples(tmp_path)  # personal paired only, no video fields
    with pytest.raises(ValueError, match='no paired sign-video captures'):
        evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign())
    np.savez(profile.sequences_dir / 'test' / 'sign.npz',
             **dict(seq, pair_id='sign-pair', video_consent=np.bool_(True), bgr_frames=np.zeros((3, 256, 256, 3), np.uint8),
                    sign_language='bsl', sign_model_sha256='a' * 64, sign_manifest_sha256='b' * 64))
    np.savez(profile.mouths_dir / 'test' / 'sign.npz', **dict(mouth, pair_id='sign-pair'))
    motion_report = evaluate(profile, 'test', Reader(), Motion())  # personal run excludes sign pairs
    assert [row['pair_id'] for row in motion_report['pairs']] == ['one']
    sign_report = evaluate(profile, 'test', Reader(), sign_recognizer=FakeSign())  # sign run excludes personal pairs
    assert [row['pair_id'] for row in sign_report['pairs']] == ['sign-pair']
    np.savez(profile.sequences_dir / 'train' / 'leak.npz', **dict(seq, pair_id='other', trial_id='other-trial'))
    with pytest.raises(ValueError, match='session shared between splits'):
        load_pairs(profile, 'test')
    (profile.sequences_dir / 'train' / 'leak.npz').unlink()
    np.savez(profile.mouths_dir / 'validation' / 'dupe.npz',
             **dict(mouth, session_id='other', trial_id='other'))
    with pytest.raises(ValueError, match='pair ids in both'):
        load_pairs(profile, 'test')


def test_nonchronological_trials_use_normalized_clock(tmp_path):
    profile, seq, mouth = sign_samples(tmp_path)
    # 'aaa' sorts before 'one' but was recorded later: replay order is nonchronological.
    np.savez(profile.sequences_dir / 'test' / 'aaa.npz',
             **dict(seq, pair_id='aaa', trial_id='aaa-trial', timestamps_ms=np.array([5000, 5040, 5200])))
    np.savez(profile.mouths_dir / 'test' / 'aaa.npz', **dict(mouth, pair_id='aaa'))
    sign = FakeSign()
    report = evaluate(profile, 'test', Reader(), sign_recognizer=sign)
    assert sign.windows == [(0, 200), (201, 301)]  # continuous normalized clock across trials
    rows = {row['pair_id']: row for row in report['pairs']}
    assert rows['aaa']['sign_events'][0]['start_ms'] == 5000  # row keeps the recorded timing
    assert rows['one']['sign_events'][0]['start_ms'] == 100


def test_cli_sign_model(tmp_path, monkeypatch):
    profile, _, _ = sign_samples(tmp_path)
    profile.save()
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'model.ts').write_bytes(b'fake weights')
    manifest_path = bundle / 'manifest.json'
    manifest_path.write_text(json.dumps({'model_file': 'model.ts', 'threshold': 0.5, 'margin': 0.1}))
    monkeypatch.setenv('QUIETBRIDGE_DATA_ROOT', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=lambda **kw: Reader()))
    monkeypatch.setitem(sys.modules, 'hackathon.signs.isolated', SimpleNamespace(IsolatedSignRecognizer=lambda path: FakeSign()))
    report = tmp_path / 'sign-report.json'
    args = ['--profile', 'alice', '--split', 'test', '--output', str(report), '--sign-model', str(manifest_path)]
    main(args)
    content = json.loads(report.read_text())
    models = content['models']
    assert models['sign_manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert models['sign_model_sha256'] == hashlib.sha256(b'fake weights').hexdigest()
    assert models['sign_threshold'] == 0.5 and models['motion_model'] is None
    assert content['summary']['fusion']['exact'] == 1 and content['summary']['sign']['windows'] == 1
    with pytest.raises(SystemExit):  # --sign-model and --motion-model are mutually exclusive
        main(['--profile', 'alice', '--split', 'test', '--output', str(tmp_path / 'x.json'),
              '--sign-model', str(manifest_path), '--motion-model', str(tmp_path / 'm.npz')])

    class Drifting(FakeSign):
        def finish(self, ts):
            manifest_path.write_text('{}')  # bundle changes mid-evaluation
            return super().finish(ts)

    monkeypatch.setitem(sys.modules, 'hackathon.signs.isolated',
                        SimpleNamespace(IsolatedSignRecognizer=lambda path: Drifting()))
    with pytest.raises(RuntimeError, match='changed during evaluation'):
        main(['--profile', 'alice', '--split', 'test', '--output', str(tmp_path / 'drift.json'),
              '--sign-model', str(manifest_path)])
    assert not (tmp_path / 'drift.json').exists()  # failures never produce a success report

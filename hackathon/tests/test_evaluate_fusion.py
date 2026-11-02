import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import hackathon.evaluate_fusion as evaluator
from hackathon.evaluate_fusion import evaluate, load_pairs, main
from hackathon.profiles import Profile


class Reader:
    def resample(self, timestamps, n_frames):
        raise AssertionError("stored model ROIs must not be resampled again")

    def encode(self, rois):
        assert len(rois) == 5  # model-rate ROIs differ from three raw capture timestamps
        return rois

    def beam_search_details(self, enc, nbest):
        assert nbest == 3
        return [{'text': 'hello', 'score': -1.0}, {'text': 'water', 'score': -2.0}]


class Motion:
    def __init__(self, label='WATER', accepted=True):
        self.label, self.accepted = label, accepted

    def diagnose(self, trial):
        return {'raw_label': self.label, 'score': 0.4, 'accepted': self.accepted,
                'd1': 0.1, 'gap': float('inf'), 'reject_reason': None}


def samples(tmp_path):
    profile = Profile('alice', 'Alice', tmp_path)
    profile.ensure_dirs()
    common = dict(pair_id='one', profile_id='alice', session_id='session', trial_id='trial',
                  split='test', timestamps_ms=np.array([100, 140, 200]))
    seq = dict(common, label='WATER', expected_text='water', landmarks=np.zeros((3, 2, 21, 3)),
               hand_present=np.ones((3, 2)), handedness=np.zeros((3, 2)))
    mouth = dict(common, trial_id='mouth-trial', text='water', gesture_label='WATER',
                 rois=np.zeros((5, 96, 96), dtype=np.uint8))
    np.savez(profile.sequences_dir / 'test' / 'a.npz', **seq)
    np.savez(profile.mouths_dir / 'test' / 'b.npz', **mouth)
    # Single-modality recordings share these directories but are not paired input.
    np.savez(profile.sequences_dir / 'test' / 'single.npz', label='WATER')
    np.savez(profile.mouths_dir / 'test' / 'single.npz', text='water')
    return profile, seq, mouth


def test_paired_comparison_and_rest(tmp_path):
    profile, _, _ = samples(tmp_path)
    report = evaluate(profile, 'test', Reader(), Motion())
    row = report['pairs'][0]
    assert report['diagnostic'] and row['lips_only']['text'] == 'hello'
    assert row['fusion']['text'] == 'water' and row['decision']['reason'] == 'agreement'
    assert row['sign_events'][0]['score'] == 0.4
    assert row['sign_events'][0]['start_ms'] == 100 and row['sign_events'][0]['end_ms'] == 200
    assert report['summary']['fusion_corrections'] == 1
    assert report['summary']['mouth_timing']['fps'] == 20
    rest = evaluate(profile, 'test', Reader(), Motion('REST'))
    assert not rest['pairs'][0]['sign_events']
    assert rest['summary']['gesture_rest'] == 1
    assert rest['pairs'][0]['fusion']['text'] == 'hello'
    rejected = evaluate(profile, 'test', Reader(), Motion(accepted=False))
    assert rejected['summary']['gesture_reject'] == 1 and not rejected['pairs'][0]['sign_events']


def test_missing_and_duplicate_pairs(tmp_path):
    profile, seq, mouth = samples(tmp_path)
    np.savez(profile.sequences_dir / 'test' / 'duplicate.npz', **seq)
    with pytest.raises(ValueError, match='duplicate'):
        load_pairs(profile, 'test')
    (profile.sequences_dir / 'test' / 'duplicate.npz').unlink()
    mouth['pair_id'] = 'other'
    np.savez(profile.mouths_dir / 'test' / 'b.npz', **mouth)
    with pytest.raises(ValueError, match='missing'):
        load_pairs(profile, 'test')


def test_cli_report_and_no_overwrite(tmp_path, monkeypatch):
    profile, _, _ = samples(tmp_path)
    profile.save()
    model = profile.profile_dir / 'models' / 'motion.npz'
    model.parent.mkdir()
    model.write_bytes(b'fake')
    report = tmp_path / 'report.json'
    monkeypatch.setenv('QUIETBRIDGE_DATA_ROOT', str(tmp_path))
    options = {}
    def reader_factory(**kwargs):
        options.update(kwargs)
        return Reader()
    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=reader_factory))
    monkeypatch.setattr(evaluator.MotionModel, 'load', lambda path: Motion())
    args = ['--profile', 'alice', '--split', 'test', '--output', str(report)]
    main(args)
    content = json.loads(report.read_text())
    assert options['beam_size'] == 4 and options['personal'] is False
    assert content['diagnostic'] and content['summary']['fusion']['exact'] == 1
    before = report.read_text()
    with pytest.raises(FileExistsError):
        main(args)
    assert report.read_text() == before

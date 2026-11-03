import hashlib
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
    assert row['gestures_only']['text'] == 'water' and row['gestures_only']['wer'] == 0
    assert report['summary']['gestures_only']['exact'] == 1
    assert row['true_label'] == row['predicted_label'] == 'WATER'
    assert report['summary']['gesture_classification']['recall'] == 1
    assert report['summary']['gesture_classification']['rest_false_emission_rate'] is None
    assert report['summary']['gesture_classification']['unknown_false_emission_rate'] is None
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


def test_classification_groups_and_emissions(tmp_path):
    profile, seq, mouth = samples(tmp_path)
    cases = [('a', 'WATER', 'WATER', True), ('b', 'WATER', 'NO', True),
             ('c', 'WATER', 'WATER', False), ('d', 'WATER', 'REST', True),
             ('e', 'REST', 'WATER', True), ('f', 'REST', 'REST', True),
             ('g', 'UNSEEN', 'NO', True), ('h', 'UNSEEN', 'NO', False)]
    (profile.sequences_dir / 'test' / 'a.npz').unlink()
    (profile.mouths_dir / 'test' / 'b.npz').unlink()
    for pair, label, prediction, accepted in cases:
        seq_copy = dict(seq, pair_id=pair, label=label, trial_id=prediction + str(accepted))
        mouth_copy = dict(mouth, pair_id=pair, gesture_label=label)
        np.savez(profile.sequences_dir / 'test' / f'{pair}.npz', **seq_copy)
        np.savez(profile.mouths_dir / 'test' / f'{pair}.npz', **mouth_copy)
    class CaseMotion:
        def diagnose(self, trial):
            key = str(trial['trial_id'].item())
            return Motion(key.removesuffix('True').removesuffix('False'), key.endswith('True')).diagnose(trial)
    report = evaluate(profile, 'test', Reader(), CaseMotion())
    assert report['summary']['gesture_classification'] == {
        'positive_count': 4, 'correct': 1, 'rejected': 2, 'wrong': 1, 'recall': .25,
        'rest_count': 2, 'rest_false_emissions': 1, 'rest_false_emission_rate': .5,
        'unknown_count': 2, 'unknown_false_emissions': 1, 'unknown_false_emission_rate': .5}
    assert report['pairs'][6]['true_label'] == 'UNSEEN'
    assert report['pairs'][7]['predicted_label'] is None


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
    assert content['models']['lip_model'] == 'stock'
    assert content['models']['motion_model_sha256'] == hashlib.sha256(b'fake').hexdigest()

    lip, lm = model.parent / 'lip.pth', model.parent / 'lm.pth'
    lip.write_bytes(b'promoted lip')
    lm.write_bytes(b'promoted lm')
    report2 = tmp_path / 'promoted.json'
    main(['--profile', 'alice', '--split', 'test', '--output', str(report2)])
    content = json.loads(report2.read_text())
    assert options['personal_vsr_path'] == str(lip) and options['personal_lm_path'] == str(lm)
    assert content['models']['lip_model_sha256'] == hashlib.sha256(b'promoted lip').hexdigest()
    assert content['models']['lip_language_model_sha256'] == hashlib.sha256(b'promoted lm').hexdigest()

    explicit_lip, explicit_lm = tmp_path / 'candidate.pth', tmp_path / 'candidate-lm.pth'
    explicit_lip.write_bytes(b'candidate')
    explicit_lm.write_bytes(b'candidate lm')
    report3 = tmp_path / 'explicit.json'
    main(['--profile', 'alice', '--split', 'test', '--output', str(report3),
          '--lip-model', str(explicit_lip), '--lip-language-model', str(explicit_lm)])
    assert options['personal_vsr_path'] == str(explicit_lip) and options['personal_lm_path'] == str(explicit_lm)
    assert json.loads(report3.read_text())['models']['lip_model_sha256'] == hashlib.sha256(b'candidate').hexdigest()
    with pytest.raises(FileNotFoundError):
        main(['--profile', 'alice', '--split', 'test', '--output', str(tmp_path / 'missing.json'),
              '--lip-model', str(tmp_path / 'missing.pth')])

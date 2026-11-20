"""evaluate_fusion reports text-only and lip-model-scored phrase offers on the same pairs.
A fake scorer: report contract only, not recognition."""
import numpy as np

from quietbridge.evaluate_fusion import evaluate
from quietbridge.profiles import Profile


class Reader:
    def encode(self, rois):
        return rois

    def beam_search_details(self, enc, nbest):
        return [{'text': 'I NEED SOME WATER NOW', 'score': -1.0}]

    def score_texts(self, enc, texts):
        table = {'I NEED SOME WATER NOW': -1.0, 'I need some water now please': -2.0,
                 'I need some waiter now': -1.6}
        return [{'normalized': table.get(t, -9.0)} for t in texts]


class TextOnlyReader(Reader):
    score_texts = None


class Motion:
    def diagnose(self, trial):
        return {'raw_label': 'REST', 'score': 0.4, 'accepted': True, 'd1': 0.1, 'gap': 1.0, 'reject_reason': None}


def make(tmp_path):
    profile = Profile('alice', 'Alice', tmp_path, lip_prompts=['I need some water now please', 'I need some waiter now'])
    profile.ensure_dirs()
    common = dict(pair_id='one', profile_id='alice', session_id='session', trial_id='trial',
                  split='test', timestamps_ms=np.array([100, 140, 200]))
    np.savez(profile.sequences_dir / 'test' / 'a.npz', **dict(common, label='REST', expected_text='I need some water now',
             landmarks=np.zeros((3, 2, 21, 3)), hand_present=np.ones((3, 2)), handedness=np.zeros((3, 2))))
    np.savez(profile.mouths_dir / 'test' / 'b.npz', **dict(common, trial_id='m', text='I need some water now', gesture_label='REST',
             rois=np.zeros((5, 96, 96), dtype=np.uint8)))
    return profile


def test_model_offers_are_reported_beside_the_text_matcher(tmp_path):
    report = evaluate(make(tmp_path), 'test', Reader(), Motion())
    section = report['summary']['phrase_suggestions']
    assert section['path'] == 'text' and section['rows_with_offer'] == 1  # text offers a near variant
    assert section['model']['path'] == 'model' and section['model']['rows_with_offer'] == 0
    row = report['pairs'][0]
    assert row['phrase_model_scores']['top_score'] == -1.0
    assert row['phrase_suggestion_model']['top1_exact'] is True


def test_reader_without_scoring_reports_the_model_part_as_not_measured(tmp_path):
    report = evaluate(make(tmp_path), 'test', TextOnlyReader(), Motion())
    assert report['summary']['phrase_suggestions']['model']['measured'] is False

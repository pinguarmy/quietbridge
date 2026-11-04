"""Offline diagnostic comparison on strictly paired profile samples; never promotes models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .contracts import LipResult, SignEvent
from .fusion import fuse
from .profiles import (ProfileManager, SPLITS, profile_language_model, profile_lip_model,
                       profile_motion_model, profile_vocabulary)
from .lips_adapter import _file_sha256
from .signs.motion import MotionModel
from lipflow.tune import compute_wer, normalize_for_wer


SIGN_MARKERS = ('video_consent', 'bgr_frames', 'sign_language')


def _samples(directory, required):
    samples = {}
    for path in sorted(directory.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if 'pair_id' not in data.files:
                continue  # Single-modality recordings share the profile directories.
            missing = required - set(data.files)
            if missing:
                raise ValueError(f'{path}: missing fields {sorted(missing)}')
            # bgr_frames is read per selected pair later; scanning every video
            # capture into memory would not scale.
            sample = {key: data[key] for key in data.files if key != 'bgr_frames'}
            sample['_is_sign_video'] = any(marker in data.files for marker in SIGN_MARKERS)
        sample['_npz_path'] = path
        pair = str(sample['pair_id'].item())
        if not pair or pair in samples:
            raise ValueError(f'{path}: empty or duplicate pair_id {pair!r}')
        samples[pair] = sample
    return samples


def _paired_metadata(directory, split):
    sessions, trials, pairs = set(), set(), set()
    for path in sorted((directory / split).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if 'pair_id' not in data.files:
                continue
            sessions.add(str(data['session_id'].item()))
            trials.add(str(data['trial_id'].item()))
            pairs.add(str(data['pair_id'].item()))
    return sessions, trials, pairs


def _check_split_isolation(profile, split):
    # Same session in sequences and mouths of one split is the paired capture
    # itself; reuse of ids across splits is what breaks test independence.
    for scope in (profile.sequences_dir, profile.mouths_dir):
        sessions, trials, pairs = _paired_metadata(scope, split)
        for other in SPLITS:
            if other == split:
                continue
            other_sessions, other_trials, other_pairs = _paired_metadata(scope, other)
            if pairs & other_pairs:
                raise ValueError(f'{scope.name}: pair ids in both {split} and {other}: {sorted(pairs & other_pairs)}')
            if sessions & other_sessions:
                raise ValueError(f'{scope.name}: recording session shared between splits {split} and {other}: '
                                 f'{sorted(sessions & other_sessions)}; record a fresh session per split')
            if trials & other_trials:
                raise ValueError(f'{scope.name}: trial ids shared between splits {split} and {other}: {sorted(trials & other_trials)}')


def load_pairs(profile, split, video=False):
    _check_split_isolation(profile, split)
    common = {'pair_id', 'profile_id', 'session_id', 'trial_id', 'split', 'timestamps_ms'}
    sequences = _samples(profile.sequences_dir / split, common | {'label', 'expected_text', 'landmarks', 'hand_present', 'handedness'})
    sign_ids = {pair for pair, sample in sequences.items() if sample.pop('_is_sign_video')}
    mouths = _samples(profile.mouths_dir / split, common | {'text', 'gesture_label', 'rois'})
    if video:
        # Standard-sign runs evaluate only the video cohort; malformed video
        # captures stay in the cohort and fail validation with an actionable
        # error instead of silently claiming no data.
        sequences = {pair: sample for pair, sample in sequences.items() if pair in sign_ids}
        if not sequences:
            raise ValueError(f'no paired sign-video captures in {profile.sequences_dir / split}; '
                             'record paired samples with video consent for --sign-model')
    else:
        # Personal-gesture runs never see standard-sign pairs (or their mouths).
        sequences = {pair: sample for pair, sample in sequences.items() if pair not in sign_ids}
        mouths = {pair: sample for pair, sample in mouths.items() if pair not in sign_ids}
    missing = sequences.keys() - mouths.keys()
    if not sequences or missing or (not video and mouths.keys() - sequences.keys()):
        raise ValueError(f'paired samples missing or empty: sequence-only={sorted(missing)}, mouth-only={sorted(mouths.keys() - sequences.keys())}')
    pairs = []
    for pair in sorted(sequences):
        seq, mouth = sequences[pair], mouths[pair]
        for field in ('profile_id', 'session_id', 'split'):
            if str(seq[field].item()) != str(mouth[field].item()):
                raise ValueError(f'{pair}: inconsistent {field}')
        if str(seq['profile_id'].item()) != profile.id or str(seq['split'].item()) != split:
            raise ValueError(f'{pair}: wrong profile or split')
        if str(seq['expected_text'].item()) != str(mouth['text'].item()) or str(seq['label'].item()) != str(mouth['gesture_label'].item()):
            raise ValueError(f'{pair}: inconsistent text or gesture label')
        pairs.append((pair, seq, mouth))
    return pairs


def _timing(times):
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (np.diff(t) < 0).any() or t[0] < 0:
        raise ValueError('timestamps_ms must be nonempty, finite, nonnegative and ordered')
    gaps = np.diff(t)
    duration = t[-1] - t[0]
    return {'start_ms': int(t[0]), 'end_ms': int(t[-1]), 'raw_frames': len(t),
            'fps': float(1000 * (len(t) - 1) / duration) if duration else None,
            'max_gap_ms': float(gaps.max()) if len(gaps) else None}


def _metric(text, reference):
    edits, words, rate = compute_wer(text, reference)
    return {'text': text, 'normalized': normalize_for_wer(text),
            'exact': normalize_for_wer(text) == normalize_for_wer(reference),
            'edits': edits, 'reference_words': words, 'wer': rate}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sign_video(seq, pair, language):
    """Validate the paired sign-video capture contract; frames stay uint8."""
    consent = seq.get('video_consent')
    if consent is None or consent.shape != () or consent.dtype != np.bool_ or not consent.item():
        raise ValueError(f'{pair}: video_consent must be a scalar boolean True')
    if 'sign_language' not in seq or str(seq['sign_language'].item()) != language:
        raise ValueError(f"{pair}: captured sign language does not match model language {language!r}")
    captured = {}
    for key in ('sign_model_sha256', 'sign_manifest_sha256'):
        value = seq.get(key)
        value = None if value is None else str(value.item())
        if value is None or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{pair}: {key} must be the captured 64 lowercase hex digest')
        captured[f'captured_{key}'] = value
    with np.load(seq['_npz_path'], allow_pickle=False) as data:
        if 'bgr_frames' not in data.files:
            raise ValueError(f'{pair}: video consent given but bgr_frames missing; re-record with video capture for --sign-model')
        frames = data['bgr_frames']
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (256, 256, 3):
        raise ValueError(f'{pair}: bgr_frames must be uint8 [N,256,256,3], got {frames.dtype} {frames.shape}')
    if frames.shape[0] != len(seq['timestamps_ms']):
        raise ValueError(f'{pair}: bgr_frames count {frames.shape[0]} != timestamps count {len(seq["timestamps_ms"])}')
    return frames, captured


def _replay_sign(recognizer, frames, times, base):
    """Replay one captured window on a normalized increasing clock; events are
    retimed to the recording so fusion uses the original capture bounds."""
    first = int(times[0])
    recognizer.start(base)
    try:
        for frame, ts in zip(frames, times):
            recognizer.process(frame, int(ts) - first + base)
        started = time.perf_counter()
        events = recognizer.finish(int(times[-1]) - first + base)
        decode_ms = (time.perf_counter() - started) * 1000
    except Exception:
        recognizer.cancel()  # reset the window so later trials still replay
        raise
    shift = first - base
    retimed = [SignEvent(e.label, e.score, e.start_ms + shift, e.end_ms + shift, e.is_final) for e in events]
    return retimed, decode_ms, int(times[-1]) - first + base + 1


def evaluate(profile, split, reader, motion=None, *, sign_recognizer=None):
    if (motion is None) == (sign_recognizer is None):
        raise ValueError('pass exactly one of motion or sign_recognizer')
    pairs = load_pairs(profile, split, video=sign_recognizer is not None)
    clock = 0
    if sign_recognizer is not None:
        # Standard signs use the recognizer's own enabled vocabulary, not the
        # profile gesture vocabulary.
        vocabulary = {label: entry for label, entry in sign_recognizer.vocabulary.items() if entry.get('enabled')}
        language = sign_recognizer.model_info['language']
    else:
        vocabulary = profile_vocabulary(profile)
    rows = []
    for pair, seq, mouth in pairs:
        st, mt = _timing(seq['timestamps_ms']), _timing(mouth['timestamps_ms'])
        if len(seq['timestamps_ms']) != len(seq['landmarks']):
            raise ValueError(f'{pair}: sequence frame/timestamp count mismatch')
        extra = {}
        if sign_recognizer is not None:
            # Validate the capture contract before the expensive lip decode.
            frames, captured = _sign_video(seq, pair, language)
            events, decode_ms, clock = _replay_sign(sign_recognizer, frames, seq['timestamps_ms'], clock)
            last = dict(sign_recognizer.last_diagnostics)
            diagnostic = {'raw_label': last.get('top_class'), 'score': last.get('score'), 'accepted': bool(events),
                          'reject_reason': None if events else last.get('reason'),
                          'accepted_label': events[0].label if events else None}
            extra = {**captured, 'sign_decode_ms': decode_ms}
        # MouthCapture stores already resampled model ROIs alongside the longer
        # or shorter raw timestamp sequence. Do not resample them a second time.
        candidates = reader.beam_search_details(reader.encode(mouth['rois']), nbest=3)
        lips = LipResult([c['text'] for c in candidates], 0, [c['score'] for c in candidates], mt['start_ms'], mt['end_ms'], mt['raw_frames'], mt['fps'], mt['max_gap_ms'])
        if sign_recognizer is None:
            diagnostic = motion.diagnose(seq)
            events = []
            if diagnostic['accepted'] and diagnostic['raw_label'] != 'REST':
                events.append(SignEvent(diagnostic['raw_label'], diagnostic['score'], st['start_ms'], st['end_ms'], True))
        start, end = min(st['start_ms'], mt['start_ms']), max(st['end_ms'], mt['end_ms'])
        decision = fuse(lips, events, vocabulary, end, start)
        reference = str(seq['expected_text'].item())
        true_label = str(seq['label'].item())
        predicted_label = (events[0].label if events else None) if sign_recognizer is not None \
            else (diagnostic['raw_label'] if diagnostic['accepted'] else None)
        gesture_decision = fuse(None, events, vocabulary, end, start)
        rows.append({'pair_id': pair, 'session_id': str(seq['session_id'].item()), 'trial_id': str(seq['trial_id'].item()),
                     'reference': reference, 'gesture_label': true_label, 'true_label': true_label,
                     'predicted_label': predicted_label, 'raw_candidates': candidates,
                     'gesture': diagnostic, 'sign_events': [asdict(e) for e in events], 'decision': asdict(decision),
                     'lips_only': _metric(candidates[0]['text'] if candidates else '', reference),
                     'gestures_only': _metric(gesture_decision.text, reference),
                     'fusion': _metric(decision.text, reference), 'sequence_timing': st, 'mouth_timing': mt, **extra})
    def aggregate(name):
        edits = sum(r[name]['edits'] for r in rows)
        words = sum(r[name]['reference_words'] for r in rows)
        return {'wer': edits / words if words else None, 'edits': edits, 'reference_words': words,
                'exact': sum(r[name]['exact'] for r in rows) / len(rows)}
    positive = [r for r in rows if r['true_label'] in vocabulary]
    rest = [r for r in rows if r['true_label'] == 'REST']
    unknown = [r for r in rows if r['true_label'] != 'REST' and r['true_label'] not in vocabulary]
    correct = sum(r['predicted_label'] == r['true_label'] for r in positive)
    rejected = sum(r['predicted_label'] is None or r['predicted_label'] == 'REST' for r in positive)
    def false_emissions(group):
        return sum(r['predicted_label'] is not None and r['predicted_label'] != 'REST' for r in group)
    classification = {
        'positive_count': len(positive), 'correct': correct, 'rejected': rejected,
        'wrong': len(positive) - correct - rejected,
        'recall': correct / len(positive) if positive else None,
        'rest_count': len(rest), 'rest_false_emissions': false_emissions(rest),
        'rest_false_emission_rate': false_emissions(rest) / len(rest) if rest else None,
        'unknown_count': len(unknown), 'unknown_false_emissions': false_emissions(unknown),
        'unknown_false_emission_rate': false_emissions(unknown) / len(unknown) if unknown else None,
    }
    summary = {'count': len(rows), 'lips_only': aggregate('lips_only'),
               'gestures_only': aggregate('gestures_only'), 'gesture_classification': classification,
               'fusion': aggregate('fusion'),
               'fusion_corrections': sum(not r['lips_only']['exact'] and r['fusion']['exact'] for r in rows),
               'fusion_regressions': sum(r['lips_only']['exact'] and not r['fusion']['exact'] for r in rows),
               'conflict': sum(r['decision']['reason'] == 'conflict' for r in rows),
               'partial_agreement': sum(r['decision']['reason'] == 'partial_agreement' for r in rows),
               'agreement': sum(r['decision']['reason'] == 'agreement' for r in rows),
               'gesture_reject': sum(not r['gesture']['accepted'] for r in rows),
               'gesture_rest': sum(r['gesture']['accepted'] and r['gesture']['raw_label'] == 'REST' for r in rows)}
    for stream in ('sequence', 'mouth'):
        timings = [r[f'{stream}_timing'] for r in rows]
        duration = sum(t['end_ms'] - t['start_ms'] for t in timings)
        gaps = [t['max_gap_ms'] for t in timings if t['max_gap_ms'] is not None]
        summary[f'{stream}_timing'] = {'raw_frames': sum(t['raw_frames'] for t in timings),
            'fps': 1000 * sum(t['raw_frames'] - 1 for t in timings) / duration if duration else None,
            'max_gap_ms': max(gaps) if gaps else None}
    limitations = ['Offline paired-sample diagnostics do not measure live camera accuracy, latency, or calibrated confidence.',
                   'Gestures-only text may contain only a slot and is compared with the full reference, so its WER and exact rate reflect that difference.',
                   'Cross-split checks cover recording session/trial/pair ids, not what models were trained on; this report does not establish test independence.',
                   'Unknown inputs are labels outside the profile vocabulary other than REST; false emissions count accepted non-REST predictions.']
    if sign_recognizer is not None:
        decodes = [r['sign_decode_ms'] for r in rows]
        current = sign_recognizer.model_info.get('sha256')
        summary['sign'] = {'windows': len(rows), 'decode_ms_mean': sum(decodes) / len(decodes),
                           'decode_ms_max': max(decodes),
                           'model_drift_pairs': sum(r['captured_sign_model_sha256'] != current for r in rows)}
        limitations.append('Isolated sign windows are replayed from captured video on a normalized clock; per-window '
                           'decode time excludes camera latency and results do not measure continuous signing.')
    return _json_safe({'diagnostic': True, 'profile_id': profile.id, 'split': split,
                       'limitations': limitations, 'summary': summary, 'pairs': rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lip-model', type=Path)
    parser.add_argument('--lip-language-model', type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument('--motion-model', type=Path)
    backend.add_argument('--sign-model', type=Path, help='isolated-sign manifest; replays paired captured video')
    args = parser.parse_args(argv)
    profile = ProfileManager().get_profile(args.profile)
    if args.output.exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    sign = motion_path = None
    if args.sign_model is not None:
        from .signs.isolated import IsolatedSignRecognizer
        sign = IsolatedSignRecognizer(args.sign_model)
        sign_manifest = json.loads(args.sign_model.read_text(encoding='utf-8'))
        checkpoint = args.sign_model.parent / sign_manifest['model_file']
        sign_hashes = (_file_sha256(args.sign_model), _file_sha256(checkpoint))
        loaded = (sign.model_info.get('manifest_sha256'), sign.model_info.get('sha256'))
        if loaded[0] is not None and loaded != sign_hashes:
            raise RuntimeError('sign model bundle changed between load and evaluation start')
    else:
        motion_path = args.motion_model or profile_motion_model(profile)
        if not motion_path.is_file():
            raise FileNotFoundError(f'Motion model not found: {motion_path}')
    lip_path = args.lip_model if args.lip_model is not None else profile_lip_model(profile)
    lm_path = args.lip_language_model if args.lip_language_model is not None else profile_language_model(profile)
    if args.lip_model is None and not lip_path.is_file():
        lip_path = None
    if args.lip_language_model is None and not lm_path.is_file():
        lm_path = None
    for path in (lip_path, lm_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f'Lip model not found: {path}')
    from lipflow.vsr import LipReader
    reader = LipReader(beam_size=4, personal=False,
                       personal_vsr_path=str(lip_path) if lip_path else None,
                       personal_lm_path=str(lm_path) if lm_path else None)
    if sign is not None:
        report = evaluate(profile, args.split, reader, sign_recognizer=sign)
        if (_file_sha256(args.sign_model), _file_sha256(checkpoint)) != sign_hashes:
            raise RuntimeError('sign model bundle changed during evaluation; refusing to write report')
    else:
        report = evaluate(profile, args.split, reader, MotionModel.load(motion_path))
    report['models'] = {'lip_model': str(lip_path) if lip_path else 'stock',
                        'lip_language_model': str(lm_path) if lm_path else 'stock',
                        'motion_model': str(motion_path) if motion_path else None, 'beam_size': 4,
                        'lip_model_sha256': _file_sha256(lip_path) if lip_path else None,
                        'lip_language_model_sha256': _file_sha256(lm_path) if lm_path else None,
                        'motion_model_sha256': _file_sha256(motion_path) if motion_path else None}
    if sign is not None:
        report['models'].update(sign_manifest=str(args.sign_model), sign_manifest_sha256=sign_hashes[0],
                                sign_model_sha256=sign_hashes[1], sign_language=sign.model_info['language'],
                                sign_threshold=sign_manifest['threshold'], sign_margin=sign_manifest['margin'])
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

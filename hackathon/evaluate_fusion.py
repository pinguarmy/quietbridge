"""Offline diagnostic comparison on strictly paired profile samples; never promotes models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .contracts import LipResult, SignEvent
from .fusion import fuse
from .profiles import ProfileManager, SPLITS, profile_motion_model, profile_vocabulary
from .signs.motion import MotionModel
from lipflow.tune import compute_wer, normalize_for_wer


def _samples(directory, required):
    samples = {}
    for path in sorted(directory.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if 'pair_id' not in data.files:
                continue  # Single-modality recordings share the profile directories.
            missing = required - set(data.files)
            if missing:
                raise ValueError(f'{path}: missing fields {sorted(missing)}')
            sample = {key: data[key] for key in data.files}
        pair = str(sample['pair_id'].item())
        if not pair or pair in samples:
            raise ValueError(f'{path}: empty or duplicate pair_id {pair!r}')
        samples[pair] = sample
    return samples


def load_pairs(profile, split):
    common = {'pair_id', 'profile_id', 'session_id', 'trial_id', 'split', 'timestamps_ms'}
    sequences = _samples(profile.sequences_dir / split, common | {'label', 'expected_text', 'landmarks', 'hand_present', 'handedness'})
    mouths = _samples(profile.mouths_dir / split, common | {'text', 'gesture_label', 'rois'})
    if not sequences or sequences.keys() != mouths.keys():
        raise ValueError(f'paired samples missing or empty: sequence-only={sorted(sequences.keys() - mouths.keys())}, mouth-only={sorted(mouths.keys() - sequences.keys())}')
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


def evaluate(profile, split, reader, motion):
    pairs = load_pairs(profile, split)
    vocabulary = profile_vocabulary(profile)
    rows = []
    for pair, seq, mouth in pairs:
        st, mt = _timing(seq['timestamps_ms']), _timing(mouth['timestamps_ms'])
        if len(seq['timestamps_ms']) != len(seq['landmarks']):
            raise ValueError(f'{pair}: sequence frame/timestamp count mismatch')
        # MouthCapture stores already resampled model ROIs alongside the longer
        # or shorter raw timestamp sequence. Do not resample them a second time.
        candidates = reader.beam_search_details(reader.encode(mouth['rois']), nbest=3)
        lips = LipResult([c['text'] for c in candidates], 0, [c['score'] for c in candidates], mt['start_ms'], mt['end_ms'], mt['raw_frames'], mt['fps'], mt['max_gap_ms'])
        diagnostic = motion.diagnose(seq)
        events = []
        if diagnostic['accepted'] and diagnostic['raw_label'] != 'REST':
            events.append(SignEvent(diagnostic['raw_label'], diagnostic['score'], st['start_ms'], st['end_ms'], True))
        start, end = min(st['start_ms'], mt['start_ms']), max(st['end_ms'], mt['end_ms'])
        decision = fuse(lips, events, vocabulary, end, start)
        reference = str(seq['expected_text'].item())
        rows.append({'pair_id': pair, 'session_id': str(seq['session_id'].item()), 'trial_id': str(seq['trial_id'].item()),
                     'reference': reference, 'gesture_label': str(seq['label'].item()), 'raw_candidates': candidates,
                     'gesture': diagnostic, 'sign_events': [asdict(e) for e in events], 'decision': asdict(decision),
                     'lips_only': _metric(candidates[0]['text'] if candidates else '', reference),
                     'fusion': _metric(decision.text, reference), 'sequence_timing': st, 'mouth_timing': mt})
    def aggregate(name):
        edits = sum(r[name]['edits'] for r in rows)
        words = sum(r[name]['reference_words'] for r in rows)
        return {'wer': edits / words if words else None, 'edits': edits, 'reference_words': words,
                'exact': sum(r[name]['exact'] for r in rows) / len(rows)}
    summary = {'count': len(rows), 'lips_only': aggregate('lips_only'), 'fusion': aggregate('fusion'),
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
    return _json_safe({'diagnostic': True, 'profile_id': profile.id, 'split': split,
                       'summary': summary, 'pairs': rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lip-model', type=Path)
    parser.add_argument('--lip-language-model', type=Path)
    parser.add_argument('--motion-model', type=Path)
    args = parser.parse_args(argv)
    profile = ProfileManager().get_profile(args.profile)
    motion_path = args.motion_model or profile_motion_model(profile)
    if not motion_path.is_file():
        raise FileNotFoundError(f'Motion model not found: {motion_path}')
    if args.output.exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    from lipflow.vsr import LipReader
    reader = LipReader(beam_size=4, personal=False,
                       personal_vsr_path=str(args.lip_model) if args.lip_model else None,
                       personal_lm_path=str(args.lip_language_model) if args.lip_language_model else None)
    report = evaluate(profile, args.split, reader, MotionModel.load(motion_path))
    report['models'] = {'lip_model': str(args.lip_model) if args.lip_model else 'stock',
                        'lip_language_model': str(args.lip_language_model) if args.lip_language_model else 'stock',
                        'motion_model': str(motion_path), 'beam_size': 4}
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

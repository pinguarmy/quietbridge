"""Real files and fake readers expose model identity drift without inference."""
import hashlib
import sys
from types import SimpleNamespace

import pytest

from hackathon.evaluate_language import compare_language_models
from hackathon.tests.test_evaluate_language import make_data


def test_visual_hash_binds_both_language_variants(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    candidate = tmp_path / 'lm.pth'
    visual = tmp_path / 'lip.pth'
    candidate.write_bytes(b'language')
    visual.write_bytes(b'visual')
    constructions = []

    class Reader:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

        def read(self, rois, fast=False):
            return 'RIGHT ANSWER', .01

    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=Reader))
    report = compare_language_models(data, candidate, visual)
    assert report['visual_model_sha256'] == hashlib.sha256(b'visual').hexdigest()
    assert all(item['personal_vsr_path'] == str(visual) for item in constructions)


def test_weight_change_during_evaluation_does_not_produce_report(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    candidate = tmp_path / 'lm.pth'
    candidate.write_bytes(b'before')

    class Reader:
        def __init__(self, **kwargs):
            pass

        def read(self, rois, fast=False):
            candidate.write_bytes(b'after')
            return 'RIGHT ANSWER', .01

    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=Reader))
    with pytest.raises(RuntimeError, match='weights changed'):
        compare_language_models(data, candidate)

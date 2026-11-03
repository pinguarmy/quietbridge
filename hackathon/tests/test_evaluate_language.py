import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hackathon.evaluate_language import compare_language_models, main


def make_data(tmp_path):
    root = tmp_path / "mouths"
    for split, session in (("train", "train-session"), ("validation", "val-session"), ("test", "test-session")):
        folder = root / split
        folder.mkdir(parents=True)
        np.savez(folder / "sample.npz", rois=np.zeros((4, 96, 96), dtype=np.uint8),
                 text=np.array("RIGHT ANSWER"), session_id=np.array(session),
                 trial_id=np.array(split), split=np.array(split))
    return root


def test_same_vsr_and_rois_compare_only_language_candidate(tmp_path, monkeypatch):
    root = make_data(tmp_path)
    candidate = tmp_path / "lm.pth"
    candidate.write_bytes(b"candidate")
    constructions = []

    class Reader:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.candidate = kwargs["personal_lm_path"] is not None

        def read(self, rois, fast=False):
            return ("RIGHT ANSWER" if self.candidate else "WRONG"), 0.01

    monkeypatch.setitem(sys.modules, "lipflow.vsr", SimpleNamespace(LipReader=Reader))
    report = compare_language_models(root, candidate)
    assert report["variants"]["stock"]["metrics"]["wer"] == 1.0
    assert report["variants"]["candidate"]["metrics"]["wer"] == 0.0
    assert report["deployment"] is False
    assert report["candidate_sha256"] is not None
    assert report["visual_model_sha256"] is None
    assert len(constructions) == 2
    assert {item["personal_vsr_path"] for item in constructions} == {None}
    assert constructions[0]["beam_size"] == constructions[1]["beam_size"] == 4


def test_language_report_never_overwrites(tmp_path, monkeypatch):
    root = make_data(tmp_path)
    candidate = tmp_path / "lm.pth"
    candidate.write_bytes(b"candidate")
    output = tmp_path / "report.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        main(["--data", str(root), "--candidate", str(candidate), "--output", str(output)])
    assert output.read_text() == "existing"

"""Local TorchScript artifact and isolated-window contract tests."""
import hashlib
import json

import numpy as np
import pytest

from hackathon.signs.isolated import IsolatedSignRecognizer


def model_files(tmp_path, logits=(0.0, 5.0, 0.0), *, language="bsl", enabled=True, sha_override=None):
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor([values], dtype=torch.float32))

        def forward(self, x):
            return self.values + x.mean() * 0

    artifact = tmp_path / "model.pt"
    traced = torch.jit.trace(Toy(logits), torch.zeros(1, 3, 16, 224, 224))
    traced.save(str(artifact))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "language": language, "model_file": "model.pt",
        "model_sha256": sha_override or digest,
        "labels": ["HELLO", "WATER", "THANKS"],
        "input": {"frames": 16, "size": 224, "fps": 25, "mean": [.5, .5, .5], "std": [1, 1, 1]},
        "vocabulary": {"WATER": {"label": "BSL_WATER", "phrase": "water", "matches": ["water"], "enabled": enabled, "verified": False}},
        "threshold": .8, "margin": .2, "source_url": "https://example.invalid/model",
        "license": "user-supplied; verify terms",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def run_window(recognizer, timestamps):
    recognizer.start(timestamps[0])
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    for ts in timestamps:
        recognizer.process(frame, ts)
    return recognizer.finish(timestamps[-1])


def test_torchscript_local_hash_bsl_event_and_model_metadata(tmp_path):
    path = model_files(tmp_path)
    recognizer = IsolatedSignRecognizer(path)
    events = run_window(recognizer, list(range(0, 1001, 25)) )
    assert len(events) == 1
    event = events[0]
    assert (event.label, event.start_ms, event.end_ms, event.is_final) == ("BSL_WATER", 0, 1000, True)
    assert event.score > .8
    assert recognizer.model_info["language"] == "bsl"
    assert recognizer.model_info["status"] == "experimental"
    assert recognizer.vocabulary["BSL_WATER"]["verified"] is False
    assert recognizer.last_diagnostics["raw_frames"] == 41
    assert recognizer.last_diagnostics["input_fps"] == pytest.approx(40)
    assert recognizer.last_diagnostics["clips"] > 1
    recognizer.close()


def test_hash_checked_before_torchscript_load(tmp_path, monkeypatch):
    path = model_files(tmp_path, sha_override="0" * 64)
    import torch
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: pytest.fail("load must not run"))
    with pytest.raises(ValueError, match="SHA-256"):
        IsolatedSignRecognizer(path)


def test_nonfinite_checkpoint_output_is_rejected(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    import torch
    recognizer._model = lambda x: torch.tensor([[float('nan'), float('inf'), 0.]])
    assert run_window(recognizer, list(range(0, 601, 25))) == []
    assert recognizer.last_diagnostics["reason"] == "nonfinite_model_output"


def test_fast_short_capture_is_not_stretched_into_a_sign(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    assert run_window(recognizer, list(range(0, 400, 20))) == []
    assert recognizer.last_diagnostics["reason"] == "insufficient_temporal_coverage"


def test_disabled_top_class_never_remaps_to_enabled_entry(tmp_path):
    path = model_files(tmp_path, enabled=False)
    recognizer = IsolatedSignRecognizer(path)
    assert run_window(recognizer, list(range(0, 601, 25))) == []
    assert recognizer.last_diagnostics["reason"] == "top_class_unmapped_or_disabled"


def test_unknown_top_head_is_silent(tmp_path):
    path = model_files(tmp_path, logits=(6.0, 0.0, 0.0))
    recognizer = IsolatedSignRecognizer(path)
    assert run_window(recognizer, list(range(0, 601, 25))) == []
    assert recognizer.last_diagnostics["reason"] == "top_class_unmapped_or_disabled"


def test_threshold_and_margin_gate_predictions(tmp_path):
    path = model_files(tmp_path, logits=(0.0, 1.0, .9))
    recognizer = IsolatedSignRecognizer(path)
    assert run_window(recognizer, list(range(0, 601, 25))) == []
    assert recognizer.last_diagnostics["reason"] in ("below_threshold", "below_margin")


def test_short_window_gap_and_bounded_resampling_diagnostics(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    assert run_window(recognizer, list(range(0, 351, 25))) == []
    assert recognizer.last_diagnostics["reason"] == "insufficient_frames"
    # A retained sparse interval is not made into synthetic motion by resampling.
    recognizer.start(400)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for ts in list(range(400, 775, 25)):
        recognizer.process(frame, ts)
    recognizer.process(frame, 1100)
    assert recognizer.finish(1100) == []
    assert recognizer.last_diagnostics["reason"] == "frame_gap_over_250ms"
    assert recognizer.last_diagnostics["raw_frames"] == 16
    assert recognizer.last_diagnostics["max_gap_ms"] == 350


def test_cancel_duplicate_timestamps_and_single_final_event(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    recognizer.start(0)
    recognizer.process(frame, 10)
    recognizer.cancel()
    assert recognizer.finish(20) == []
    assert recognizer.last_diagnostics["reason"] == "window_cancelled"
    with pytest.raises(ValueError, match="strictly increasing"):
        recognizer.process(frame, 10)
    events = run_window(recognizer, list(range(25, 626, 25)))
    assert len(events) == 1
    assert recognizer.finish(650) == []


def test_preprocess_applies_rgb_mean_and_window_covers_late_frames(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[:] = (0, 128, 255)  # BGR; RGB normalizes approximately to (.5, .002, -.5).
    tensor = recognizer._preprocess([frame] * 16)
    assert tensor.shape == (1, 3, 16, 224, 224)
    assert tensor[0, :, 0, 100, 100].tolist() == pytest.approx([.5, 128 / 255 - .5, -.5], abs=.01)
    frames = [(ts, np.full((8, 8, 3), ts % 255, np.uint8)) for ts in range(0, 20001, 100)]
    clips = recognizer._make_clips(frames)
    assert len(clips) == 16
    assert int(clips[0][0][0, 0, 0]) == 0
    assert int(clips[-1][-1][0, 0, 0]) == 20000 % 255


def test_overflow_rejects_window_and_reports_total_and_drops(tmp_path):
    recognizer = IsolatedSignRecognizer(model_files(tmp_path))
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    recognizer.start(0)
    for ts in range(1, 243):
        recognizer.process(frame, ts)
    assert recognizer.finish(242) == []
    assert recognizer.last_diagnostics["reason"] == "frame_buffer_overflow"
    assert recognizer.last_diagnostics["raw_frames"] == 242
    assert recognizer.last_diagnostics["dropped_frames"] == 2
    assert len(recognizer._frames or []) == 0


def test_manifest_rejects_preprocessing_and_path_escape(tmp_path):
    path = model_files(tmp_path)
    data = json.loads(path.read_text())
    data["model_file"] = "../model.pt"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relative filename"):
        IsolatedSignRecognizer(path)

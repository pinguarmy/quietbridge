"""Explicit per-session lip adaptation; tiny CPU tensors, no camera or VSR weights."""
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest
import torch

from hackathon.lips_adapter import LipsAdapter, load_personal_state


def tiny_model():
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.frontend = torch.nn.Linear(2, 2)
    model.encoder.embed = torch.nn.Linear(2, 2)
    model.encoder.encoders = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    model.decoder = torch.nn.Linear(2, 2)
    return model


def checkpoint(tmp_path, state):
    path = tmp_path / "pilot.pth"
    torch.save(state, path)
    return path


@pytest.fixture
def reader_modules(monkeypatch):
    constructed = []
    class Reader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = tiny_model()
            self.personal_vsr = False
            self.enc_device = torch.device("cpu")
            self.warmed_personal = None
            constructed.append(self)
        def warmup(self):
            self.warmed_personal = self.personal_vsr
    vsr = ModuleType("lipflow.vsr")
    vsr.LipReader = Reader
    face = ModuleType("lipflow.face")
    face.FaceTracker = lambda: SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "lipflow.vsr", vsr)
    monkeypatch.setitem(sys.modules, "lipflow.face", face)
    monkeypatch.delenv("QUIETBRIDGE_LIP_MODEL", raising=False)
    return constructed


def test_default_stock_is_unchanged(reader_modules):
    adapter = LipsAdapter()
    assert adapter.model_name == "stock" and adapter.model_path is None
    assert reader_modules[0].kwargs == {"beam_size": 4, "personal": False}
    assert reader_modules[0].warmed_personal is False


@pytest.mark.parametrize("prefix", ["encoder.frontend.", "encoder.embed.", "encoder.encoders.0."])
def test_load_allowed_adaptation_before_warmup(tmp_path, reader_modules, prefix):
    key = prefix + "weight"
    state = {key: torch.full((2, 2), .25)}
    path = checkpoint(tmp_path, state)
    original = path.read_bytes()
    adapter = LipsAdapter(personal_model_path=path)
    assert adapter.model_name == "personal (welcome pilot)"
    assert adapter.model_path == str(path.resolve())
    assert adapter.reader.personal_vsr is True and adapter.reader.warmed_personal is True
    assert torch.equal(adapter.reader.model.state_dict()[key], state[key])
    assert path.read_bytes() == original


def test_environment_choice_and_explicit_override(tmp_path, monkeypatch, reader_modules):
    path = checkpoint(tmp_path, {"encoder.frontend.weight": torch.ones((2, 2))})
    monkeypatch.setenv("QUIETBRIDGE_LIP_MODEL", str(path))
    assert LipsAdapter().reader.personal_vsr
    monkeypatch.setenv("QUIETBRIDGE_LIP_MODEL", str(tmp_path / "missing.pth"))
    assert LipsAdapter(personal_model_path=path).reader.personal_vsr
    with pytest.raises(FileNotFoundError):
        LipsAdapter()


@pytest.mark.parametrize("state", [
    {}, [],
    {"decoder.weight": torch.ones((2, 2))},
    {"encoder.encoders.1.weight": torch.ones((2, 2))},
    {"encoder.frontend.unknown": torch.ones((2, 2))},
    {"encoder.frontend.weight": torch.ones((3, 2))},
    {"encoder.frontend.weight": torch.ones((2, 2), dtype=torch.float64)},
    {"encoder.frontend.weight": torch.full((2, 2), float("nan"))},
    {"encoder.frontend.weight": torch.full((2, 2), float("inf"))},
    {"encoder.frontend.weight": "not a tensor"},
])
def test_invalid_checkpoint_rejected_without_patching(tmp_path, state):
    reader = SimpleNamespace(model=tiny_model(), personal_vsr=False)
    before = {key: value.clone() for key, value in reader.model.state_dict().items()}
    with pytest.raises(ValueError):
        load_personal_state(reader, checkpoint(tmp_path, state))
    assert reader.personal_vsr is False
    assert all(torch.equal(before[key], value) for key, value in reader.model.state_dict().items())


def test_load_uses_safe_torch_options(tmp_path, monkeypatch):
    path = tmp_path / "safe.pth"
    calls = []
    def load(selected, **kwargs):
        calls.append((selected, kwargs))
        return {"encoder.frontend.weight": torch.ones((2, 2))}
    monkeypatch.setattr(torch, "load", load)
    load_personal_state(SimpleNamespace(model=tiny_model()), path)
    assert calls == [(path, {"weights_only": True, "map_location": "cpu"})]


def test_timing_syncs_only_when_encoder_uses_available_mps(monkeypatch):
    from hackathon.lips_adapter import _sync_encoder
    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    _sync_encoder(SimpleNamespace(enc_device=torch.device("cpu")))
    assert calls == []
    _sync_encoder(SimpleNamespace(enc_device=torch.device("mps")))
    assert calls == ["sync"]


def test_repeated_decoder_sentence_is_not_added(monkeypatch):
    import lipflow.dictation as dictation
    monkeypatch.setattr(dictation, "clip_problem", lambda rec: None)
    monkeypatch.setattr(dictation, "rois_for", lambda rec: np.zeros((12, 96, 96), np.uint8))
    adapter = LipsAdapter.__new__(LipsAdapter)
    repeated = "I AM GOING TO HAVE A LOOK AT IT AND I AM GOING TO HAVE A LOOK AT IT"
    adapter.reader = SimpleNamespace(encode=lambda _: None,
        beam_search=lambda *args, **kwargs: [repeated], enc_device=torch.device("cpu"))
    adapter.recording = object()
    with pytest.raises(ValueError, match="Unreliable repeated decoder"):
        adapter.finish()


def test_finish_preserves_candidates_and_reports_stage_timing(monkeypatch):
    import lipflow.dictation as dictation
    rois = np.zeros((12, 96, 96), dtype=np.uint8)
    monkeypatch.setattr(dictation, "clip_problem", lambda rec: None)
    monkeypatch.setattr(dictation, "rois_for", lambda rec: rois)
    calls = []
    def encode(value):
        assert value is rois
        calls.append("encode")
        return "encoded"
    def search(encoded, nbest):
        assert encoded == "encoded" and nbest == 3
        calls.append("search")
        return ["WELCOME", "HELLO"]
    adapter = LipsAdapter.__new__(LipsAdapter)
    adapter.reader = SimpleNamespace(encode=encode, beam_search=search, enc_device=torch.device("cpu"))
    adapter.recording = object()
    result = adapter.finish()
    assert result.candidates == ["WELCOME", "HELLO"] and result.decode_ms >= 0
    assert calls == ["encode", "search"]
    assert set(adapter.last_metrics) == {"roi_ms", "encode_ms", "search_ms"}
    assert all(value >= 0 for value in adapter.last_metrics.values())
    assert adapter.recording is None
    assert adapter.finish().candidates == [] and adapter.last_metrics == {}

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from hackathon.signs import export_bsl1k
from hackathon.signs.isolated import IsolatedSignRecognizer

FAKE_I3D = '''
import torch
import torch.nn as nn

class InceptionI3d(nn.Module):
    """Tiny stand-in with the official signature and dict output."""
    def __init__(self, num_classes=400, num_in_frames=64, **kwargs):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return {"logits": self.fc(x.mean(dim=(2, 3, 4)))}
'''

GLOSSES = [f"gloss_{i:04d}" if i else "hello_world" for i in range(export_bsl1k.NUM_CLASSES)]


@pytest.fixture()
def inputs(tmp_path):
    source = tmp_path / "bsl1k"
    (source / "models").mkdir(parents=True)
    (source / "models" / "i3d.py").write_text(FAKE_I3D)
    model = nn.Linear(3, export_bsl1k.NUM_CLASSES)
    # DataParallel prefix over the fake model's 'fc' submodule
    state = {f"module.fc.{k}": v for k, v in model.state_dict().items()}
    checkpoint = tmp_path / "bsl1k.pth.tar"
    torch.save({"state_dict": state}, checkpoint)
    labels = tmp_path / "class-ordered.json"
    labels.write_text(json.dumps(GLOSSES))
    return source, checkpoint, labels


def test_export_writes_valid_bundle(inputs, tmp_path):
    source, checkpoint, labels = inputs
    out = tmp_path / "bundle"
    manifest = export_bsl1k.export_model(source, checkpoint, labels, out,
                                         ["hello_world", "gloss_0007"])
    model_bytes = (out / "model.pt").read_bytes()
    assert manifest["model_sha256"] == hashlib.sha256(model_bytes).hexdigest()
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["schema_version"] == 1 and saved["language"] == "bsl"
    assert saved["model_file"] == "model.pt"
    assert saved["input"] == {"frames": 16, "size": 224, "fps": 25,
                              "mean": [0.5, 0.5, 0.5], "std": [1, 1, 1]}
    assert saved["threshold"] == 0.8 and saved["margin"] == 0.2
    assert saved["source_url"] == "https://github.com/gulvarol/bsl1k"
    assert "not redistributed" in saved["license"]
    assert saved["labels"] == GLOSSES
    vocab = saved["vocabulary"]
    assert len(vocab) == 1064 and set(vocab) == set(GLOSSES)
    entry = vocab["hello_world"]
    assert entry["label"] == "BSL_0000" and entry["phrase"] == "hello world"
    assert entry["matches"] == ["hello world"] and entry["enabled"] is True
    assert all(e["verified"] is False for e in vocab.values())
    assert len({e["label"] for e in vocab.values()}) == 1064  # unique runtime labels
    enabled = [k for k, e in vocab.items() if e["enabled"]]
    assert enabled == ["hello_world", "gloss_0007"]
    traced = torch.jit.load(str(out / "model.pt"))
    logits = traced(torch.zeros(export_bsl1k.INPUT_SHAPE))
    assert tuple(logits.shape) == (1, 1064)  # strict load stripped 'module.' prefix


def test_exported_bundle_loads_in_recognizer(inputs, tmp_path):
    """Real integration: export a tiny bundle, construct the actual reader."""
    source, checkpoint, labels = inputs
    out = tmp_path / "bundle"
    export_bsl1k.export_model(source, checkpoint, labels, out, ["hello_world"])
    recognizer = IsolatedSignRecognizer(out / "manifest.json")
    try:
        assert recognizer.model_info["language"] == "bsl"
        assert recognizer.model_info["class_count"] == 1064
        assert recognizer.model_info["enabled_count"] == 1
        entry = recognizer.vocabulary["BSL_0000"]
        assert entry["phrase"] == "hello world" and entry["enabled"] is True
        assert entry["verified"] is False
        logits = recognizer._model(torch.zeros(export_bsl1k.INPUT_SHAPE))
        assert tuple(logits.shape) == (1, 1064)
    finally:
        recognizer.close()


def test_labels_must_have_exactly_1064_unique(inputs, tmp_path):
    source, checkpoint, labels = inputs
    labels.write_text(json.dumps(GLOSSES[:100]))
    with pytest.raises(ValueError, match="1064"):
        export_bsl1k.export_model(source, checkpoint, labels, tmp_path / "o", ["a"])
    labels.write_text(json.dumps(["dup"] * export_bsl1k.NUM_CLASSES))
    with pytest.raises(ValueError, match="unique"):
        export_bsl1k.export_model(source, checkpoint, labels, tmp_path / "o", ["dup"])


def test_enable_requires_known_gloss_and_at_least_one(inputs, tmp_path):
    source, checkpoint, labels = inputs
    with pytest.raises(ValueError, match="not in the label list"):
        export_bsl1k.export_model(source, checkpoint, labels, tmp_path / "o", ["nope"])
    with pytest.raises(ValueError, match="at least one"):
        export_bsl1k.export_model(source, checkpoint, labels, tmp_path / "o", [])


def test_refuses_existing_output_and_missing_files(inputs, tmp_path):
    source, checkpoint, labels = inputs
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        export_bsl1k.export_model(source, checkpoint, labels, out, ["hello_world"])
    assert (out / "keep.txt").read_text() == "mine"
    with pytest.raises(ValueError, match="i3d.py not found"):
        export_bsl1k.export_model(tmp_path / "nope", checkpoint, labels, tmp_path / "x",
                                  ["hello_world"])
    with pytest.raises(ValueError, match="checkpoint not found"):
        export_bsl1k.export_model(source, tmp_path / "nope.pth", labels, tmp_path / "x",
                                  ["hello_world"])


def test_cli_requires_acknowledgement(inputs, tmp_path, capsys):
    source, checkpoint, labels = inputs
    argv = ["--source", str(source), "--checkpoint", str(checkpoint),
            "--labels", str(labels), "--output", str(tmp_path / "o"),
            "--enable", "hello_world"]
    with pytest.raises(SystemExit):
        export_bsl1k.main(argv)
    assert "acknowledge" in capsys.readouterr().err
    assert export_bsl1k.main(argv + ["--acknowledge-model-terms"]) == 0

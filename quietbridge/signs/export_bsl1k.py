"""Export a user-supplied BSL-1K I3D checkpoint to a traced TorchScript bundle.

QuietBridge does not bundle or redistribute BSL-1K artifacts. The user must
supply (1) a locally installed copy of gulvarol/bsl1k, (2) a checkpoint they
trust, and (3) an externally converted class-ordered label list of exactly
1064 official BSL-1K glosses. This tool does not grant any license for those
external artifacts; verify upstream permission yourself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

NUM_CLASSES = 1064
NUM_IN_FRAMES = 16
INPUT_SHAPE = (1, 3, NUM_IN_FRAMES, 224, 224)
SOURCE_URL = "https://github.com/gulvarol/bsl1k"
LICENSE_NOTE = "External BSL-1K artifacts; verify upstream permission; not redistributed"
ACK_REMINDER = (
    "You affirm you have the right to use the supplied BSL-1K source and "
    "checkpoint. QuietBridge does not redistribute them and grants no license."
)


def _load_i3d_module(source: Path):
    path = source / "models" / "i3d.py"
    if not path.is_file():
        raise ValueError(f"models/i3d.py not found under source: {source}")
    spec = importlib.util.spec_from_file_location("_bsl1k_i3d", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "InceptionI3d"):
        raise ValueError(f"{path} does not define InceptionI3d")
    return module


def _load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))  # never pickle arbitrary data
    if not isinstance(data, list) or len(data) != NUM_CLASSES:
        raise ValueError(f"labels must be a JSON list of exactly {NUM_CLASSES} glosses")
    if any(not isinstance(g, str) or not g.strip() for g in data):
        raise ValueError("every label must be a non-empty string")
    glosses = [g.strip() for g in data]
    if len(set(glosses)) != len(glosses):
        raise ValueError("labels must be unique; duplicates would collapse the class mapping")
    return glosses


def _load_state_dict(checkpoint: Path) -> dict:
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)  # safe load only
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    # Strip the exact DataParallel 'module.' prefix; strict load rejects anything else.
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def _build_model(i3d_module):
    return i3d_module.InceptionI3d(num_classes=NUM_CLASSES, num_in_frames=NUM_IN_FRAMES)


class _LogitsWrapper:  # forward returns the logits tensor, not the dict
    @staticmethod
    def wrap(model):
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x)
                return out["logits"] if isinstance(out, dict) else out

        return Wrapper(model)


def export_model(
    source: Path,
    checkpoint: Path,
    labels: Path,
    output: Path,
    enable: list[str],
    *,
    model_factory=None,
) -> dict:
    """Trace the checkpoint and write model.pt + manifest.json into a NEW dir."""
    import torch

    source, checkpoint, labels, output = map(Path, (source, checkpoint, labels, output))
    if not enable:
        raise ValueError("at least one --enable gloss is required")
    i3d_module = _load_i3d_module(source)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint not found: {checkpoint}")
    glosses = _load_labels(labels)
    unknown = [g for g in enable if g not in glosses]
    if unknown:
        raise ValueError(f"--enable terms not in the label list: {unknown}")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")

    model = (model_factory or _build_model)(i3d_module)
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(_LogitsWrapper.wrap(model), torch.zeros(INPUT_SHAPE))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    model_bytes = buf.getvalue()

    enabled_set = set(enable)
    manifest = {
        "schema_version": 1,
        "language": "bsl",
        "model_file": "model.pt",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "labels": glosses,
        "input": {"frames": NUM_IN_FRAMES, "size": 224, "fps": 25,
                  "mean": [0.5, 0.5, 0.5], "std": [1, 1, 1]},
        "vocabulary": {
            g: {"label": f"BSL_{i:04d}", "phrase": g.replace("_", " "),
                "matches": [g.replace("_", " ")],
                "enabled": g in enabled_set, "verified": False}
            for i, g in enumerate(glosses)
        },
        "threshold": 0.8,
        "margin": 0.2,
        "source_url": SOURCE_URL,
        "license": LICENSE_NOTE,
    }
    output.mkdir(parents=True)
    # model.pt first, manifest last: a failed run leaves no final manifest.
    (output / "model.pt").write_bytes(model_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="path to user-installed gulvarol/bsl1k checkout")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to user-trusted bsl1k .pth.tar checkpoint")
    parser.add_argument("--labels", type=Path, required=True,
                        help="JSON list of exactly 1064 class-ordered glosses")
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory")
    parser.add_argument("--enable", action="append", default=[], metavar="GLOSS",
                        help="gloss to pre-enable; repeatable, at least one required")
    parser.add_argument("--acknowledge-model-terms", action="store_true",
                        help="confirm you hold rights to the external artifacts")
    args = parser.parse_args(argv)
    if not args.acknowledge_model_terms:
        parser.error(f"--acknowledge-model-terms is mandatory. {ACK_REMINDER}")
    manifest = export_model(args.source, args.checkpoint, args.labels, args.output, args.enable)
    n_enabled = sum(1 for v in manifest["vocabulary"].values() if v["enabled"])
    print(f"wrote {args.output}/model.pt + manifest.json "
          f"({NUM_CLASSES} classes, {n_enabled} enabled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

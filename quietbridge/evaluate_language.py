"""Compare a personal language candidate on identical final-test mouth ROIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lips_adapter import _file_sha256
from .signs.train_lips import evaluate, load_dataset, metrics, target_prompt


def compare_language_models(data_root, candidate_path, visual_path=None):
    """Run both LMs with the same VSR and beam4. This never promotes a model."""
    from lipflow.vsr import LipReader
    data = load_dataset(data_root)
    if not data["test"]:
        raise ValueError("A nonempty independent test split is required")
    candidate_path = Path(candidate_path).expanduser()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Language candidate not found: {candidate_path}")
    visual_path = Path(visual_path).expanduser() if visual_path else None
    if visual_path is not None and not visual_path.is_file():
        raise FileNotFoundError(f"Visual model not found: {visual_path}")
    candidate_hash = _file_sha256(candidate_path)
    visual_hash = _file_sha256(visual_path)
    target = target_prompt(data["train"])
    readers = {}
    for name, language_path in (("stock", None), ("candidate", candidate_path)):
        readers[name] = LipReader(
            beam_size=4, personal=False,
            personal_vsr_path=str(visual_path) if visual_path else None,
            personal_lm_path=str(language_path) if language_path else None,
        )
    report = {"diagnostic": True, "beam": 4, "data": str(Path(data_root).expanduser()),
              "candidate": str(candidate_path),
              "candidate_sha256": candidate_hash,
              "visual_model": str(visual_path) if visual_path else "stock",
              "visual_model_sha256": visual_hash, "variants": {}}
    for name, reader in readers.items():
        rows = evaluate(reader, data["test"])
        report["variants"][name] = {"metrics": metrics(rows, target), "transcripts": rows}
    if candidate_hash != _file_sha256(candidate_path) or visual_hash != _file_sha256(visual_path):
        raise RuntimeError("Model weights changed during recognition evaluation; no report was saved")
    report["deployment"] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--visual-model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    report = compare_language_models(args.data, args.candidate, args.visual_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "deployment": False,
                      "metrics": {name: value["metrics"] for name, value in report["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()

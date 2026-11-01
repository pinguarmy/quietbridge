"""Explicit, version-pinned setup for the local MediaPipe gesture model."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"


def default_model_path() -> Path:
    """Keep weights outside Git; allow the demo host to choose its own cache."""
    configured = os.environ.get("QUIETBRIDGE_GESTURE_MODEL")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "quietbridge" / "gesture_recognizer.task"


def download_model(path: Path | None = None) -> Path:
    """Download only when explicitly requested, and verify the pinned artifact."""
    path = default_model_path() if path is None else Path(path).expanduser()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError(f"Existing model checksum differs; not overwriting: {path}")
        return path
    with urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Downloaded gesture model checksum does not match pinned version")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing a model another setup process just wrote.
    with path.open("xb") as model:
        model.write(data)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly download weights")
    parser.add_argument("--model", type=Path, help="Override model path")
    args = parser.parse_args()
    path = args.model or default_model_path()
    if args.download:
        path = download_model(path)
    elif not path.is_file():
        parser.error(f"Model missing: {path}; run with --download before the demo")
    print(path)
    print(f"sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

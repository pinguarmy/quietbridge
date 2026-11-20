"""Explicit, version-pinned setup for the local MediaPipe gesture model.

The download itself is quietbridge.assets (the same verified, resumable
downloader as the lip-reading files and the page's Setup panel).
"""

import argparse
import hashlib
import os
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
MODEL_SIZE = 8373440  # bytes; checked with the hash before the file is installed


def default_model_path() -> Path:
    """Keep weights outside Git; QUIETBRIDGE_GESTURE_MODEL can point to another local copy."""
    configured = os.environ.get("QUIETBRIDGE_GESTURE_MODEL")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "quietbridge" / "gesture_recognizer.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for block in iter(lambda: model.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(path: Path | None = None, *, progress=None, cancel=None) -> Path:
    """Download only when explicitly requested, and verify the pinned artifact.

    An existing file is never overwritten; one with another checksum is refused.
    """
    from quietbridge import assets
    path = default_model_path() if path is None else Path(path).expanduser()
    if path.exists():
        if _sha256(path) != MODEL_SHA256:
            raise ValueError(f"Existing model checksum differs; not overwriting: {path}")
        return path
    asset = assets.Asset("gesture_recognizer.task", MODEL_URL, MODEL_SIZE, MODEL_SHA256, 1, "gesture")
    return assets.fetch(asset, path=path, progress=progress, cancel=cancel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="Explicitly download weights")
    parser.add_argument("--model", type=Path, help="Override model path")
    args = parser.parse_args()
    path = args.model or default_model_path()
    if args.download:
        from quietbridge import assets
        print(assets.GESTURES_LICENSE, flush=True)
        try:
            path = download_model(path)
        except ValueError as exc:  # DownloadError is a ValueError with an actionable message
            raise SystemExit(f"Gesture model download failed: {exc}") from exc
    elif not path.is_file():
        parser.error(f"Model missing: {path}; run with --download first")
    print(path)
    print(f"sha256: {_sha256(path)}")


if __name__ == "__main__":
    main()

"""Download only the existing local vision assets; never installs or replaces an app.

Files go to models/ beside the lipflow package, the only place lipflow.vsr and
lipflow.face load them from.
"""
import argparse
import importlib.util
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "vsr/model.json": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
    "vsr/model.pth": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
    "lm/model.json": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
    "lm/model.pth": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def _declared_total(response, offset, resume):
    """Full file size the server committed to, or None if it did not say."""
    if resume:
        match = re.fullmatch(r"bytes \d+-\d+/(\d+)", response.headers.get("Content-Range", "").strip())
        if match:
            return int(match.group(1))
    length = response.headers.get("Content-Length", "").strip()
    if length.isdigit():
        return (offset if resume else 0) + int(length)
    return None


def download(url, path, *, name=None, _restarted=False):
    """Fetch url to path through a resumable .part file; install only a complete file.

    urllib does not raise when a connection closes before Content-Length bytes
    arrive, so the byte count is checked before the .part file is renamed.
    """
    name = name or path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset and not _restarted:
            # The partial file cannot be continued (e.g. it is already complete
            # but unverified, or the upstream file changed): start again.
            part.unlink()
            return download(url, path, name=name, _restarted=True)
        raise
    with response:
        status = getattr(response, "status", 200)
        resume = bool(offset) and status == 206 and \
            response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        if status == 206 and not resume:
            # A partial body for another range must never become the whole file.
            if _restarted:
                raise ValueError(f"{name}: server returned an unexpected byte range")
            part.unlink(missing_ok=True)
            return download(url, path, name=name, _restarted=True)
        expected = _declared_total(response, offset, resume)
        count = offset if resume else 0
        report_at = count + 64 * 1024 * 1024
        with part.open("ab" if resume else "wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                count += len(chunk)
                if count >= report_at:
                    print(f"  {name}: {count // (1024 * 1024)} MiB", flush=True)
                    report_at += 64 * 1024 * 1024
    if expected is not None and count != expected:
        # Keep the .part file: rerunning resumes from the bytes received.
        raise ValueError(f"{name}: incomplete download ({count} of {expected} bytes); rerun to resume")
    if count == 0:
        raise ValueError(f"{name}: server returned an empty file")
    if expected is None:
        print(f"  {name}: server did not declare a size; completeness not verified", flush=True)
    part.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing assets (~1.2 GB); research-use weights, see NOTICE")
    args = parser.parse_args()
    missing = []
    missing_modules = []
    for name, url in ASSETS.items():
        path = ROOT / "models" / name
        if args.download and not (path.is_file() and path.stat().st_size):
            print(f"Downloading {name}…", flush=True)
            try:
                download(url, path, name=name)
            except (OSError, ValueError) as exc:  # URLError/HTTPError are OSErrors
                print(f"  {name}: download failed: {exc}", flush=True)
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK     ' if present else 'MISSING'} models/{name}", flush=True)
        if not present:
            missing.append(name)
    for module in ["cv2", "numpy", "torch", "mediapipe"]:
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK     ' if present else 'MISSING'} Python module {module}")
        if not present:
            missing_modules.append(module)
    print("Sign recognizer:", "present" if (ROOT / "hackathon/signs/recognizer.py").is_file() else "missing")
    print("No inference, camera access or accuracy test was performed by this check.")
    if missing or missing_modules:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

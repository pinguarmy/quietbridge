"""Download only the existing local vision assets; never installs or replaces an app."""
import argparse
import importlib.util
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "vsr/model.json": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
    "vsr/model.pth": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
    "lm/model.json": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
    "lm/model.pth": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing assets (~1.2 GB); research-use weights, see NOTICE")
    args = parser.parse_args()
    missing = []
    missing_modules = []
    for name, url in ASSETS.items():
        path = ROOT / "models" / name
        if args.download and not (path.is_file() and path.stat().st_size):
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {name}…", flush=True)
            offset = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                count = offset if resume else 0
                report_at = count + 64 * 1024 * 1024
                with part.open("ab" if resume else "wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        count += len(chunk)
                        if count >= report_at:
                            print(f"  {name}: {count // (1024 * 1024)} MiB", flush=True)
                            report_at += 64 * 1024 * 1024
            part.replace(path)
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK     ' if present else 'MISSING'} models/{name}", flush=True)
        if not present:
            missing.append(name)
    for module in ["cv2", "numpy", "torch", "mediapipe"]:
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK     ' if present else 'MISSING'} Python module {module}")
        if not present:
            missing_modules.append(module)
    print("Sign recognizer:", "present" if (ROOT / "hackathon/signs/recognizer.py").is_file() else "awaiting Peter's integration")
    print("No inference, camera access or accuracy test was performed by this check.")
    if missing or missing_modules:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

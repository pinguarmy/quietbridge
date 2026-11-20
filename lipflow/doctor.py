"""`lipflow doctor`: check what Lipflow and QuietBridge need, and say plainly what was not checked.

Every line starts with one status word:

  PASS  checked, and fine
  FAIL  checked, and broken or missing; any FAIL makes the exit code 1
  WARN  checked; an optional part is missing or degraded (the exit code stays 0)
  N/A   does not apply on this platform
  SKIP  not checked, with the reason
  INFO  a fact about this machine, not a check

Nothing here runs a model, records a clip or plays speech: a PASS for a model file means the file
is present (and the expected size or checksum), not that recognition works for you.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

from .paths import MODELS, WHO

# (path under models/, minimum bytes, required): sizes catch truncated downloads.
MODEL_FILES = [
    ("vsr/model.json", 1, True),
    ("vsr/model.pth", 900e6, True),
    ("lm/model.json", 1, True),
    ("lm/model.pth", 200e6, True),
    ("face_landmarker.task", 3e6, True),
    # Only training needs the tokenizer (lipflow train-lm, QuietBridge language training).
    ("lm/unigram5000.model", 100e3, False),
]


class Report:
    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "N/A": 0, "SKIP": 0, "INFO": 0}
        self.failures: list[str] = []

    def section(self, title: str):
        print(f"\n{title}", file=self.out)

    def add(self, status: str, what: str, fix: str = ""):
        self.counts[status] += 1
        if status == "FAIL":
            self.failures.append(what)
        print(f"  {status:<4}  {what}", file=self.out)
        if fix and status in ("FAIL", "WARN", "SKIP"):
            print(f"        fix: {fix}", file=self.out)

    def check(self, required: bool, good: bool, what: str, fix: str = ""):
        self.add("PASS" if good else ("FAIL" if required else "WARN"), what, fix)

    def run(self, title: str, fn, *args):
        """Run one group of checks; a crash inside it is reported as a failure, not a traceback."""
        try:
            fn(self, *args)
        except Exception as e:  # noqa: BLE001  (the doctor must always reach its summary)
            self.add("FAIL", f"{title}: the check itself crashed ({e.__class__.__name__}: {e})",
                     "report this with the output above")


def _setup_hint(rel: str = "") -> str:
    script = r"run .\setup.ps1" if sys.platform == "win32" else "run ./setup.sh"
    # quietbridge.assets (also the Setup panel in the QuietBridge page) fetches every file here.
    return script + " (or: uv run python -m quietbridge.assets --download)"


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def check_models(rep: Report):
    rep.add("INFO", f"model folder: {MODELS}")
    for rel, minimum, required in MODEL_FILES:
        path = os.path.join(MODELS, rel)
        what = f"models/{rel}" + ("" if required else " (optional: needed only for training)")
        if not os.path.isfile(path):
            rep.check(required, False, f"{what}: missing", _setup_hint(rel))
            continue
        size = os.path.getsize(path)
        if size < minimum:
            rep.check(required, False, f"{what}: only {_mb(size)}, expected at least {_mb(minimum)} "
                                       "(an interrupted download?)", f"delete it and {_setup_hint(rel)}")
        else:
            rep.check(required, True, f"{what} ({_mb(size)}; presence and size only)")


def check_runtime(rep: Report):
    for module, required, why in [("torch", True, "lip-reading model"),
                                  ("cv2", True, "camera and video"),
                                  ("mediapipe", True, "face and hand landmarks"),
                                  ("sentencepiece", False, "training the language model")]:
        try:
            mod = __import__(module)
        except Exception as e:  # noqa: BLE001  (a broken native wheel raises more than ImportError)
            rep.check(required, False, f"Python module {module} ({why}) does not import: {e}",
                      "run `uv sync`")
            continue
        rep.add("PASS", f"Python module {module} {getattr(mod, '__version__', '')}".rstrip()
                + f" imports ({why})")
    import torch
    dev = "cuda (NVIDIA GPU)" if torch.cuda.is_available() else \
        "mps (Apple GPU)" if torch.backends.mps.is_available() else "cpu"
    rep.add("INFO", f"lip-reading encoder would run on: {dev}")


def check_camera_device(rep: Report, enabled: bool):
    if not enabled:
        rep.add("SKIP", "camera: not opened (--no-camera)")
        return
    if sys.platform == "darwin":
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        names = {0: "not asked yet (you'll be prompted on first use)", 1: "restricted", 2: "denied", 3: "granted"}
        rep.check(True, status in (0, 3), f"camera permission: {names.get(status, status)} "
                                          "(permission only; no frame was read)",
                  f"System Settings → Privacy & Security → Camera → enable {WHO}")
        return
    import cv2
    if sys.platform == "win32":
        from .camera import resolve_camera
        from .dictation import load_settings
        idx = resolve_camera(load_settings().get("camera", "auto"))
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        try:
            ok = cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
        rep.check(True, ok, f"camera {idx} opens and sends a frame",
                  "Settings → Privacy & security → Camera → turn on \"Let desktop apps access your camera\", "
                  "and close other apps using the camera")
        return
    # Linux and other Unix: QuietBridge typing works without a camera, so a missing one is a warning.
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        rep.add("WARN", "camera: no video device found (/dev/video* is absent); typing works without one, "
                        "lip reading and gestures need one",
                "connect a camera, or check that this machine/container exposes /dev/video*")
        return
    cap = cv2.VideoCapture(devices[0], cv2.CAP_V4L2)
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    rep.check(False, ok, f"camera {devices[0]} opens and sends a frame"
                         + (f" ({len(devices)} video devices found)" if len(devices) > 1 else ""),
              "close other apps using the camera; your user may need to be in the 'video' group")


def check_dictation_app(rep: Report):
    """The original Lipflow menu-bar / tray app: global hotkey and pasting."""
    if sys.platform == "darwin":
        try:
            import Quartz
        except ImportError as e:
            rep.add("FAIL", f"macOS frameworks (pyobjc) do not import: {e}", "run `uv sync`")
            return
        rep.check(True, Quartz.CGPreflightListenEventAccess(), "Input Monitoring (for the push-to-talk key)",
                  f"System Settings → Privacy & Security → Input Monitoring → enable {WHO}, then restart it")
        rep.check(True, Quartz.CGPreflightPostEventAccess(), "Accessibility (to paste at your cursor)",
                  f"System Settings → Privacy & Security → Accessibility → enable {WHO}")
    elif sys.platform == "win32":
        try:
            from pynput import keyboard  # noqa: F401
        except Exception as e:  # noqa: BLE001
            rep.add("FAIL", f"pynput keyboard hook does not load: {e}", "run `uv sync`")
        else:
            rep.add("PASS", "pynput keyboard hook loads (Windows needs no permission for it; "
                            "the hook was not installed)")
    else:
        rep.add("N/A", f"Lipflow menu-bar / tray dictation app: macOS and Windows only (this is {sys.platform}); "
                       "use QuietBridge (`uv run quietbridge`) or `lipflow file VIDEO` here")
        return
    from .cleanup import Cleaner
    c = Cleaner()
    rep.add("INFO", f"text-cleanup backend: {c.describe()}"
            + ("  (set ANTHROPIC_API_KEY or run Ollama for much better accuracy)" if c.backend == "basic" else ""))


def check_quietbridge(rep: Report):
    """Prerequisites the QuietBridge app (quietbridge/) points its users at this command for."""
    try:
        from quietbridge.signs.assets import MODEL_SHA256, default_model_path
    except Exception as e:  # noqa: BLE001
        rep.add("SKIP", f"stock gesture model: could not import quietbridge.signs.assets ({e})")
    else:
        path = default_model_path()
        download = "uv run python -m quietbridge.signs.assets --download"
        if not path.is_file():
            rep.add("WARN", f"stock gesture model: missing at {path}; typing and lip reading work without it",
                    download)
        else:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            rep.check(True, h.hexdigest() == MODEL_SHA256,
                      f"stock gesture model {path}: " + ("pinned SHA-256 matches" if h.hexdigest() == MODEL_SHA256
                                                         else "SHA-256 differs from the pinned version"),
                      f"move the file away, then {download}")
    try:
        from quietbridge.speech import Speaker
        speaker = Speaker()
        backend = speaker.backend
    except Exception as e:  # noqa: BLE001
        rep.add("SKIP", f"local speech: could not query quietbridge.speech.Speaker ({e})")
        return
    if backend:
        exe = getattr(speaker, "_resolve", lambda: (backend, None))()[1]
        rep.add("PASS", f"local speech backend: {backend}" + (f" ({exe})" if exe else "")
                + "; found on PATH, no sound was played")
    else:
        hint = getattr(speaker, "_unavailable_message", lambda: "no local speech command found")()
        rep.add("WARN", "local speech: no backend found; text still shows, but Speak won't work",
                hint)


def doctor(check_camera: bool = True) -> int:
    rep = Report()
    print("Lipflow / QuietBridge doctor")
    print("(PASS = checked and fine, FAIL = required and broken, WARN = optional, N/A = not on this "
          "platform, SKIP = not checked)")
    rep.section("Model files")
    rep.run("model files", check_models)
    rep.section("Python runtime")
    rep.run("Python runtime", check_runtime)
    rep.section("Camera")
    rep.run("camera", check_camera_device, check_camera)
    rep.section("Lipflow dictation app")
    rep.run("dictation app", check_dictation_app)
    rep.section("QuietBridge")
    rep.run("QuietBridge", check_quietbridge)
    c = rep.counts
    print(f"\nSummary: {c['PASS']} passed, {c['FAIL']} failed, {c['WARN']} warnings, "
          f"{c['N/A']} not applicable, {c['SKIP']} not checked.")
    if rep.failures:
        print("Required checks failed: " + "; ".join(rep.failures))
        return 1
    print("No required check failed." + (" Review the warnings above." if c["WARN"] else ""))
    return 0

"""lipflow [run] | lipflow file VIDEO | lipflow doctor"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# OpenCV would ask for camera access from its capture thread, which silently fails; the app
# asks on the main thread instead (app.request_camera).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# Push-to-talk key names, listed here so that parsing arguments never imports a platform hotkey
# module (macOS Quartz in hotkey.py, pynput in win/hotkey.py). Only `run` and `onboard` need one.
# tests/test_cli.py checks these match the KEYS of both hotkey modules.
MAC_KEYS = ("right_option", "left_option", "right_command", "right_control", "fn")
MAC_DEFAULT_KEY = "right_option"
WIN_KEYS = ("right_control", "right_alt", "left_alt", "right_shift")
WIN_DEFAULT_KEY = "right_control"

APP_UNAVAILABLE = ("the Lipflow dictation app (menu bar / system tray, global push-to-talk key) needs "
                   "macOS or Windows; this is {platform}. Here you can use QuietBridge (`uv run quietbridge`), "
                   "`lipflow file VIDEO` and `lipflow doctor`.")


def _app():
    """(Options, run) for this OS's front end: menu bar on macOS, system tray on Windows.

    Raises SystemExit with a one-line message where the app cannot run."""
    if sys.platform not in ("darwin", "win32"):
        raise SystemExit("lipflow: " + APP_UNAVAILABLE.format(platform=sys.platform))
    try:
        if sys.platform == "win32":
            from .win.app import Options, run
        else:
            from .app import Options, run
    except ImportError as e:  # e.g. pyobjc / pynput / pystray missing from a partial install
        raise SystemExit(f"lipflow: the dictation app could not load ({e}). Run `uv sync`, then "
                         "`uv run lipflow doctor`.") from e
    return Options, run


def _missing_models(*rels: str) -> list[str]:
    from .paths import MODELS
    return [rel for rel in rels if not os.path.isfile(os.path.join(MODELS, rel))]


def _setup_hint() -> str:
    return r".\setup.ps1" if sys.platform == "win32" else "./setup.sh"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):  # ✓ and → in a cp1252 console or a pipe
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    # customize/tune have their own argparse and need no camera, hotkey or UI: dispatch
    # before anything heavy is imported so they also work on a bare install.
    if argv and argv[0] == "customize":
        from .customize import customize_handler
        sys.exit(customize_handler(argv[1:]))
    if argv and argv[0] == "tune":
        from .tune import tune_handler
        sys.exit(tune_handler(argv[1:]))

    keys, default_key = (WIN_KEYS, WIN_DEFAULT_KEY) if sys.platform == "win32" else (MAC_KEYS, MAC_DEFAULT_KEY)

    p = argparse.ArgumentParser(prog="lipflow", description="Silent dictation by lip reading.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start the dictation app in the menu bar / system tray "
                                   "(default; macOS and Windows only)")
    r.add_argument("--key", default=default_key, choices=list(keys), help="push-to-talk key")
    r.add_argument("--beam", type=int, default=4, help="beam size (higher = slower, about the same accuracy)")
    r.add_argument("--cleanup", default="auto", choices=["auto", "claude", "gemini", "local", "ollama", "basic"],
                   help="text-cleanup backend ('auto' = the saved setting, else first available)")
    r.add_argument("--cleanup-style", default=None, choices=["faithful", "polish"],
                   help="faithful: fix mis-read words only; polish: also drop fillers and tidy phrasing "
                        "(default: the saved setting, else faithful)")
    r.add_argument("--camera", default="auto",
                   help="'auto' (the built-in camera), a camera number, part of a camera's name (Mac), "
                        "or a video file")
    r.add_argument("--copy-only", action="store_true", help="copy to the clipboard instead of pasting")
    r.add_argument("--no-preview", action="store_true", help="don't show live words while you talk")

    f = sub.add_parser("file", help="lip-read a video file")
    f.add_argument("video")
    f.add_argument("--start", type=float, default=0.0)
    f.add_argument("--end", type=float, default=None)
    f.add_argument("--beam", type=int, default=10)
    f.add_argument("--cleanup", default="auto",
                   choices=["auto", "claude", "gemini", "local", "ollama", "basic", "none"])
    f.add_argument("--cleanup-style", default=None, choices=["faithful", "polish"],
                   help="faithful: fix mis-read words only; polish: also drop fillers and tidy phrasing")

    d = sub.add_parser("doctor", help="check models, runtime, camera, permissions and QuietBridge prerequisites")
    d.add_argument("--no-camera", action="store_true",
                   help="don't open the camera (reported as not checked); for CI and headless machines")
    onboarding = sub.add_parser("onboard", help="open personal recording and training (macOS and Windows only)")
    onboarding.add_argument("--start", choices=["welcome", "practice"], default="welcome",
                            help="practice: go straight to recording if permissions are already granted")
    sub.add_parser("train-lm", help="fine-tune the language model on your imported phrases")
    sub.add_parser("customize", help="import words/phrases and set cleanup options (own --help)")
    sub.add_parser("tune", help="replay clips against cleanup backends and score them (own --help)")
    w = sub.add_parser("import-wispr", help="learn your phrasing from your Wispr Flow history (stays local)")
    w.add_argument("--from-text", help="import a plain-text file of your writing instead (one phrase per line)")

    if not argv or argv[0] not in {"run", "file", "doctor", "import-wispr", "onboard", "train-lm", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    cmd = args.cmd

    if cmd == "file":
        if "://" not in args.video and not os.path.isfile(args.video):  # OpenCV also reads stream URLs
            sys.exit(f"lipflow file: no such video file: {args.video}")
        missing = _missing_models("vsr/model.json", "vsr/model.pth", "lm/model.json", "lm/model.pth",
                                  "face_landmarker.task")
        if missing:
            sys.exit(f"lipflow file: missing model files in models/: {', '.join(missing)}. "
                     f"Run {_setup_hint()} to download them.")
        from .offline import ClipError, transcribe_file
        from .vsr import LipReader
        try:
            raw = transcribe_file(args.video, LipReader(beam_size=args.beam), args.start, args.end)
        except ClipError as e:
            sys.exit(f"lipflow file: {e}")
        print("raw:  ", raw)
        if args.cleanup != "none":
            from .cleanup import Cleaner
            c = Cleaner(args.cleanup, style=args.cleanup_style)
            print(f"text:  {c([raw])}   [{c.describe()} · {c.style}]")
    elif cmd == "import-wispr":
        from .personal import PHRASES, import_wispr, save_phrases
        from .vocab import PATH as WORDS
        try:
            if args.from_text:
                with open(args.from_text, encoding="utf-8") as src:
                    stats = save_phrases(src.read().splitlines()) | {"source": args.from_text}
            else:
                stats = import_wispr()
        except (OSError, RuntimeError) as e:  # no file / no Wispr database / no text column
            sys.exit(f"lipflow import-wispr: {e}")
        print(f"Imported {stats['phrases']:,} phrases ({stats['words']:,} words) from {stats['source']}")
        print(f"  saved to {PHRASES}")
        if stats["new_names"]:
            print(f"  added {len(stats['new_names'])} names/terms to {WORDS}. Review them: "
                  "Lipflow menu → Edit custom words")
        print("Restart Lipflow to use them.")
    elif cmd == "train-lm":
        from .train_lm import train
        try:
            r = train()
        except (FileNotFoundError, ValueError) as e:  # no phrases, no valid sentences, missing model files
            sys.exit(f"lipflow train-lm: {e}")
        print(f"Your held-out phrases: perplexity {r['before']['yours']:.1f} → {r['after']['yours']:.1f}; "
              f"general text {r['before']['general']:.1f} → {r['after']['general'] or r['before']['general']:.1f}"
              f" ({'saved' if r['saved'] else 'not better, not saved'})")
    elif cmd == "onboard":
        Options, run = _app()
        run(Options(onboard=True, onboarding_start=args.start))
    elif cmd == "doctor":
        from .doctor import doctor
        sys.exit(doctor(check_camera=not args.no_camera))
    else:
        Options, run = _app()
        camera = int(args.camera) if args.camera.isdigit() else args.camera
        run(Options(key=args.key, beam=args.beam, backend=args.cleanup, style=args.cleanup_style,
                    camera=camera, paste=not args.copy_only, live_preview=not args.no_preview))


if __name__ == "__main__":
    main()

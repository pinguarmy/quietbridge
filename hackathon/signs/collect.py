"""Collect consented personal gesture feature trials from a local camera."""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import numpy as np

LABELS = ("THUMB_UP", "THUMB_DOWN", "OPEN_PALM", "POINTING_UP", "NEUTRAL")
MIN_POSITIVE_FRAMES = 8
MAX_FEATURE_FRAMES = 30
DEFAULT_OUTPUT = Path.home() / ".local" / "share" / "quietbridge" / "gestures"


def _check_session_split(output, split, session):
    other_split = "test" if split == "train" else "train"
    other_dir = Path(output) / other_split
    if other_dir.exists() and any(other_dir.glob(f"{session}_*.npz")):
        raise ValueError(f"session {session!r} already has trials in {other_split}; refusing split leakage")


def save_trial(output, split, session, label, features):
    """Save a single immutable feature-only trial and return its path."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if label not in LABELS:
        raise ValueError(f"unsupported label: {label}")
    if not session or Path(session).name != session or session in (".", ".."):
        raise ValueError("session must be a non-empty filename-safe identifier")

    output = Path(output).expanduser()
    _check_session_split(output, split, session)

    array = np.asarray(features, dtype=np.float32)
    if array.size == 0:
        array = np.empty((0, 63), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 63:
        raise ValueError("features must have shape (N, 63)")
    array = array[:MAX_FEATURE_FRAMES]
    if label != "NEUTRAL" and len(array) < MIN_POSITIVE_FRAMES:
        raise ValueError(f"positive trial needs at least {MIN_POSITIVE_FRAMES} single-hand frames")

    trial_id = str(uuid.uuid4())
    destination_dir = output / split
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{session}_{trial_id}.npz"
    # Exclusive creation preserves old records even in the unlikely event of a UUID collision.
    with destination.open("xb") as archive:
        np.savez_compressed(
            archive,
            features=array,
            label=np.asarray(label),
            session_id=np.asarray(session),
            trial_id=np.asarray(trial_id),
        )
    return destination


def _capture(args):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    from .assets import default_model_path
    from .features import hand_features

    model_path = default_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Gesture model missing: {model_path}; download it explicitly before collecting")

    camera = cv2.VideoCapture(args.camera)
    recognizer = None
    window = "QuietBridge gesture collection"
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}")
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        recognizer = vision.GestureRecognizer.create_from_options(options)
        cv2.namedWindow(window)
        print("Camera is active only while this command runs. Press Space to start a trial; Q quits.")
        trial_number = 0
        quit_requested = False
        while trial_number < args.trials and not quit_requested:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped providing frames")
            preview = cv2.flip(frame, 1)
            cv2.putText(preview, f"{args.label}: Space starts trial {trial_number + 1}/{args.trials}; Q quits",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            print("Get ready; keep the selected pose steady.")
            phase_start = time.perf_counter()
            aborted = False
            while time.perf_counter() - phase_start < 1.0:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped providing frames")
                preview = cv2.flip(frame, 1)
                cv2.putText(preview, "Starting in 1 second... (Q quits)", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
            if aborted:
                break

            print("Recording for 2 seconds. Hold still; press Q to discard and quit.")
            features = []
            record_start = time.perf_counter()
            next_sample = record_start
            while time.perf_counter() - record_start < 2.0:
                now = time.perf_counter()
                if now < next_sample:
                    time.sleep(min(next_sample - now, 0.01))
                    continue
                next_sample += 1.0 / 15.0
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped providing frames")
                timestamp_ms = int(time.perf_counter() * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = recognizer.recognize_for_video(image, timestamp_ms)
                if len(result.hand_landmarks) == 1:
                    handedness = result.handedness[0][0].category_name
                    features.append(hand_features(result.hand_landmarks[0], handedness))
                preview = cv2.flip(frame, 1)
                cv2.putText(preview, f"Recording {len(features)} valid frames / 30 (Q discards)",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
            if aborted:
                break
            try:
                _check_session_split(args.output, args.split, args.session)
            except ValueError as error:
                print(f"Trial not saved: {error}")
                break
            try:
                path = save_trial(args.output, args.split, args.session, args.label, features)
            except ValueError as error:
                print(f"Trial not saved: {error}. Press Space to record again.")
                continue
            trial_number += 1
            print(f"Saved trial {trial_number}/{args.trials}: {path}")
            if args.label == "NEUTRAL" and not features:
                print("This neutral trial has no hand frames; collect a separate neutral trial with a hand present too.")
            print("Lower your hand or change pose, then press Space for the next trial.")
        if trial_number == args.trials:
            print("Collection complete.")
    finally:
        if recognizer is not None:
            recognizer.close()
        camera.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--session", required=True, help="User-named participant/session identifier")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    _capture(args)


if __name__ == "__main__":
    main()

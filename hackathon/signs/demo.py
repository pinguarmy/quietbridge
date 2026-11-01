"""Standalone gesture check. No speech, uploads, or recordings; not Sam's server."""

import argparse
from dataclasses import asdict
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path, help="Process a local, consented video")
    source.add_argument("--image", type=Path, help="Repeat one static image: smoke test, not live QA")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=30, help="Live camera time limit")
    args = parser.parse_args()

    import cv2
    from .recognizer import SignRecognizer

    recognizer = SignRecognizer(model_path=args.model)
    cap = None
    times = []
    event_count = 0
    try:
        if args.image:
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError(f"Cannot read image: {args.image}")
            print("STATIC IMAGE SMOKE: repeated identical frame; not measured gesture response")
            packets = ((image, i * 100) for i in range(10))
        else:
            import sys
            camera = args.video is None
            backend = cv2.CAP_AVFOUNDATION if camera and sys.platform == "darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(args.camera if camera else str(args.video), backend)
            if not cap.isOpened():
                raise RuntimeError("Cannot open source. Allow Camera access; close other camera apps.")
            fps = cap.get(cv2.CAP_PROP_FPS) if not camera else 0
            if not camera and fps <= 0:
                raise ValueError("Video needs a valid frame rate for capture timestamps")
            started = time.perf_counter()

            def frames():
                last = -100
                index = 0
                while not camera or time.perf_counter() - started < args.seconds:
                    ok, frame = cap.read()
                    if not ok:
                        if camera:
                            raise RuntimeError("Camera stopped delivering frames")
                        break
                    ts = int(time.perf_counter() * 1000) if camera else round(index * 1000 / fps)
                    index += 1
                    if ts - last < 66:
                        continue
                    last = ts
                    yield frame, ts

            packets = frames()

        last_label = "No final gesture yet"
        for frame, timestamp_ms in packets:
            before = time.perf_counter()
            events = recognizer.process(frame, timestamp_ms)
            process_ms = (time.perf_counter() - before) * 1000
            times.append(process_ms)
            for event in events:
                event_count += 1
                last_label = event.label
                print(json.dumps({"event": asdict(event), "process_ms": round(process_ms, 2)}, ensure_ascii=False))
            if not args.headless and not args.image:
                preview = cv2.flip(frame, 1)
                cv2.putText(preview, last_label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                cv2.putText(preview, "Hold gesture; lower hands to re-arm. Q quits.", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("QuietBridge gesture QA (local only)", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        result = {"frames": len(times), "final_events": event_count}
        if times:
            ordered = sorted(times)
            result.update(process_ms_median=round(statistics.median(times), 2),
                          process_ms_p95=round(ordered[min(len(ordered) - 1, int(len(ordered) * .95))], 2))
        print(json.dumps(result))
    finally:
        recognizer.close()
        if cap is not None:
            cap.release()
        if not args.headless and not args.image:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Peter's gesture recognizer · Track B

Sam owns lips, runtime, fusion, UI and speech. Peter owns this directory. The baseline reuses MediaPipe's pretrained GestureRecognizer without a new dependency. Optional personal template collection and training are described in [PERSONALIZATION.md](PERSONALIZATION.md); these are not general sign-language recognition.

For guided Air recordings, run `.venv/bin/python -m hackathon.signs.lab` and open <http://127.0.0.1:8766>. Choose **Gestures** or **Lips**, then explicitly **Start recording / Finish & save**. One camera supplies a full hand-tracking view and a zoomed mouth view; the preview route tracks faces without saving mouth clips until recording is started. Gesture recordings save coordinates; lip recordings save labelled 96×96 grayscale mouth crops, never microphone audio or raw full frames. This standalone lab does not modify Sam's integration or require the Pro. Current custom motion-v2 labels are HELLO, WELCOME, OPENAI, DDE, HACKATHON and REST; collecting those labels does not mean their dynamic recognizer is trained. Preserve earlier static-pose data separately. See [the recording plan](PERSONALIZATION.md).

These are **four predefined communication gestures, not BSL/ASL translation**. A supported model class is not evidence of reliable live recognition. All vocabulary entries remain `verified: false` until the team tests the gesture and agrees its phrase.

## Integration contract

```python
from hackathon.signs.recognizer import SignRecognizer

recognizer = SignRecognizer()  # Load once; never download implicitly.
try:
    events = recognizer.process(frame_bgr, timestamp_ms)
finally:
    recognizer.close()
```

- Input: read-only, **unmirrored** OpenCV BGR frames from the runtime's single camera. The recognizer never opens another camera.
- Time: strictly increasing `int(time.perf_counter() * 1000)` on the same machine. Output is a flat list of the shared `hackathon.contracts.SignEvent` objects.
- Sam's existing `_hands` worker can call this directly at roughly 15 fps, keeping only the latest frame. Inference blocks that worker, not capture or UI.
- Only the raw supported top-1 category counts. No allowlist forces unknown poses into a supported class. Conflicting hands or any uncertain hand produce no candidate.
- A final event needs at least four consecutive observations spanning 250 ms. A sampling gap over 250 ms breaks the evidence.
- Holding a pose does not repeat the event. Lower hands or return to an unrecognized neutral pose for at least 400 ms of continuous observations before the next gesture. Switching directly to another gesture does not bypass the lock.
- The score is the minimum classifier score during the stable segment, **not measured accuracy or a calibrated probability**. Recognition and stabilization default to 0.75, matching fusion. A sub-threshold observation cannot lock out a later valid gesture.
- No inference network calls, recordings or automatic speech.

## Labels and proposed phrases

| Output label | MediaPipe category | Pose | Proposed demo phrase, not a translation |
| --- | --- | --- | --- |
| `THUMB_UP` | `Thumb_Up` | Thumb up | Yes. |
| `THUMB_DOWN` | `Thumb_Down` | Thumb down | No. |
| `OPEN_PALM` | `Open_Palm` | Open palm facing camera | Please stop. |
| `POINTING_UP` | `Pointing_Up` | Index finger pointing up | One moment, please. |

The recognizer does not emit untrained `HELP` or `WATER` labels. Do not relabel a thumbs-up as HELP. Pose-to-phrase meanings are demo conventions requiring user agreement. See `labels.json` for descriptions and `vocabulary.example.json` for Sam's `phrase/matches/verified` format.

**Replace the placeholder mapping before live integration:** Sam's current public vocabulary uses `HELP/STOP/YES/NO/WATER`, with no overlap with these output labels. Enabling those old entries still cannot accept this recognizer's events. After live checks, Sam should update the shared vocabulary, or use a local copy of this directory's example and enable only tested, agreed entries:

```sh
.venv/bin/python -m hackathon.server --sign-only --vocabulary /absolute/path/to/tested-vocabulary.json
```

The example intentionally has all entries false. Passing it unchanged checks loading but cannot produce live phrases. Do not mark every entry verified merely to get UI output.

## Model setup: once before going offline

From the repository root, using the existing environment:

```sh
.venv/bin/python -m hackathon.signs.assets --download
```

The explicit setup downloads Google's pinned 8,373,440-byte model to `~/.cache/quietbridge/gesture_recognizer.task`, outside Git. SHA-256: `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`. Downloads are checked; a different existing file is not overwritten. Override with `QUIETBRIDGE_GESTURE_MODEL=/path/to/gesture.task` or the constructor's `model_path`. Copy this cached model to the demo host if needed; inference then works offline.

## Local checks

```sh
# Logic and mocked inference-boundary tests; no camera.
.venv/bin/python -m pytest hackathon/signs/tests -q

# Standalone live check: preview, no speech or recording; Q quits.
.venv/bin/python -m hackathon.signs.demo --camera 0 --seconds 60

# A local, consented video. File timestamps are not wall-clock response latency.
.venv/bin/python -m hackathon.signs.demo --video /path/to/consented.mp4 --headless

# Repeated static frame: model/wiring smoke test only, not live gesture QA.
.venv/bin/python -m hackathon.signs.demo --image /path/to/hand.jpg --headless
```

Do not run the standalone checker and Sam's server together: they would compete for the camera. Use Sam's server for the integrated demo; the recognizer itself never opens a camera.

## Verification record: 1 November 2026

- Final combined command: `.venv/bin/python -m pytest tests hackathon/tests hackathon/signs/tests -q` — **167 passed, 11 skipped, 12 unittest subtests passed**. The gesture directory has 35 tests covering flat events, score thresholds, hold suppression, neutral rearming, conflicting hands, timestamps, vocabulary and missing-model messages. Mock backend predictions do not establish live accuracy.
- The 11 pre-existing skips cover Windows-only checks, opt-in pasting and a missing real-video sample. They are not failed tests suppressed to claim success.
- The actual Google model loaded. Ten repeated frames from each official `thumbs_down.jpg` and `pointing_up.jpg` produced one correct final event at synthetic `end_ms=300`; conservative scores were approximately 0.773 and 0.813, with mean processing around 25 and 23 ms. Artificial timestamps mean 300 ms is **not measured human response time**.
- Two official thumbs-up images did not remain above 0.75, so no final event was emitted. The threshold was not lowered for the samples. `OPEN_PALM` has not been checked on a real image. These few examples cannot establish accuracy or justify enabling vocabulary entries.
- Camera attempt in the current host process failed with `not authorized to capture video (status 0)`, before any live frame arrived. Allow the actual launching terminal/host in macOS **System Settings → Privacy & Security → Camera**. No personal frames were saved or uploaded.
- Model runtime identified this execution host as Apple M2; these are not MacBook Pro demo-machine measurements.

Official static-image sources: [thumbs up](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg), [thumbs down](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_down.jpg), [pointing up](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg). Images are temporary local files, not committed assets.

## Remaining live acceptance

1. Perform each gesture five times, holding about one second and lowering hands for at least half a second between trials. Log correct, missed and incorrect labels.
2. Hold one gesture for three seconds: one event only. Conflicting hands must not emit.
3. Test neutral poses for 30 seconds, then a second person and different lighting. If unreliable, keep only three demonstrated gestures rather than expanding vocabulary.
4. Agree each pose's phrase before enabling its entry. Then test sign-only, combined agreement, conflict confirmation, Wi-Fi off and three consecutive integrated demos.

`process_ms` measures inference work, not complete gesture-to-visible-text latency. Static images and synthetic predictions cannot prove live accuracy; historical lip-reader results cannot validate this gesture module.

## Sources and restrictions

- [MediaPipe Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python): pretrained classes, VIDEO mode and timestamp requirements.
- [Pinned model](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task). Check artifact-specific terms before redistribution or commercial use; the original Lipflow code's MIT license does not establish every weight's license.
- Existing Auto-AVSR/LRS3 research-use restrictions remain unchanged. Do not commit or upload private recordings or personal data.

# QuietBridge local hackathon demo

Sam owns the integration outside `hackathon/signs/`; Peter owns that directory. The
shared dataclasses in `hackathon/contracts.py` were pushed on `main` in commit
`44191d1`. The existing menu-bar app and recognition engine are unchanged.

## Run now without models

Python 3.11 or 3.12 is required. From the repository root:

```sh
python -m hackathon.server --rehearsal
```

Open <http://127.0.0.1:8765> for the animated product story, or go directly to
<http://127.0.0.1:8765/communicate> for communication. Start communicating and
Skip animation remain available throughout the story, and reduced-motion users
see readable copy immediately. This mode uses explicitly synthetic text and sign events;
it never opens the camera and does not claim vision accuracy or latency. Open
**Settings → Try an example** to preview matching or conflicting inputs. The app
returns to the communication screen. Choose an alternative when the inputs conflict,
then explicitly press **Confirm & speak**. Native
speech uses the installed macOS `say` voice; it requires no cloud service. No message
is spoken automatically. The visible lip sentence always needs confirmation because
the current engine does not supply a calibrated confidence score.

## Run live on the demo Mac

The target is **Peter's MacBook Pro**. See the [demo-machine handoff](../docs/demo-mac-handoff.md)
for reusing existing models while preserving the original personalized checkout.

Use the existing Python environment if already installed. For a fresh checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m hackathon.assets --download
.venv/bin/python -m hackathon.signs.assets --download
.venv/bin/python -m hackathon.server
```

The asset command downloads the same lip-reader and face-tracker files as the
upstream setup (about 1.2 GB). Alternatively, copy Peter's existing `models/vsr/`,
`models/lm/` and `models/face_landmarker.task` into this checkout's ignored `models/`
directory. It does not replace an app bundle, change hotkeys, load personal fine-tunes,
import dictation history, or download a cloud/text cleanup model.

Wait for the lip reader to show **ready**; it loads once and performs a warmup. Allow
Camera access for the terminal/Python process if macOS asks. Close the original
Lipflow app or other camera apps before testing. Python owns the only camera; the
browser receives local JPEG previews and does not request its own camera access.

Choose inputs on the separate **Settings** page; the main screen opens with lips
and gestures enabled. Diagnostics, vocabulary and rehearsal controls stay in Settings.
Hold the record button or Space on the communication page
for a 2–4 second sentence, then release. There is a 400 ms tail and a 10 second
recording limit; Escape or Cancel discards a recording. The hand worker continues
while the lip model decodes. For sign-only without loading the lip model:

```sh
.venv/bin/python -m hackathon.server --sign-only --camera 0
```

The server binds only to `127.0.0.1`. It has no API key requirement, external browser
assets, cloud calls, automatic speech, persistent recordings, or text-history file.
Models stay local and retain their research-use restrictions; see `NOTICE`.

## Peter's integration

Peter's `hackathon/signs/recognizer.py` is now integrated with this interface:

```python
from hackathon.contracts import SignEvent

class SignRecognizer:
    def __init__(self): ...
    def process(self, frame_bgr, timestamp_ms: int) -> list[SignEvent]: ...
    def close(self): ...
```

The runtime loads it once. `frame_bgr` is an unmirrored OpenCV BGR array; treat it as
read-only. Preview is mirrored for the user, but model input is not. `timestamp_ms`
is `int(time.perf_counter() * 1000)`, shared with lip capture. Process the supplied
frame only; **do not open another camera**. Hand processing is capped near 15 fps,
and its queue holds only the latest frame. Return only a final event after temporal
stabilization, return `[]` while a gesture stays held, and re-arm after neutral.
Use ordinary Python strings/floats/ints/bools in events. Scores below 0.75, UNKNOWN,
nonfinal events, future events and stale events never enter fusion.

`hackathon/vocabulary.json` now matches Peter's four labels: `THUMB_UP`,
`THUMB_DOWN`, `OPEN_PALM`, and `POINTING_UP`. The proposed phrases are Yes,
No, Please stop, and One moment. All four have `enabled: true` for confirmation-based
use and `verified: false` because live accuracy has not been measured. An unverified
enabled gesture displays its fixed phrase but requires Confirm & speak. Set
`enabled: false` to disable a mapping. Set `verified: true` only after the team tests
the corresponding recognizer output and agrees its phrase. `matches` contains whole word/phrase aliases
used to rerank lip candidates. The optional `--vocabulary /path/to/mapping.json`
flag selects another mapping. Restart the server after recognizer or mapping changes.
This is a small gesture recognizer, not a general BSL/ASL translator.

Use Peter's standalone checker to validate gesture quality. Enabling a mapping
does not claim recognition accuracy. Older mappings without `enabled` retain the
old behavior: only `verified: true` permits their events. See
`hackathon/signs/README.md` for model setup and live trials. Its earlier note about
the root placeholder labels is superseded by the now-aligned root vocabulary.
Do not run the standalone checker and integrated server together.

## Fusion and measurements

- Sign-only uses a fixed phrase for an enabled, supported final event. Unverified
  mappings require explicit confirmation; verified mappings still require a Speak click.
- Lip-only shows the top candidate and asks for confirmation.
- Combined reranks up to three lip candidates when a single sign concept agrees.
  A keyword cannot prove a whole sentence, so the user explicitly reviews the text
  and presses Confirm & speak.
- Conflict shows alternatives; no concatenated glosses or invented sentences.
- Negation and quantities prevent simple concept agreement. Multiple labels also
  require a choice. Missing lip output can still use a supported sign.
- Events align to the recording window's final four seconds, using capture time
  rather than decode completion time. Slow decoding does not age valid signs out.
- For combined input, hold Record first, then gesture and mouth your words. A
  gesture held before recording will not emit again until the hand returns neutral.
- Every decision has a revision. A new result invalidates stale confirmation/speech
  requests. Speech checks both the current revision and exact reviewed text.

The main UI shows measured release-to-text time when available. Settings →
Diagnostics shows this time, crop-preparation-plus-decode time and local fusion
compute time. Hand
process time, final-event age and TTS dispatch time are included in `/api/state`.
Dispatch time is not audible speech onset. Missing measurements remain blank.
Rehearsal never populates vision timing values.

## Checks and remaining live acceptance

```sh
python -m unittest discover -s hackathon/tests -v
python -m hackathon.assets
```

The standard-library integration tests use synthetic events and a loopback HTTP
server, without model downloads. They do not establish live accuracy. Before the
demo, test five new lip sentences, each chosen gesture five times plus neutral,
agreement and conflict, missing face/hand, and Wi-Fi off. Log actual errors and
latency; require three consecutive successful live runs. Keep a consented offline
sample recording as a fallback; never commit personal clips or model weights.

Optional API suggestions are deferred until the local path is demonstrated.

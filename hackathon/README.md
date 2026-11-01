# QuietBridge local hackathon demo

Sam owns the integration outside `hackathon/signs/`; Peter owns that directory. The
shared dataclasses in `hackathon/contracts.py` were pushed on `main` in commit
`44191d1`. The existing menu-bar app and recognition engine are unchanged.

## Run now without models

Python 3.11 or 3.12 is required. From the repository root:

```sh
python -m hackathon.server --rehearsal
```

Open <http://127.0.0.1:8765>. This mode uses explicitly synthetic text and sign events;
it never opens the camera and does not claim vision accuracy or latency. Try matching
and conflicting inputs, confirm one displayed alternative, and press Speak. Native
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

Select Gestures, Lips or Both together. Hold the record button or Space on the page
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

Provide `hackathon/signs/recognizer.py` with:

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

Sam will update `hackathon/vocabulary.json` to Peter's exact labels and fixed
phrases. Current entries are placeholders, all `verified: false`: live fusion ignores
them. Set an entry to `verified: true` only after the team tests the corresponding
recognizer output and agrees its phrase. `matches` contains whole word/phrase aliases
used to rerank lip candidates. The optional `--vocabulary /path/to/mapping.json`
flag selects another mapping. Restart the server after recognizer or mapping changes.
This is a small gesture recognizer, not a general BSL/ASL translator.

## Fusion and measurements

- Sign-only uses a fixed phrase for a supported, verified final event.
- Lip-only shows the top candidate and asks for confirmation.
- Combined reranks up to three lip candidates when a single sign concept agrees.
  A keyword cannot prove a whole sentence, so the user confirms before speaking.
- Conflict shows alternatives; no concatenated glosses or invented sentences.
- Negation and quantities prevent simple concept agreement. Multiple labels also
  require a choice. Missing lip output can still use a supported sign.
- Events align to the recording window's final four seconds, using capture time
  rather than decode completion time. Slow decoding does not age valid signs out.
- Every decision has a revision. A new result invalidates stale confirmation/speech
  requests. Speech requires an explicit click against the current revision.

The UI shows measured release-to-text time (including tail, crop preparation and
decode), crop-preparation-plus-decode time, and local fusion compute time. Hand
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

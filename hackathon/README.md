# QuietBridge application guide

QuietBridge is a local-first, per-user communication tool. The main app turns lip movements and supported gestures into reviewable text and local speech. The recording lab handles profile setup, customization and independent data collection.

## Surfaces

```sh
# Main app, browser camera
uv run python -m hackathon.server --browser-camera

# Main app with a personal profile and explicit lip candidates
uv run python -m hackathon.server --browser-camera --profile alice \
  --lip-model /absolute/path/to/lip-candidate.pth \
  --lip-language-model /absolute/path/to/lm-candidate.pth

# Profile setup and guided recording
uv run python -m hackathon.signs.lab
```

Main app: <http://127.0.0.1:8765>. Recording lab: <http://127.0.0.1:8766>. Both bind to loopback. Rehearsal remains available with `--rehearsal`; it uses synthetic input and does not validate recognition.

## Profiles and data

`hackathon/profiles.py` owns safe profile IDs, customization and directory isolation. A profile contains:

```text
profile.json
sequences/{train,validation,test}/
mouths/{train,validation,test}/
models/
```

A session ID belongs to exactly one split. Validation and final test require new recording sessions. Existing legacy recordings are not migrated or deleted.

The main app uses `models/motion.npz` only when it exists beside the selected profile. Otherwise it keeps stock gestures and their compatible vocabulary and displays a setup warning. Candidate filenames are not automatically promoted.

## Recognition contracts

Shared types live in `hackathon/contracts.py`.

- `SignEvent` carries a supported label, model-specific score, start/end monotonic milliseconds and final status.
- `LipResult` carries ordered candidates, decoder scores, capture timing, actual raw fps and frame gaps.
- `Decision` carries review status, reason and matched gesture labels.

Scores are not calibrated probabilities and are not comparable across recognizers. Stock MediaPipe applies its own 0.75 floor before emission. MotionModel applies its learned distance/margin acceptance; fusion does not apply a second universal 0.75 gate.

Custom motion uses `SignRecognizer.start()`, `process()` and `finish()` for one explicit window. `cancel()` discards it. REST, rejected and empty windows emit no event.

## Runtime invariants

- One camera supplies independent latest-frame workers.
- Each recording has a generation ID. Late results from cancelled generations are discarded.
- Sign-only custom windows never start lip capture or decoding.
- Combined lip decoding waits for the current motion result. Motion failure lets lips continue without gesture evidence.
- Frames before the window and after its tail are excluded, including frames arriving while `finish()` runs.
- Explicit windows retain gestures from the whole recording; only ambient stock events use the four-second freshness rule.
- Fusion reranks on agreement, reports partial agreement and composes supported gesture-only phrases in time order. UNKNOWN contributes no text.
- The user reviews text and controls speech. No uncertain output is spoken automatically.

## Training and evaluation

See the root [README](../README.md) for exact motion, visual-lip and language-model candidate commands. Training and evaluation use profile data but do not deploy automatically.

Reports must distinguish positive gesture recall from REST silence, and ordinary lip sentences from named entities or other difficult targets. Preserve raw candidates and per-trial diagnostics. Never tune on final test.

## Checks

```sh
uv run pytest
uv run lipflow doctor
node --check hackathon/ui/app.js
```

Unit tests, fake backends and loopback HTTP checks establish software behavior only. Real camera quality, live accuracy, long-running false triggers and audible speech require separate on-device acceptance.

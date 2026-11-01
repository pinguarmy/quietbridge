# Immediate demo integration handoff

## Implemented vs not yet accepted

- The stock recognizer emits `THUMB_UP`, `THUMB_DOWN`, `OPEN_PALM`, `POINTING_UP`. Replace the old HELP/STOP/YES/NO/WATER mapping before enabling live fusion. Only mark a phrase verified after live testing and user agreement.
- Personal static-template data collection, training and opt-in inference exist locally. No new user data or live accuracy improvement has been established. Reports distinguish label-only correctness from the actual 0.75 runtime gate.
- DDE is now a requested two-hand temporal expression. The browser Gesture Lab captures its labelled sequences but **does not recognize DDE yet**. Do not map any stock pose to DDE or describe sequence recording as training success.
- The requested line is “Hi, welcome to OpenAI DDE Community Hackathon.” The broader demo now permits clearly labelled personal symbols; genuine BSL fingerspelling remains a separately checked capability. Never force the fixed line onto unrelated input.

## Proposed custom concept vocabulary

The latest user proposal is complementary inputs, not word-for-word BSL translation:

| Concept label | User's proposed expression | Fixed spelling |
| --- | --- | --- |
| `HELLO` | One open hand: curl fingers in and open once | Hi |
| `WELCOME` | Both hands close in front of chest, draw back toward chest | Welcome |
| `OPENAI` | Both thumbs extended, mirrored inward half-circle arcs, finish one thumb above the other | OpenAI |
| `DDE` | Index extended, move forward and back once | DDE |
| `HACKATHON` | Both open palms move inward from sides and stop without clapping | Hackathon |
| `REST` | Relaxed half-fist or hands lowered; no command | No output |

These are **custom dynamic collection labels**, not live-supported concept outputs or standard BSL. This motion-v2 definition replaces the guided static poses (thumb up, index up, open palm); preserve those earlier train/test files as a separate baseline, never mix them with dynamic-v2 trials as if they were the same action. REST is the user's relaxed half-fist/lowered hands and must emit nothing. No dynamic sequence classifier has been trained yet. Save each whole motion including timing, using the new `peter-motion-v2-*` sessions. Keep independent REST/incomplete/contrasting sequences and fresh test sessions.

Lips supply the sentence; gestures provide contextual evidence for names. Use the fixed concept-to-spelling map before any AI suggestion. Do not infer a whole welcome line from one concept or treat OAI/HCKTH custom shorthand as standard BSL. The current general fusion handles one label and treats multiple distinct labels as conflict; composing several aligned concepts will need Sam's explicit integration change and tests, not just a vocabulary edit.

## Mouth-data capture

The same simplified lab now has Gestures / Lips tabs. MouthCapture saves labelled aligned `(T,96,96)` grayscale mouth crops and text, plus capture timestamps/session/split metadata, outside Git under `~/.local/share/quietbridge/mouths/{train,test}`. It reuses the FaceTracker/ROI pipeline without loading the VSR network or opening another camera. Raw full frames and audio are not saved. These mouth crops remain sensitive personal visual data. Explicit Start/Finish is required; a genuine mouth box appears only while actively tracked. Failed quality checks must not advance the lesson count. This is training-data collection, not a claim of successful decoding or a trained personal model.

Routes: `/api/mouth/start` takes `{text,split,session}`, `/api/mouth/stop` saves, `/api/mouth/discard` drops an incomplete buffer; `state.mouth` / `frame.mouth` expose status. Do not start both a hand-dataset trial and mouth-dataset trial together. Test data uses separate sessions. Runtime service must be restarted at an idle point to load added routes; do not interrupt a user recording.

## Personal lip model

`hackathon/lips_adapter.py` currently passes `personal=False`. Please agree and add an explicit opt-in personal lip-model selection, display whether it loaded, and preserve stock mode. The old Lipflow app already supports face adaptation, but its existing personal file is not proof that the new welcome phrase works. Use separate training, validation and final-test recordings; retain the old model.

## Optional Luna Fast suggestions

The user asked for a quick text-optimization layer. Old Lipflow's current cleanup configuration is CPA Gemini/polish, not fully local inference. The new hackathon runtime does not call that cleanup path. Luna Fast via CPA is likewise an API, not an on-device model.

Shared runtime/UI changes belong to Sam. Suggested minimal hook:

1. Start one asynchronous request only after local candidates exist and the user opts in.
2. Send bounded candidate text and stable gesture labels, no frames or raw video.
3. Bind the request and result to the source revision; discard stale responses.
4. Display the suggestion separately. Preserve raw candidates and the local decision.
5. Never auto-confirm, overwrite confirmed text or change already-spoken content.
6. Require explicit selection/confirmation before Speak.
7. API missing, timeout or failure must not affect the local flow.
8. Measure complete request latency on the event network before making speed claims. Provider name “Fast” is not a timing measurement.

## Machines and current runnable surfaces

- Air: camera, live gesture processing, UI, confirmation and audible output.
- Home Pro: independent `~/code/quietbridge-demo`, Python 3.11 environment, stock vision assets, PyTorch MPS verified. The lip adapter actually loaded and warmed up (about 4.69 seconds in one measurement). No new personal training has run.
- Current UI server: `python -m hackathon.server --rehearsal --port 8765`; explicitly synthetic, no camera.
- Browser gesture lab: `python -m hackathon.signs.lab --port 8766`; browser camera input stays on the Air/local loopback. No raw frame persistence or microphone input. User explicitly starts and saves each sequence.
- Current server cannot use an Air camera by merely viewing a web page served on the Pro. Remote VSR needs a separate authenticated/encrypted mouth-clip path and measured end-to-end performance. Do not remove loopback/Host/Origin checks or expose it publicly as a shortcut.

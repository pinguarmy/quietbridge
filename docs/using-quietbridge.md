# Using QuietBridge

Reference for the main communication page, profiles and personalization. Start with the [README quick start](../README.md#quick-start). Command examples use `uv run quietbridge`, which is the same as `uv run python -m quietbridge.server`.

## Profiles and data

Each profile has isolated configuration and data under `~/.local/share/quietbridge/profiles/<id>/`:

```text
profile.json                 Custom concepts, instructions and lip prompts
sequences/{train,validation,test}/   Hand landmarks
mouths/{train,validation,test}/      Labelled mouth crops
models/                      Explicitly selected model candidates
corrections.jsonl            Corrections you explicitly chose to learn from
```

A session ID belongs to exactly one split. Validation and final test require new recording sessions; never split validation from the same recording session as training. Existing legacy Lipflow recordings are not migrated or deleted.

Stock models and gestures remain the default. Select a profile explicitly:

```sh
uv run quietbridge --browser-camera --profile alice
```

If `alice` has a promoted `models/motion.npz`, the app uses its custom vocabulary and dynamic recognizer. Otherwise it keeps compatible stock gestures and shows a setup warning. Candidate filenames are never promoted automatically.

## Personalize: the recording lab

Choose **Personalize** in the main app to create a profile, customize gestures and prompts, and record independent sessions. The main app opens its own local recording page on demand; no second command or fixed port is required. Camera tracking loads only after an explicit recording action. Rehearsal permits profile editing but does not permit camera capture.

You can still run the recording lab separately:

```sh
uv run python -m quietbridge.signs.lab
```

Open <http://127.0.0.1:8766>, or choose **Personalize** in the main app. New profiles begin with **Choose phrases**, then **Practice**, **Check on another occasion**, **Final check** and **Activate**. These remain separate train, validation and test recording sessions. Gesture details expand individually; internal labels and sentence templates stay under Advanced. Capture statistics and session IDs are optional details. The model page shows one training action for the selected model type and preserves explicit activation and rollback.

The browser targets 25 fps and displays the actual received rate; resampling does not recover lost frames. Personalize asks before saving local training samples, and separately asks before each full-frame sign-video trial. Saving or discarding a trial turns the camera off. Leaving with an active trial warns first; the page then attempts an authenticated discard and releases browser tracks. Exit delivery is best effort, not a guarantee after an abrupt browser termination.

Training and promoting candidates is described in [Training and evaluation](training-and-evaluation.md).

## Recording windows

Start the camera, then choose **Start lips + gesture recording** for one custom motion and its words. Choose **Finish recording** to decode the shared window. The button is a toggle, so there is no sustained hold requirement; Space from the page background also toggles capture. **Finish message** stops the camera for final review. REST, rejected and cancelled windows emit nothing. Pure lip input and stock gestures retain automatic capture. The main recording limit defaults to 30 seconds and can be changed with `--max-utterance-seconds`.

In browser-camera mode (`--browser-camera`) the preview is the browser's own local video. Frames are also uploaded to the loopback server, for recognition only. Without `--browser-camera`, the server opens the camera itself (`--camera <index>`).

During a camera session, lip activity starts a chunk, about 700 ms of low activity ends it, and a bounded in-memory pre-roll retains the preceding 300 ms. Chunks accumulate into a session until you stop the camera. Review alternatives replace one window at a time and retain the selected text from every other window; the app does not build every possible cross-window combination. These geometric thresholds are segmentation hints, not speech confidence, and still need real-user validation.

A custom motion model always needs its own label-to-phrase vocabulary, which a profile supplies. Starting with a motion model and no vocabulary (for example through `QUIETBRIDGE_MOTION_MODEL` without `--profile` or `--vocabulary`) stops with an error; stock or fixed meanings are never applied to it.

Automatic motion segmentation is available as an opt-in experiment with `QUIETBRIDGE_MOTION_AUTO=1`. It detects movement/rest windows without the Record button. Its thresholds are engineering defaults, not calibrated user-specific probabilities; validate false triggers and missed motions before choosing it for regular use.

## The communication page

The communication page separates capture, review and speech. Its short first-use guide can be skipped and reopened. Typing is visible without a camera or training, and a skip link goes straight to it. **Use this message** applies the words without speaking or learning automatically; Ctrl+Enter (Cmd+Enter on a Mac) in the text box does the same and never speaks. Unfinished edits stay in this tab through Settings/Personalize navigation and new recognition results; they are cleared when applied or deliberately discarded with **New message**. **Speak** is disabled while an edit is unapplied. **Stop speaking** interrupts local speech. **Show large** displays the applied (or explicitly selected) message full-screen for showing the screen to someone. Message and camera order stays fixed at every screen size, with message first in both visual and keyboard order. The page follows the operating system's light/dark setting. The main page names the active backend and enabled concepts. Mouth framing is optional.

On-device speech uses macOS `say`; Windows PowerShell (or `pwsh`) with `System.Speech`; or on Linux the first installed of `espeak-ng`, `espeak` and Speech Dispatcher's `spd-say`. If none is available the page says so and offers **Show large** or **Copy message**. Audible output from these backends has not been verified on real hardware in this round.

Add `--open` to open the page in the default browser once the server is listening; this is best effort, and the URL is always printed.

### Server restarts and pending actions

The page tracks the running server's identity separately from a message revision. After a server restart it stops browser capture, accepts the fresh session and preserves unfinished typing for review. Message actions have a 15-second deadline and are never automatically retried, because a timed-out action may already have committed. The next poll can still show fresh state while an action is pending. New message only clears edits that existed when it was confirmed, not words typed while waiting.

## Explicit and promoted models

An explicit personal lip candidate and language model can be selected without replacing the legacy Lipflow defaults:

```sh
uv run quietbridge --browser-camera --profile alice \
  --lip-model /absolute/path/to/lip-candidate.pth \
  --lip-language-model /absolute/path/to/language-candidate.pth
```

The main app loads promoted `models/motion.npz`, `models/lip.pth` and `models/lm.pth` from the selected profile at startup or profile reload. Saving a model in Personalize does not change a recognizer already in memory. Stop the camera, open Settings and choose **Reload current models**. Reload clears the draft and preserves explicit command-line overrides. Wait for model readiness before starting another recording. Settings shows the active profile, separate visual/language model identities, actual frame rate, frame gaps and latency. Missing explicit model paths fail visibly; an old personal model is never loaded silently instead.

For an explicit isolated-sign bundle (`--sign-model`), see [Standard signs with lips](standard-signs.md).

## Corrections

You can edit an incorrect message before speaking. Applying a correction never speaks automatically. Only an explicit learning choice stores it in that profile's local `corrections.jsonl`; otherwise it remains an in-memory correction. Future language-model candidates include those approved corrected texts in training and report their count. Known validation/test sources and contradictory source metadata are excluded from correction learning, including historical records. Training reports source coverage, excluded records and source session/trial IDs. Consented day-to-day corrections without a recording source remain usable, but are marked as unknown provenance. This is not proof of evaluation independence; do not learn from a held-out session.

## Fusion and safety

- Lip candidates retain decoder scores and capture timing; scores are not calibrated probabilities.
- Gesture agreement can rerank candidates. Partial agreement remains reviewable.
- An optional profile `intent_template`, such as `I want {gesture}`, fills a concept only when the lip candidate exactly matches the declared static context. Unknown words and conflicting gestures are not erased.
- Supported gesture-only sequences compose in time order. Unknown input never contributes text.
- Combined decoding waits for the current dynamic-gesture result. Cancellation invalidates late work from either worker.
- An explicit recording window retains gestures from the whole window; ambient stock events still expire.
- Text and local speech work offline. Optional API formatting stays asynchronous and revision-bound. It can normalize spacing or add a final period, not replace words or translate sign grammar. Suggestions are separate choices; errors preserve local text, and changing AI settings removes unselected suggestions.
- The person reviews the text and controls speech. No uncertain output is spoken automatically.

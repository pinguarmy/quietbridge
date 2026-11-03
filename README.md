# QuietBridge

**Lips and custom gestures, turned into a voice.** A local-first communication tool under active development.

One webcam supports silent lip movements, stock gestures or a personal dynamic-gesture model. The person reviews the proposed text and presses **Speak** for on-device speech. QuietBridge is not a general sign-language translator.

## Product workflow

Each profile has isolated configuration and data under `~/.local/share/quietbridge/profiles/<id>/`:

```text
profile.json                 Custom concepts, instructions and lip prompts
sequences/{train,validation,test}/   Hand landmarks
mouths/{train,validation,test}/      Labelled mouth crops
models/                      Explicitly selected model candidates
```

Open the recording lab to create a profile, customize gestures and prompts, and record independent sessions:

```sh
uv run python -m hackathon.signs.lab
```

Open <http://127.0.0.1:8766>, or choose **Personalize** in the main app. New profiles begin with **Choose phrases**, then **Practice**, **Check on another occasion**, **Final check** and **Activate**. These remain separate train, validation and test recording sessions. Gesture details expand individually; internal labels and sentence templates stay under Advanced. Capture statistics and session IDs are optional details. The model page shows one training action for the selected model type and preserves explicit activation and rollback.

The browser targets 25 fps and displays the actual received rate; resampling does not recover lost frames. Both local server processes must be running to move between the main app and Personalize.

## Run the main app

```sh
uv sync --frozen
uv run python -m hackathon.assets --download
uv run python -m hackathon.signs.assets --download
uv run python -m hackathon.server --browser-camera
```

Open <http://127.0.0.1:8765>. Stock models and gestures remain the default. Select a profile explicitly:

```sh
uv run python -m hackathon.server --browser-camera --profile alice
```

If `alice` has a promoted `models/motion.npz`, the app uses its custom vocabulary and dynamic recognizer. Otherwise it keeps compatible stock gestures and shows a setup warning. Start the camera, then hold **Record lips + gesture** or Space for one custom motion and its words. Release to decode the shared window. **Finish message** stops the camera for final review. REST, rejected and cancelled windows emit nothing. Pure lip input and stock gestures retain automatic capture.

The communication page separates capture, review and speech. Mouth framing is an optional preview. You can type a message without a camera; applying it never speaks automatically. On narrow screens, a nonempty message moves above the camera so review and speech remain easy to find.

Automatic motion segmentation is available as an opt-in experiment with `QUIETBRIDGE_MOTION_AUTO=1`. It detects movement/rest windows without the Record button. Its thresholds are engineering defaults, not calibrated user-specific probabilities; validate false triggers and missed motions before choosing it for regular use.

During a camera session, lip activity starts a chunk, about 700 ms of low activity ends it, and a bounded in-memory pre-roll retains the preceding 300 ms. Chunks accumulate into a session until you stop the camera. Review alternatives replace one window at a time and retain the selected text from every other window; the app does not build every possible cross-window combination. These geometric thresholds are segmentation hints, not speech confidence, and still need real-user validation.

An explicit personal lip candidate and language model can be selected without replacing the legacy Lipflow defaults:

```sh
uv run python -m hackathon.server --browser-camera --profile alice \
  --lip-model /absolute/path/to/lip-candidate.pth \
  --lip-language-model /absolute/path/to/language-candidate.pth
```

The main app loads promoted `models/motion.npz`, `models/lip.pth` and `models/lm.pth` from the selected profile at startup or profile reload. Saving a model in Personalize does not change a recognizer already in memory. Stop the camera, open Settings and choose **Reload current models**. Reload clears the draft and preserves explicit command-line overrides. Wait for model readiness before starting another recording. Settings shows the active profile, separate visual/language model identities, actual frame rate, frame gaps and latency. The main recording limit defaults to 30 seconds and can be changed with `--max-utterance-seconds`.

You can edit an incorrect message before speaking. Applying a correction never speaks automatically. Only an explicit learning choice stores it in that profile's local `corrections.jsonl`; otherwise it remains an in-memory correction. Future language-model candidates include those approved corrected texts in training and report their count. Known validation/test sources and contradictory source metadata are excluded from correction learning, including historical records. Training reports source coverage, excluded records and source session/trial IDs. Consented day-to-day corrections without a recording source remain usable, but are marked as unknown provenance. This is not proof of evaluation independence; do not learn from a held-out session.

## Train and evaluate candidates

The recording lab collects data; it does not silently deploy a model. Train a motion candidate from custom profile sessions:

```sh
uv run python -m hackathon.signs.train_motion \
  --data "$HOME/.local/share/quietbridge/profiles/alice/sequences" \
  --session-prefix "" \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/motion.candidate.npz"
```

The trainer fits on train, calibrates acceptance thresholds on independent validation, then evaluates final test once. Per-label thresholds fall back to train leave-one-out rules when validation coverage is insufficient. The report separates positive recall, REST silence and wrong labels and includes per-trial rejection diagnostics. Review validation and final test before promoting a candidate to `models/motion.npz`. The old `--allow-legacy-no-validation` exists only to reproduce historical data.

Train a lip candidate using the same beam width as the main app:

```sh
uv run python -m hackathon.signs.train_lips \
  --data "$HOME/.local/share/quietbridge/profiles/alice/mouths" \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/lip.candidate.pth" \
  --beam 4
```

Training requires an independent `validation/` session. Selection rejects overall WER regression, target-prompt regression and increased contrast false welcomes. Existing candidates are not automatically promoted.

Train a personal language-model candidate from the same profile's independent text sessions:

```sh
uv run python -m hackathon.signs.train_language \
  --data "$HOME/.local/share/quietbridge/profiles/alice/mouths" \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/lm.candidate.pth"
```

The language report records held-out and general-English perplexity. A candidate is saved only when validation improves without excessive general regression. Review the report before selecting it with `--lip-language-model`; no candidate is deployed automatically.

Perplexity alone does not establish recognition quality. Compare the same VSR and test ROIs with stock versus personal language models:

```sh
uv run python -m hackathon.evaluate_language \
  --data "$HOME/.local/share/quietbridge/profiles/alice/mouths" \
  --candidate "$HOME/.local/share/quietbridge/profiles/alice/models/lm.candidate.pth" \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/lm-comparison.json"
```

The comparison is diagnostic, uses beam4 and never promotes a candidate. Language promotion requires valid finite, nonnegative error rates and matching candidate/visual SHA-256 identities. Changing either model invalidates the old comparison. Evaluate again to save a new report without overwriting the old one; unchanged reports can be reused.

## Paired fusion evaluation

Use the recording lab's **Paired** mode to capture the same expression as aligned mouth crops and hand landmarks. Both files share a `pair_id`, profile, session and split, while retaining independent trial IDs. Ordinary single-modality recordings can remain in the same profile directories.

After promoting the models to evaluate, run the same paired samples through both paths:

```sh
uv run python -m hackathon.evaluate_fusion \
  --profile alice --split test \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/fusion-diagnostic.json"
```

The report compares lips-only, gestures-only and fusion WER and exact intent. It also reports gesture positive recall, wrong labels, and REST/unknown false emissions with their denominators. Missing negative samples produce null rates, not claims of zero false triggers. A gesture may supply only one slot, so its standalone text can be shorter than the full reference. Default models follow the profile's promoted paths, with SHA-256 identities; explicit flags override them. The output remains diagnostic. It neither proves that the model never saw test data nor measures live accuracy or response time.

See the [recognition audit and cooperation plan](docs/recognition-audit.md) for supported sign input, fixed fusion defects and remaining evaluation limits.

For capture-only checks, `uv run python -m hackathon.signs.capture_analysis <mouth.npz>` reports actual fps, frame gaps, face coverage and mouth-activity edges. Older recordings without activity metadata report those fields as null.

## Fusion and safety

- Lip candidates retain decoder scores and capture timing; scores are not calibrated probabilities.
- Gesture agreement can rerank candidates. Partial agreement remains reviewable.
- An optional profile `intent_template`, such as `I want {gesture}`, fills a concept only when the lip candidate exactly matches the declared static context. Unknown words and conflicting gestures are not erased.
- Supported gesture-only sequences compose in time order. Unknown input never contributes text.
- Combined decoding waits for the current dynamic-gesture result. Cancellation invalidates late work from either worker.
- Text and local speech work offline. Optional API suggestions must remain asynchronous and revision-bound.

## Verification

```sh
uv run pytest
node --check hackathon/ui/app.js
node --test hackathon/tests/test_ui_flow.cjs hackathon/signs/tests/test_lab_*.cjs
```

Synthetic checks and fake backends establish integration behavior, not live recognition accuracy. Real use still requires new-session camera trials, paired lips/gesture evaluation, REST/unknown false-trigger measurements and audible speech checks. Do not tune on the final-test session.

## References and licensing

- The [original Lipflow guide](docs/lipflow-legacy.md) preserves the base application and historical benchmarks.
- Based on [amywork777/lipflow](https://github.com/amywork777/lipflow), retained as the `upstream` remote.
- See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution and model restrictions. Treat LRS3 weights as non-commercial research use.

This repository is private. Do not commit keys, models, profile settings or personal recordings.

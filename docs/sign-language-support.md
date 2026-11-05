# Standard sign support and runtime findings

Checked 2026-11-04. Standard sign recognition is now part of QuietBridge's product direction. Personal gestures alone do not meet that goal.

## What is implemented

A local RGB-video backend can run an explicitly supplied BSL isolated-sign model alongside lip reading. Both workers use the same hold/release window. The sign model supplies a concept, lips supply ranked English candidates, and local fusion preserves alternatives and disagreements for review. For example, an accepted WATER sign can select `water` rather than `tea` from the lip candidates. This is not BSL sentence-to-English translation.

The BSL-1K path uses the official 1,064-class I3D checkpoint and exact class ordering, exported to a fixed-shape TorchScript bundle.[1] The app does not replace BSL with ASL or relabel stock poses as standard signs. ASL is an allowed bundle language, but no ASL pretrained checkpoint was integrated or verified here.

Only explicitly enabled concepts can contribute text. The adapter ranks the **whole classifier head** before checking the enabled set; it never chooses the best of four enabled words when the actual best prediction is another class. Low score, low margin, unmapped class, insufficient capture, large frame gaps, nonfinite output or buffer overflow produce no event. A prediction always requires user confirmation. A high softmax score still cannot prove that an arbitrary movement is a valid sign: this classifier has no trained REST/unknown class.

Implementation: [video adapter](../hackathon/signs/isolated.py), [BSL exporter](../hackathon/signs/export_bsl1k.py), [shared workers](../hackathon/runtime.py), [worker regressions](../hackathon/tests/test_sign_window_runtime.py).

## How to enable it

1. Establish permission for the external **source, model weights and data** separately. Downloadability is not a usage or redistribution license.
2. Install the upstream BSL-1K source outside this repository. Obtain its official checkpoint and class-ordered labels. Keep the 1,064 entries in their exact model order, not alphabetical order inferred from another dataset.[1]
3. Export into a new local directory. Run the acknowledgement flag only if you actually hold the required rights:

```sh
uv run python -m hackathon.signs.export_bsl1k \
  --source /absolute/path/to/external/bsl1k \
  --checkpoint /absolute/path/to/bsl1k.pth.tar \
  --labels /absolute/path/to/class-ordered.json \
  --output /absolute/path/to/new-bsl-bundle \
  --enable water --enable apple \
  --acknowledge-model-terms

uv run python -m hackathon.server --browser-camera \
  --sign-model /absolute/path/to/new-bsl-bundle/manifest.json
```

The exporter imports the explicitly supplied external source and reads checkpoint tensors with `weights_only=True`; no unsafe-pickle fallback is attempted. The official vocabulary is a small pickle containing `words` and `words_to_id`.[1] Convert trusted vocabulary metadata to the JSON string list in `words` order with a data-only reader that rejects object construction; do not load arbitrary pickles. No upstream source, weight file, private recording or converted model is committed to this repository.

The runtime verifies the exported model hash before loading. Only use trusted local TorchScript bundles. The hash identifies bytes, not their permission, provenance or accuracy.

In the app, check Settings for **BSL isolated signs**, model identity and enabled concepts. Start the camera, keep your upper body, face and both hands in view, choose Start recording for one sign and its words, then choose Finish recording. Keep each sign short. The adapter retains at most 240 resized frames, approximately 9.6 seconds at 25 fps; it rejects overflow rather than classifying a truncated ending. Finish message, review, edit if necessary, then choose Speak. Multiple windows accumulate a message; they are not a continuous-sign decoder.

Explicit sign-model failure is visible and does not fall back to stock poses. The command-line selection persists across profile reloads, like other explicit model choices. Personal gesture training does not fine-tune this BSL model.

## Artifact and license checks

| Path | Verified facts | Decision |
| --- | --- | --- |
| BSL-1K I3D | Official public checkpoint and 1,064-class vocabulary exist. The inspected repository has no root license and GitHub reports `license: null`.[1] | External engineering adapter implemented. Redistribution and product-use permission are **not established**. Do not bundle or silently activate it. |
| OpenHands | Official AI4Bharat repository publishes ASL/WLASL checkpoint links and Apache-2.0 source. It separately tells users to check dataset terms.[2] | Possible future ASL route. Do not infer a blanket model/data license from the code license. Not integrated here. |
| BOBSL | Data access requires a BBC agreement and a personal password. The official page excludes independent researchers, secondary-school students and commercial organisations.[3] | No protected dataset was requested or downloaded; this is not a one-click personal-app dataset route. |
| BSLDict | The project states CC BY 4.0 for its dataset and retains video owners' copyrights.[4] | Its public demo is usable as a disclosed engineering input; this does not establish BSL-1K checkpoint permission or independent accuracy. |

The actual inspected BSL-1K checkpoint URL is:

`https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k.pth.tar`

It is 107,178,502 bytes, SHA-256 `29db97e577af6c08e6a081b726802e7707de35a02c2545ec5d8f46c1217f93ec`. Source was inspected at commit `bfeaa3a8463152a79bcfd0fafd358c4d203b4c16`. The official vocabulary SHA-256 is `b9e451cc38b3fbaab6ec0bdc899562d53b92e2d3654a72c0a0b7cde89addd3a8`.

The fixed inference input is RGB `[1, 3, 16, 224, 224]`, sampled at a 25 fps target. Frames are resized to 256 square, center-cropped and normalized from byte RGB by `/255 - 0.5`, matching the inspected upstream demo and normalization. Capture statistics describe actual frames, not recovered motion. The one-window aggregation averages clip logits; that aggregation is an engineering choice, not a published webcam benchmark.

## What was actually run

- Real official checkpoint loaded strictly into the external I3D architecture. No missing/unexpected state-dict keys were ignored.
- Exported checkpoint loaded through the actual QuietBridge adapter. Original and exported logits matched exactly on one real public-video clip, maximum absolute difference `0.0`. This tests conversion, not label correctness.
- Public BSLDict demo input, 174 frames at 25 fps, passed through the adapter. The whole-window top class was `over`, score `0.0488`, margin `0.0041`. With only apple/garden/tree/water enabled, it emitted **nothing**. This multi-sign clip was not a successful positive-recognition test; no threshold was lowered to manufacture a result.
- On this Apple M5 Pro host with CPU inference and four Torch threads, individual 16-frame forward passes took about 0.20–0.26 seconds. The 16-clip whole window took 3.21 seconds. These are single-run smoke timings, not camera-to-text latency or a portability guarantee.
- An isolated local server loaded both the real BSL bundle and the existing real lip model. The rendered browser reported `lips=ready`, `signs=ready`, backend `isolated`, matching model hash and four enabled concepts. No real webcam recording, BSL-user benchmark, audible speech or production-process replacement was performed.

Public input: `https://github.com/gulvarol/bsldict/blob/master/demo/sample_data/input.mp4`. Local engineering evidence is kept outside Git under the QuietBridge cache; it is not shipped as product data.

TorchScript is deprecated in the installed Torch version, though export and loading succeeded. Fixed-shape tracing is constrained to the declared input; future variable-shape/streaming backends need their own validation.

## Consented paired capture and replay, 2026-11-04

The old paired recorder saved hand landmarks and mouth crops, not RGB frames. The old evaluator accepted only MotionModel. Those were real gaps: the BSL video backend could not be measured on its own paired recordings.

The lab now accepts `--sign-model` and checks the local artifact identity without loading TorchScript inference. **Paired → Standard sign + lips** uses its enabled labels plus REST and UNKNOWN. Video is off by default and requires explicit consent for each trial, checked by both page and API. It saves at most 240 actual resized BGR frames with their original sequence timestamps, language, model SHA-256 and manifest SHA-256. Face, body and hands are included: this is more sensitive than coordinates, and is described before capture. No camera was activated for the engineering checks below. Discard saves no recording. Limit overflow is rejected before saving either half of the pair.

Run the lab with the trusted manifest, record independent sessions, then run:

```sh
uv run python -m hackathon.evaluate_fusion \
  --profile alice --split validation \
  --sign-model /absolute/path/to/bsl-bundle/manifest.json \
  --output /absolute/path/to/new-sign-report.json
```

The evaluator replays the actual isolated-sign adapter, including whole-head ranking, fixed preprocessing and rejection gates. It does not resample already stored mouth ROIs. Separate trials use a normalized monotonic replay clock; reports retain the original capture timing. Personal-motion and standard-video cohorts are separate, including in personal motion training. Missing consent, malformed video and known cross-split session/trial/pair reuse fail visibly. Captured and current bundle identities are recorded so deliberate re-evaluation with another bundle is possible. Report files are never overwritten. The report remains diagnostic: these checks cannot establish what a pretrained model or historical corrections have seen.

Actual engineering checks used the existing public demo, not a new signer benchmark:

- After a post-hoc window scan, the **actual adapter** accepted garden on frames 80–95 at softmax `0.9144` and apple on frames 116–131 at `0.99999`. Tree on frames 128–143 scored `0.7903` and was rejected by the unchanged `0.8` gate. These selected windows are not independent trials, and softmax is not calibrated confidence. A separate diagnostic's direct-224 resize was not used as adapter evidence.
- Replayed frames 108–131 through the real loopback capture API with real MediaPipe hand/face tracking. It saved a `[24,256,256,3]` video sequence and `[24,96,96]` mouth ROIs; their raw timestamps matched exactly. The replay used source-time 25 fps, not measured live-camera throughput.
- The real lip model and real BSL bundle then read those saved files through both the evaluator function and command-line path. Lips returned `HOW MUCH`, signs returned `apple` (score `0.9643`), and fusion kept `HOW MUCH` as **conflict / confirmation needed**, rather than fabricating a combined sentence. Sign inference took about `496 ms` in that one four-thread CPU run; three-path evaluation took `1.21 s`, excluding model load and capture.
- The recording's intended sentence was a placeholder from the default profile, **not the video's verified transcript**. Its WER/exact metrics must not be reported as accuracy. It proves capture-to-replay integration and conflict handling only. Cached recordings/reports remain outside Git. No private samples, camera, speech or paid AI calls were used.

Implementation: [paired recorder](../hackathon/signs/lab.py), [consent controls](../hackathon/signs/lab.html), [three-path evaluator](../hackathon/evaluate_fusion.py), [capture tests](../hackathon/signs/tests/test_sign_video_capture.py), [evaluation tests](../hackathon/tests/test_evaluate_sign_fusion.py).

## AI text repair: actual problems and current boundary

The old optional backend still contained the hackathon welcome prompt and a two-message whitelist. It rejected normal daily sentences and could replace unrelated input with a fixed welcome. This was a real defect, not a provider-quality guess.

The fixed path:

- Formats the **current fused message**, not an earlier top lip candidate.
- Has no fixed welcome phrase. Candidate strings are data, not provider instructions.
- Allows whitespace normalization and, for an unterminated message, one final period. Changed words, names, numbers, negations and existing punctuation are rejected.
- Leaves local text intact on timeout/error. A bounded worker keeps the latest pending request. Results are tied to revision/generation; they cannot confirm or speak automatically.
- Removes an unselected AI alternative when AI configuration changes. The UI no longer retains a removed selection. Text the user already confirmed is preserved. Queued requests from old settings are skipped before sending. A request already starting or in flight cannot be recalled, but its stale result is ignored.

This intentionally **does not repair wrong recognized words or translate a sign sequence into grammar**. Calling it unrestricted AI repair would be misleading. It is now labelled formatting in Settings. Meaning-changing suggestions need a separate reviewed design and paired evaluation before being added; an LLM should not silently decide whether the user meant `I need water` or `I do not need water`.

Provider calls were tested with fake HTTP responses and timeout injection, not a paid live provider. The app's six-second timeout remains. The separate legacy `lipflow doctor` reported a configured local cleanup backend; that does not mean the main app's optional OpenAI-compatible API is enabled or connected.

Implementation: [cleanup client](../hackathon/cleanup.py), [safety checks](../hackathon/tests/test_cleanup_safety.py), [runtime checks](../hackathon/tests/test_cleanup_runtime.py).

## Whole-runtime portability checks

The prior main Windows run was red: three path-dependent assertions failed while 513 tests passed. The mouth-capture test split a Windows path using `/`; two correction tests expected portable `/` references while the writer serialized native separators. This round uses `Path.parent`/`Path.name` for filesystem assertions and persists new correction references in POSIX relative form. Existing records are not rewritten. The upload-limit tests also distinguish deterministic header-only HTTP 413 rejection from platform-dependent reset/abort during a large body upload; the health check remains.

These are specific portability fixes, not evidence that all Windows camera or BSL paths work. Remote CI is checked separately from the local Mac suite.

Local final integration run: 562 Python tests passed, 11 skipped, 34 subtests passed; 16 Node UI tests passed. The additional deterministic upload-limit check passed with the existing large-body check in a focused two-test run. Installed Torch emitted deprecation warnings; they were not suppressed. The real model/browser checks above are separate evidence.

2026-11-04 paired-capture integration: the final local suite passed **576 Python tests**, with 11 skipped and 34 subtests passed; **21 Node UI tests** passed. The real public-frame capture/replay checks above are separate from these synthetic/API tests. Windows is checked on the pushed commit separately; a previous green commit does not validate this change.

## Remaining product work, in order

1. Obtain suitable BSL model/data rights and recruit BSL users. Freeze a short useful vocabulary and record fresh paired lips/signs sessions, including REST, similar signs and unknown movements.
2. Measure positive recall, wrong signs, false emissions, correction count, negation/number errors and end-to-end latency against both single-input baselines. Do not select thresholds on the final test.
3. Improve rejection and capture guidance before enabling more vocabulary. A pretrained 1,064-class head is not a promise of 1,064 usable signs for a new signer.
4. Add continuous sign sequence alignment, non-manual grammar and fingerspelling as distinct capabilities. Do not try to recover them by merely prompting the text formatter.

Sources:

[1] [BSL-1K official repository, pretrained artifacts and vocabulary](https://github.com/gulvarol/bsl1k)
[2] [OpenHands official repository and licensing notice](https://github.com/AI4Bharat/OpenHands)
[3] [BOBSL official access agreement and restrictions](https://www.robots.ox.ac.uk/~vgg/data/bobsl/)
[4] [BSLDict official dataset licensing](https://www.robots.ox.ac.uk/~vgg/research/bsldict/index.html#data)

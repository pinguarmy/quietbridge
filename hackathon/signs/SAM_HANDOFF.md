# Current demo handoff — 1 November 2026

## Source and running state

The user explicitly authorized direct integration to `main`; commit `f25f571` was pushed. Do not keep waiting for a PR or describe the browser camera, tracking and training tools as local-only. Coordinate shared-file changes with Sam, but the user's direct-main instruction overrides the earlier PR-only plan. No Git action is part of this documentation update.

The latest automatic-camera and fast-text changes are **in progress, not deployed**. Keep that separate from the already pushed baseline and the currently running processes:

- <http://127.0.0.1:8765>: main browser-camera demo, stock lip model, Lips Only mode.
- <http://127.0.0.1:8766>: tracking and explicit dataset recording lab.

Use the browser camera and its real permission flow. Do not bypass native camera permissions or switch to another capture path to hide a permission failure. The rendered UI was checked in the actual browser: large hand camera on the left, zoomed lips on the right, 20-point outer and 20-point inner lip rings, and text below. The browser fixes allow same-origin blob media under CSP and remove double mirroring. Tracking is not a transcript or evidence of recognition accuracy.

## Custom dynamic vocabulary and dataset

| Label | User-agreed custom motion |
| --- | --- |
| HELLO | One open hand: curl fingers inward and reopen once |
| WELCOME | Both hands close in front of the chest, draw back toward the chest |
| OPENAI | Both thumbs extended; mirrored inward half-circle arcs, ending one thumb above the other |
| DDE | Index extended; move forward and back once |
| HACKATHON | Both open palms move inward from the sides and stop without clapping |
| REST | Relaxed half-fist or lowered hands; no output |

These are personal communication gestures, **not standard BSL or fingerspelling**. The requested sentence is “Hi, welcome to OpenAI DDE Community Hackathon.” Lips supply sentence candidates; gestures may provide evidence for concepts and names. Never infer that entire sentence from one gesture or force it onto unrelated lip input.

Motion-v2 has **30 training sequences and 20 test sequences**; the final extra WELCOME recording brings WELCOME's test count to four. Preserve earlier static-pose and DDE attempts separately. Do not combine different actions merely because they share a label.

The evaluated sequence classifier returned the correct result on **11/20 trials (55%)**, rejected nine, and produced no wrong labels in that small evaluation. With the **0.75 score gate**, correctness was **8/20 (40%)**. REST was silent on **3/3** test trials. These are small held-out results, not a broad zero-error or false-trigger guarantee. This classifier is **not wired as continuous gesture recognition** in the main demo; do not present classifier evaluation as continuous live acceptance.

## Lip data and selected adaptation

There are **18 labelled training clips**, split into **12 fitting clips and six validation clips**, plus **13 test clips**. The 31 private files were transferred to the home Pro and SHA-verified. Raw training media, recordings, personal weights and credentials must stay out of GitHub.

Three epochs of frontend adaptation took **57.11 seconds** on the Pro. Validation WER changed from **55.17% to 48.28%**; test WER changed from **60.00% to 55.38%**. Lower is better. Full target-sentence exact recognition remained **0/3**; the target welcome sentence was not spuriously produced on the **10 contrasting test clips (0/10)**. Do not turn that small negative sample into a general no-hallucination guarantee.

The candidate was selected but **not deployed**. Returning the selected weights to the Air is still in progress. The running main demo remains in stock mode; do not claim the new personal model is loaded.

## Speed evidence

On the Air, beam-4 encoding plus search took approximately **0.696 seconds for a 2.41-second clip**, and **1.282 seconds for a 6.14-second clip**. On the Pro, two beam-4 reads of the same short clip were approximately **0.29 seconds each**. These are measured local model operations, not upload-plus-inference latency; the environments were not identical. They do not prove the home-Pro remote path is faster end to end. Beam 2 degraded the short-clip transcription, so do not select it solely for speed.

## Direct OpenAI text suggestions

The user supplies their own key and has explicitly accepted the stated **2× default-price fast tier**. The verified official model is `gpt-6-luna`, with `service_tier: "fast"`. The intended route is **Air → OpenAI directly**, not through the Pro. Settings/key stay in runtime memory only; never persist the key to files, logs, source or the Vault.

The backend field/activation and automatic-camera work remain pending deployment. Do not say fast mode is active merely because the setting or code exists. Text suggestions must remain optional and separate from raw candidates, carry the source revision, discard stale responses, preserve negation and names, and never auto-confirm or replace already confirmed/spoken text. Send bounded text and stable concept labels, not training video or camera frames. Measure actual latency after activation; “fast” is not a timing result.

## Immediate acceptance still required

1. Finish and deploy the automatic-camera/fast changes only after checks; verify the running process loaded them.
2. Complete the private selected-weight transfer and explicitly verify stock versus adapted model selection on the Air.
3. Decide whether the small sequence classifier is useful enough for the demo; continuous gesture integration is not yet done.
4. Test complete browser-camera → candidates → user confirmation → audible speech, including unrelated input, REST, missing hands/face, timeout and disconnect behavior.
5. Rehearse three consecutive complete demos. Prepared messages must be clearly labelled fallbacks, never live transcripts.

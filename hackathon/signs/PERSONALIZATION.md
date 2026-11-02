# Personal data, model results and remaining demo work

Status: 1 November 2026. The browser-camera/tracking/training baseline was pushed directly to `main` as `f25f571`, with explicit user authorization. Latest automatic-camera and fast-text work is still in progress, **not deployed**. Detailed integration state is in [SAM_HANDOFF.md](SAM_HANDOFF.md).

## Recording and preview

Use the Air's browser camera. The real rendered layout has a large hand-tracking camera on the left, a zoomed mouth on the right, 20 outer plus 20 inner lip points, and text below. The same-origin blob CSP and double-mirroring defects were fixed and checked in the actual browser. Normal browser/native permission rules still apply; do not bypass them.

- <http://127.0.0.1:8765>: main browser-camera demo, currently **stock** lip model and **Lips Only** mode.
- <http://127.0.0.1:8766>: tracking and data-save lab. Choose Gestures or Lips, start one attempt, then explicitly finish/save it.

Hand samples contain landmarks, masks and timestamps. Mouth samples contain labelled, aligned `(T,96,96)` grayscale mouth crops. Full camera frames and microphone audio are not saved by these dataset recorders. Mouth crops are still sensitive personal visual data. Keep all recordings, personal models and keys outside GitHub.

Local private roots:

```text
~/.local/share/quietbridge/sequences/{train,test}
~/.local/share/quietbridge/mouths/{train,test}
```

Tracking shows that points were located; it does not prove that a gesture, word or sentence was recognized. Recording alone does not train a model.

## Custom gestures: fixed definitions and recorded results

These are personal symbols, **not BSL translation or letter-by-letter fingerspelling**. The target message is “Hi, welcome to OpenAI DDE Community Hackathon.” Keep lips' raw sentence candidates visible; a concept-to-name convention must not force the entire target sentence onto unrelated input.

| Label | One complete motion |
| --- | --- |
| HELLO | Open one hand, curl fingers inward and reopen once |
| WELCOME | Both hands close in front of chest, then draw back toward chest |
| OPENAI | Both thumbs extended, mirrored inward half-circle arcs, finish with one thumb higher |
| DDE | Index extended, move forward and back once |
| HACKATHON | Both open palms move inward from the sides and stop without clapping |
| REST | Relaxed half-fist or lowered hands; no command |

For a fresh attempt: remain still briefly, perform the motion once at a comfortable pace, then return to REST. Preserve the complete temporal sequence and both hands. Keep neutral, incomplete and contrasting actions as negatives. The current motion-v2 dataset has **30 train and 20 test sequences**; the last extra WELCOME trial gives that label four test examples. Earlier static-pose data—20 train, 16 test—and four early DDE attempts remain a separate baseline, not extra examples of these new motions.

The sequence classifier's held-out evaluation was:

| Measure | Result |
| --- | --- |
| Correct labels before the 0.75 score gate | 11/20 (55%) |
| Rejected trials before that gate | 9/20 |
| Wrong labels observed in this evaluation | 0/20 |
| Correct with the 0.75 gate | 8/20 (40%) |
| REST trials producing no output | 3/3 |

This small evaluation does **not** justify a broad zero-error or no-false-trigger guarantee. It is also not a continuous event test. The sequence classifier is **not wired into continuous gesture recognition** in the main demo. The four stock poses and older static-template tools are separate capabilities, not substitutes for evaluating these dynamic motions.

Do not lower thresholds against the final test set just to make the welcome demo pass. Add new independent training sessions if needed, then reserve new final tests. A clear retry or explicit candidate selection is preferable to a forced guess.

## Lip adaptation: what actually improved

The recorded mouth dataset has **18 training clips** and **13 test clips**. Training used **12 fitting clips**, reserving six of the 18 for validation. All 31 private files were transferred to the home MacBook Pro and checked by SHA; no recording was committed to GitHub.

On the Pro, **three epochs of frontend adaptation took 57.11 seconds**:

| Metric | Stock | Selected candidate |
| --- | --- | --- |
| Validation WER | 55.17% | 48.28% |
| Test WER | 60.00% | 55.38% |

WER is word error rate; lower is better. This is a measured improvement on these clips, **not reliable full-sentence recognition**: the complete welcome sentence still matched exactly on **0/3** target test clips. The target sentence was not falsely produced on the **10 contrasting clips (0/10)**; that limited sample is not a general hallucination guarantee.

The candidate was selected, but **has not been deployed**. Its private weight transfer back to the Air is ongoing. The running main server is still using stock weights. Do not describe the adaptation as live merely because training finished.

Further data should target actual errors in these results, not just repeat a single sentence indefinitely. Keep validation for model selection and an untouched final test session. Preserve old weights and show which model is loaded. Rehearsed-phrase success must not be described as open-vocabulary lip-reading accuracy.

## Machine split and speed evidence

Use the Air for camera capture, hand tracking, UI, confirmation and audible output. The home Pro has a separate `~/code/quietbridge-demo` environment with PyTorch MPS and the vision assets; it has now performed the personal lip adaptation. Training on the Pro does not require that computer at the venue.

Measured beam-4 work:

- Air: a 2.41-second clip took about **0.696 seconds** for encoding plus search; a 6.14-second clip took about **1.282 seconds**.
- Pro: two reads of the same short clip took about **0.29 seconds each**, total local read, **excluding transmission**.

The environments were not identical, and these operations are not the complete release-to-visible-text route. The measurements do not prove remote Pro inference is faster after network upload/return. Beam 2 degraded the short-clip text; do not choose it merely to reduce elapsed time. A web tunnel alone does not forward an Air camera or return the Pro's native speech to the Air.

## Text optimization: direct API, not model training

The user now supplies an OpenAI key for an **Air-direct** request route, not a Pro relay. Official `gpt-6-luna` and `service_tier: "fast"` were checked; the user explicitly accepted the stated **2× default price**. Key/settings are runtime-memory-only: no key in source, files, logs, shared media or the Vault.

Activation/backend and automatic-camera changes are still being finished and are **not deployed yet**. Keep local/raw results usable while optional, revision-bound text suggestions run. Never auto-confirm, overwrite confirmed/spoken text, or turn a weak gesture into a fabricated welcome line. Text cleanup is separate from learning a person's mouth or gestures.

## What remains before calling the demo ready

- Complete selected-weight transfer and verify explicit personal-model loading on the Air.
- Finish, check and activate the pending automatic-camera/fast changes; verify the actual running process and browser behavior.
- Evaluate or improve dynamic gestures on new data, then separately wire and test continuous events if used.
- Verify the full route with real inputs: candidates, conflicts, confirmation and audible speech, plus REST, unrelated sentences, missing modalities and network/API failure.
- Complete three consecutive rehearsals. Keep any prepared message clearly labelled as a fallback, never a live transcript.

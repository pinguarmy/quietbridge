# QuietBridge recognition audit

Date: 2026-11-03. Scope: current local application, personal recording workflow, recognizers, fusion and offline evaluation. Code inspection, pure-function reproductions, fake-worker integration and rendered browser checks support these findings. No new human recognition benchmark was collected.

## Conclusion

QuietBridge is a personal communication aid for silent English speech, personal gestures and an explicit experimental standard-sign video backend. The BSL-1K path now loads real external weights for isolated word-level windows alongside lips; setup and the limits of actual verification are in [standard-sign support](sign-language-support.md). It is not a full BSL, ASL or multilingual sign-language translator. Splitting the work between lip and hand models is sensible, but does not guarantee higher accuracy. The system must show a gain over both single-input baselines on the same independent recordings before claiming one.

BSL has its own grammar and uses handshapes, facial expressions and body language. ASL is also a distinct natural language, not English performed with the hands. Sign languages are not universal.[1][2] A list of gesture names cannot substitute for language-specific sequence recognition and translation.

## What actually works today

| Input | Implemented capability | Boundary |
| --- | --- | --- |
| Stock hand model | Four mapped poses: thumb up, thumb down, open palm and pointing up | Not a BSL/ASL vocabulary or fingerspelling decoder |
| Personal motion model | Matches one bounded movement against personal hand-landmark templates; rejects REST and unmatched inputs | One label per explicit recording window; no continuous sign-language grammar |
| Explicit sign-video backend | Loads a local hashed TorchScript bundle, including exported BSL-1K 1,064-class weights; shares capture window with lips | Engineering smoke only, selected concepts require confirmation, no trained REST/unknown or sentence grammar |
| Lip reader | Ranked text candidates from aligned mouth crops, with optional personal visual/language weights | Ambiguous visual speech; candidate scores are not calibrated probabilities |
| Fusion | Keyword reranking and a single user-declared template slot | No learned word-level alignment or general translation |
| Speech | Local text-to-speech after explicit user action | Does not establish that recognition was correct |

Evidence: [stock pose map](../hackathon/signs/recognizer.py#L43-L58), [one-event motion classification](../hackathon/signs/recognizer.py#L250-L296), [hand-only features](../hackathon/signs/motion.py#L143-L191), [fusion rules](../hackathon/fusion.py).

The personal motion input contains two sets of 21 hand landmarks and presence masks. It does not encode body posture or facial grammar as sign-language features. The separate lip model is not a non-manual sign-language grammar model. Supporting full BSL would require a BSL-specific dataset, temporal language model, non-manual features, fingerspelling handling where needed, and validation with BSL users. ASL and other sign languages need their own validated language paths. Full translation remains future work. The explicit isolated-sign backend is now implemented and distinguished from personal gestures in Settings; it does not provide those missing language capabilities.

## First-use and everyday communication, 2026-11-05

Independent onboarding reviews exposed message/profile edit loss, a separate-server Personalize prerequisite, sustained-hold recording, missing New message, unclear active input and mobile live reordering. The current pass preserves unfinished text through navigation and incoming results, makes typing visible, adds a reopenable short guide and revision-checked New message, uses a single-press capture toggle, and keeps message/camera order fixed. Personalize opens an owned on-demand loopback recorder with lazy tracking. Form edits survive add/remove, profile creation is name-first with keyboard focus, and local training capture has explicit consent and camera-off control.

The frozen criteria, independent scores, fixes and remaining verification limits are in [the usability review](ux-review.md). These changes improve engineering usability; they do not measure recognition accuracy or replace testing with intended users.

## Paired standard-sign capture and evaluation, 2026-11-04

Found and closed a data-path gap: paired recording stored landmarks/mouth crops, while the BSL backend needs RGB video; evaluation only ran personal MotionModel. Optional standard-sign recording now has per-trial video consent before camera access, a bounded full-frame buffer and model/manifest identity. Default capture remains coordinate-only. Video pairs replay the actual sign backend alongside lips and fusion; personal-motion training/evaluation excludes them.

Real public-frame replay passed through MediaPipe, paired files and both real vision models. Apple was accepted; conflicting lips remained confirmation-needed rather than being overwritten. Post-hoc selected public windows and a placeholder intended sentence are **not accuracy evidence**. No threshold tuning, private uploads or webcam activation. Commands and exact observations are in [the support report](sign-language-support.md#consented-paired-capture-and-replay-2026-11-04).

## Standard-sign and runtime follow-up, 2026-11-03 evening

Standard BSL word-level input is now an explicit external video-model backend, not a relabelled personal gesture. Actual BSL-1K weights were loaded and converted, with exact logits equivalence on one public clip; the public whole-window sample was low-score and produced no text. The real lip and sign models loaded together in an isolated server. This establishes engineering integration, not useful signer accuracy or permission to distribute the external model. See [setup and evidence](sign-language-support.md).

The optional AI client still forced a fixed hackathon welcome phrase and rejected ordinary sentences. That stale prompt/whitelist is removed. Formatting now preserves all original words and symbols, uses the current fused message, and discards unselected AI alternatives after settings changes. It does not repair wrong words. The full suite also covers disabled/stale UI choices, real TorchScript bundle conversion, short/gapped/overflowed sign capture and chronological hand-tail draining on camera stop.

The hand worker previously finished after processing only one queued tail packet, and camera stop explicitly discarded another queued frame. It now drains all frames through the shared window before classifying; a gated worker regression checks every timestamp. Worker stop events are captured when each thread is created rather than read from later mutable runtime state.

## Follow-up audit, 2026-11-03 afternoon

The follow-up reproduced four concrete failures and implemented bounded fixes:

- Two-window alternatives could truncate the message. `hello/yellow` then `world/word` exposed `yellow` instead of `yellow world`. Both automatic and explicit windows now rebuild complete-message single-window alternatives. A separate regression ensures a displayed fused proposal is still allowed by confirmation.
- The language promotion gate accepted NaN and negative error rates. It also accepted a stock-visual report after the visual model or language candidate contents changed. The gate now rejects invalid rates and requires matching weight hashes. Re-evaluation uses a new report filename while preserving the old evidence.
- Promotion saved files but left the running recognizer unchanged. Personalize now distinguishes saved models from loaded models and links to the idle-only Reload current models action. Real browser checks in isolated rehearsal mode confirmed the displayed model hash changes on reload. No real checkpoint was loaded in that check.
- Known held-out correction sources could enter language training. The correction writer and historical-record reader now exclude them and contradictory metadata. Unknown daily sources remain consent-based and explicitly unverified. Training surfaces source coverage and warnings, not an evaluation-independence guarantee.

The fixes prevent these demonstrated errors. They do not establish improved camera recognition, independent model provenance for arbitrary checkpoints, or statistical significance. Reusing one final-test session to choose many candidates remains an evaluation limitation.

Final follow-up verification: 525 Python tests passed, 11 skipped, 34 subtests passed; 13 Node UI checks passed. The browser reload check used isolated rehearsal data and observed a changed language-model hash without changing the profile. Regression readers/workers were fake; no new human accuracy dataset, camera capture or real checkpoint inference was used.

## Findings, in impact order

### 1. Default camera flow did not start personal gesture windows. Fixed.

Opening a browser camera session hid the recording control. Automatic mouth chunks deliberately did not start the explicit motion recognizer. A user could therefore see working hand tracking while their custom gesture was never classified in that flow.

The default personal motion mode now keeps a Hold control. Lips and one gesture share its start/end boundary; the completed window is fused before its text enters the session. Gesture-only results survive stopping the camera. Pure lips and experimental automatic-motion input retain their existing automatic behavior. Worker tests cover combined and gesture-only start/release/finish paths. They do not prove physical-camera recognition quality.

Evidence: [session fusion](../hackathon/runtime.py), [cross-module regressions](../hackathon/tests/test_runtime_merge.py), [UI state checks](../hackathon/tests/test_ui_flow.cjs).

### 2. Candidate reranking could drop a negation or number. Fixed for the reproduced paths.

With WATER and a template `I need {gesture}`, candidates `I do not need water` and `I need` previously produced `I need water` as the default, marked as agreement. Ordinary keyword reranking could also prefer `I need water` over `I do not need water` or `I need two waters`.

Templates now fill only the leading lip candidate's exact static context. A matching lower candidate cannot erase the leading candidate. Ordinary reranking preserves the leading candidate when the proposed replacement changes a detected negation, number or listed safety-critical word. These cases become a conflict, and alternatives require a choice. This is a specific rule-based safeguard, not complete semantic verification.

Evidence: [fusion](../hackathon/fusion.py), [regression examples](../hackathon/tests/test_integration.py).

### 3. Paired evaluation omitted the gesture-only baseline. Fixed.

Reports previously compared only lips-only and fusion. They now include all three paths, gesture positive counts/recall, wrong labels, and REST/unknown false-emission counts and rates. Empty denominators remain null. Default evaluation weights now follow promoted profile paths, with hashes and explicit override support.

These are offline diagnostics. A gesture can supply only a noun while the reference is a whole sentence; standalone gesture WER should be read alongside gesture classification recall. Per-window false-emission rates do not measure false triggers per hour of continuous use. The evaluator does not establish that a weight file never saw the test set.

Evidence: [evaluator](../hackathon/evaluate_fusion.py), [paired-report tests](../hackathon/tests/test_evaluate_fusion.py).

### 4. Correction learning could consume held-out sources. Known-source path fixed in follow-up.

The follow-up audit exercised a correction pointing to a validation/test NPZ. These known sources could reach language training. Recording now rejects them, and the reader excludes historical held-out records and contradictory metadata. The training report records source coverage, skips and known session/trial IDs. A repeated prompt across independent sessions is not itself the same recording; the checks use recording provenance, not just matching transcript strings.

Corrections without sources remain available for explicitly consented daily learning and are marked unknown. This does not establish held-out independence. Freeze a new held-out session after the candidate and correction set are fixed; do not learn from it. No evidence here establishes that existing private data was contaminated.

Evidence: [correction record and reader](../hackathon/corrections.py), [correction ingestion and coverage](../hackathon/signs/train_language.py), [recording isolation](../hackathon/signs/train_language.py#L49-L80).

### 5. Model promotion checks are safety filters, not proof of benefit. Open.

The language gate rejects reported aggregate WER regression and mismatched/missing target coverage. Follow-up reproduced NaN and negative WER passing the old gate; finite nonnegative numbers are now required. A stock-visual report also remained promotable after visual weights or candidate content changed. The report is now bound to both content hashes. Stale or legacy unbound reports require a new evaluation, preserving old reports. These checks still accept ties and do not provide uncertainty bounds. The paired report is diagnostic and does not enforce training provenance. Repeatedly selecting candidates against one final-test set also stops that set being a clean independent final test.

Keep validation for all tuning and selection. Use a fresh final test once for the frozen configuration, report the number of people, sessions, clips and errors, and do not call synthetic tests accuracy evidence.

Evidence: [language gate](../hackathon/model_lifecycle.py#L65-L102), [promotion eligibility](../hackathon/model_lifecycle.py#L259-L291).

## How lips and hands should cooperate

Use one bounded expression with independent evidence, not two systems voting on a guessed sentence.

1. Capture the same time window. Both recognizers retain their own quality signals and rejection decisions.
2. Lips propose sentence candidates. Gestures provide a known concept, a complete configured phrase, or one declared slot. Neither is required to fabricate missing input.
3. If the top lip text is `I need` and WATER is accepted with `I need {gesture}`, propose `I need water` for review.
4. If lips say `I do not need water`, preserve the negation. If lips say `I need tea`, do not silently replace tea with water. Show the disagreement and let the user choose or type.
5. Optional AI wording stays separate from recognition. It may offer wording, but cannot certify intent or replace the selected message by default.
6. Speak only the exact text the user chose, bound to its revision.

The follow-up audit reproduced two windows `hello/yellow` and `world/word`: selecting `yellow` could drop the entire second window. Alternatives now replace one window while retaining all other selected windows, producing `yellow world` and `hello word`. Automatic lip windows retain their n-best alternatives too. The displayed fusion proposal remains confirmable. This is bounded window-level recombination, not a learned word-aligned model or every possible combination.

## Simplest user journey

Daily use: Start camera → hold for one personal gesture and its words, or use automatic pure-lip input → Finish message → review/edit → Speak. Typing works without the camera. No training is required to try stock inputs or type.

Personalization: choose a few useful phrases → record practice → record a separate check session on another occasion → final check → explicitly activate a candidate. Advanced labels, templates, session IDs, capture statistics and provider settings stay out of the daily page.

Model saving was also misleading: promotion changed files while the running reader kept old weights. Personalize now says Save for main app, and Settings exposes Reload current models through the existing idle-only profile reload. The UI reports warmup and failure instead of claiming loading already succeeded. Explicit startup paths continue to override profile files. Reload clears the draft.

Two limitations remain in setup: the app and recording lab are separate local server processes, and training still needs independent recording sessions. Hiding those requirements would make the UI simpler on paper but the product less reliable. A single launcher and validated recording guidance are the next UX tasks, ahead of adding more model controls.

## What would establish an accuracy improvement

Freeze the vocabulary, weights, beam, thresholds and correction dataset. On a new paired session, compare lips-only, gestures-only and fusion using the exact same inputs. Include ordinary resting motion, unseen movements, negation, numbers and names, not just positive prompted examples.

Report WER and exact intended message, gesture positive recall, REST/unknown false emissions with denominators, critical-word errors, correction/confirmation count and end-to-text latency. Use separate continuous recordings to measure false triggers per hour. Report per-person/session results and uncertainty, not just one pooled percentage. Fusion should improve on the strongest single-input path without increasing critical errors. If it does not, ship the simpler single-input path for that user.

For developer/agent division, keep capture/runtime, hand recognition, lip recognition, UI and evaluation in separate write scopes. Agree on timestamp, candidate, rejection and model-identity contracts first. Have an integration owner review cross-module behavior and an independent evaluator hold the test set. More agents or more passing unit tests do not establish a recognition gain.

## Sources

[1] [British Deaf Association: help and resources](https://bda.org.uk/help-resources), retrieved 2026-11-03. Used for BSL grammar, non-manual components and language differences, not its historical user statistics.

[2] [NIDCD: American Sign Language](https://www.nidcd.nih.gov/health/american-sign-language), updated 2021-10-29, retrieved 2026-11-03. Used for ASL's independent grammar and the distinction between ASL and BSL.

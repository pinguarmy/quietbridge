# QuietBridge recognition audit

Date: 2026-11-03. Scope: current local application, personal recording workflow, recognizers, fusion and offline evaluation. Code inspection, pure-function reproductions, fake-worker integration and rendered browser checks support these findings. No new human recognition benchmark was collected.

## Conclusion

QuietBridge is a personal communication aid for silent English speech and a small configured gesture vocabulary. It is not a full BSL, ASL or multilingual sign-language translator. Splitting the work between lip and hand models is sensible, but does not guarantee higher accuracy. The system must show a gain over both single-input baselines on the same independent recordings before claiming one.

BSL has its own grammar and uses handshapes, facial expressions and body language. ASL is also a distinct natural language, not English performed with the hands. Sign languages are not universal.[1][2] A list of gesture names cannot substitute for language-specific sequence recognition and translation.

## What actually works today

| Input | Implemented capability | Boundary |
| --- | --- | --- |
| Stock hand model | Four mapped poses: thumb up, thumb down, open palm and pointing up | Not a BSL/ASL vocabulary or fingerspelling decoder |
| Personal motion model | Matches one bounded movement against personal hand-landmark templates; rejects REST and unmatched inputs | One label per explicit recording window; no continuous sign-language grammar |
| Lip reader | Ranked text candidates from aligned mouth crops, with optional personal visual/language weights | Ambiguous visual speech; candidate scores are not calibrated probabilities |
| Fusion | Keyword reranking and a single user-declared template slot | No learned word-level alignment or general translation |
| Speech | Local text-to-speech after explicit user action | Does not establish that recognition was correct |

Evidence: [stock pose map](../hackathon/signs/recognizer.py#L43-L58), [one-event motion classification](../hackathon/signs/recognizer.py#L250-L296), [hand-only features](../hackathon/signs/motion.py#L143-L191), [fusion rules](../hackathon/fusion.py).

The personal motion input contains two sets of 21 hand landmarks and presence masks. It does not encode body posture or facial grammar as sign-language features. The separate lip model is not a non-manual sign-language grammar model. Supporting full BSL would require a BSL-specific dataset, temporal language model, non-manual features, fingerspelling handling where needed, and validation with BSL users. ASL and other sign languages need their own validated language paths. This is a proposed future project, not a switch already present in Settings.

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

### 4. Correction learning does not establish held-out provenance. Open.

The recording loaders reject shared recording session/trial IDs across splits. However, explicitly approved correction text is added to language training separately. Source references are optional, and the learning reader does not enforce their split. A user could copy a validation/test correction into learning and contaminate later evaluation. No evidence here establishes that existing private data is contaminated.

Before reporting personal-model gains, freeze a new held-out session after the candidate and correction set are fixed. Do not learn from that session. A later implementation should attach source session/split to corrections and exclude held-out sources. A repeated prompt across independent sessions is not itself the same recording; distinguish transcript overlap from actual test exposure.

Evidence: [correction record and reader](../hackathon/corrections.py#L18-L42), [correction ingestion](../hackathon/signs/train_language.py#L117-L124), [recording isolation](../hackathon/signs/train_language.py#L49-L80).

### 5. Model promotion checks are safety filters, not proof of benefit. Open.

The language gate rejects a reported aggregate WER regression and target regression when both target metrics are present. It accepts ties and does not provide uncertainty bounds. The paired report is diagnostic and does not enforce training provenance. Repeatedly selecting candidates against one final-test set also stops that set being a clean independent final test.

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

The current window implementation retains original candidates, but does not rebuild every earlier alternative with every later chunk. For a long message, some alternatives represent an earlier prefix. Do not describe this as full word-level multi-window fusion. Short expressions and direct editing are the usable path today.

## Simplest user journey

Daily use: Start camera → hold for one personal gesture and its words, or use automatic pure-lip input → Finish message → review/edit → Speak. Typing works without the camera. No training is required to try stock inputs or type.

Personalization: choose a few useful phrases → record practice → record a separate check session on another occasion → final check → explicitly activate a candidate. Advanced labels, templates, session IDs, capture statistics and provider settings stay out of the daily page.

Two limitations remain in setup: the app and recording lab are separate local server processes, and training still needs independent recording sessions. Hiding those requirements would make the UI simpler on paper but the product less reliable. A single launcher and validated recording guidance are the next UX tasks, ahead of adding more model controls.

## What would establish an accuracy improvement

Freeze the vocabulary, weights, beam, thresholds and correction dataset. On a new paired session, compare lips-only, gestures-only and fusion using the exact same inputs. Include ordinary resting motion, unseen movements, negation, numbers and names, not just positive prompted examples.

Report WER and exact intended message, gesture positive recall, REST/unknown false emissions with denominators, critical-word errors, correction/confirmation count and end-to-text latency. Use separate continuous recordings to measure false triggers per hour. Report per-person/session results and uncertainty, not just one pooled percentage. Fusion should improve on the strongest single-input path without increasing critical errors. If it does not, ship the simpler single-input path for that user.

For developer/agent division, keep capture/runtime, hand recognition, lip recognition, UI and evaluation in separate write scopes. Agree on timestamp, candidate, rejection and model-identity contracts first. Have an integration owner review cross-module behavior and an independent evaluator hold the test set. More agents or more passing unit tests do not establish a recognition gain.

## Sources

[1] [British Deaf Association: help and resources](https://bda.org.uk/help-resources), retrieved 2026-11-03. Used for BSL grammar, non-manual components and language differences, not its historical user statistics.

[2] [NIDCD: American Sign Language](https://www.nidcd.nih.gov/health/american-sign-language), updated 2021-10-29, retrieved 2026-11-03. Used for ASL's independent grammar and the distinction between ASL and BSL.

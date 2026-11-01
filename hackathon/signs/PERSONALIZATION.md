# Personal gesture data and demo plan

## Current target: custom dynamic gestures plus labelled lip clips

The requested demo message is **“Hi, welcome to OpenAI DDE Community Hackathon.”** Peter's latest definition uses custom motions, not standard BSL or letter-by-letter DDE spelling:

| Label | One complete recorded motion |
| --- | --- |
| HELLO | Open one hand, curl fingers inward and reopen once |
| WELCOME | Both hands together in front of the chest, draw back toward the chest |
| OPENAI | Both thumbs extended, mirrored inward half-circle arcs, finish with one thumb above the other |
| DDE | Index extended, move forward and back once |
| HACKATHON | Both open palms move inward from the sides and stop without clapping |
| REST | Relaxed half-fist or lowered hands, no output |

Use <http://127.0.0.1:8766> → **Gestures**. Start still for half a second, do the motion once, return to REST for half a second, then Finish & save. Guided sessions use `peter-motion-v2-*`; each label gets five independent training trials, then three new test trials. The dynamic recognizer is not trained yet. The skeleton means tracking, not successful concept recognition. These custom gestures must not be presented as general BSL translation.

Earlier static data is preserved separately: 20 train trials and 16 test trials for the old thumb/index/palm/neutral conventions, plus four older DDE attempts. Do not mix those pose-v1 and DDE-fingerspelling files with motion-v2 data just because a label name matches.

Use **Lips** on the same page to record the displayed prompt silently. The recorder saves aligned 96×96 grayscale mouth crops with the exact text/session/split; it does not decode text or train automatically. The six prompts include the target line, two shorter components and three contrasting sentences. The initial guided pass is three recordings per prompt for training, followed by two fresh recordings per prompt for test. Preserve this separation. Mouth crops contain personal visual information even though full-face video and microphone audio are not saved. Data is under `~/.local/share/quietbridge/mouths/{train,test}`; hand sequences are under `~/.local/share/quietbridge/sequences/{train,test}`.

The lab is separate from Sam's integrated runtime and does not enable AI suggestions or remote lip decoding. Changes to Python capture routes take effect only after an idle service restart; do not interrupt an active user trial. No improvement in real recognition is claimed before independent evaluation.

The static collector and template classifier described below are useful for the existing four-pose baseline only. They **cannot train or validate this BSL sentence**: collection currently keeps one hand and discards motion and facial information. Do not ask Peter to record the welcome sentence through this collector or claim that more static samples will make it a BSL translator.

Before collecting BSL training data, obtain a checked video/reference from a fluent BSL signer or qualified interpreter, including how the names OpenAI, DDE and Hackathon are expressed or fingerspelled. Agree the complete expression rather than joining English dictionary signs word by word. Preserve two hands, body-relative motion and relevant nonmanual cues in a separate consented sequence dataset. Unknown input and incomplete sequences must not trigger the welcome phrase.

A feasible limited demo can recognize **one known, independently validated BSL welcome message** from Peter, with neutral, incomplete and contrasting sequences held out. It must be described as known-message recognition, not unrestricted BSL translation. English lip reading and BSL can express the same message without word-for-word timing alignment.

For lips, prioritize both the full welcome line and short components, with separate final-test recordings. Displayed proper-name spelling can be a visible user-confirmed correction; never force the fixed welcome line onto unrelated lip input. Suggested chunking is provisional and must be tested, not treated as an accuracy improvement already achieved.

### Welcome-message recording proposal, after linguistic review

- **Mouth training:** ten independent full-line recordings, five “Hi, welcome to” recordings, five “OpenAI DDE Community Hackathon” recordings, plus ten distinct contrasting sentences. This is 30 training clips, not a guarantee that the phrase becomes recognizable. Record the full line at a natural pace; do not force it into a two-second window.
- **Mouth validation:** a separate session with five new full-line trials and five contrasting trials. Use this for comparing stock/adapted models, not gradient updates.
- **Mouth final check:** another five target and five contrasting trials after model selection. Count exact target recognition, transcription errors and false welcome outputs on contrasts. Keep raw model candidates visible alongside any explicitly confirmed spelling correction.
- **BSL training:** only after the complete expression is checked, start with ten independent complete sequences plus ten neutral, incomplete or different-message sequences. Preserve temporal order and both hands. This is a proposed pilot size, not evidence of sufficiency.
- **BSL validation and final check:** use separate sessions, initially five target and five non-target sequences each. Freeze the expression and decision threshold before final evaluation. A whole-message template match does not demonstrate that the system understood each letter or translated arbitrary BSL.
- Do not require English mouthing to accompany BSL. Nonmanual BSL mouth patterns have linguistic meaning and should not be overwritten to suit the English lip reader. Show independent lip-only and BSL-only examples first; demonstrate combined input only if the chosen expression supports it naturally and testing confirms it.

Verified starting references: [UCL BSL SignBank](https://bslsignbank.ucl.ac.uk/) provides dictionary and fingerspelling entries; [Signature's fingerspelling resources](https://www.signature.org.uk/students/learning-support/fingerspelling/) explain two-handed BSL spelling and uses for names/new words; [BDA's BSL overview](https://bda.org.uk/bsl-deaf-culture/) describes its distinct grammar, body language and facial expression. These sources do **not** validate this complete welcome sentence or establish a unique sign for OpenAI, DDE or Hackathon. A fluent signer/interpreter still needs to check the reference expression. Do not claim a term has no BSL sign merely because one dictionary search did not find it.

## Reliability-first welcome demo

Make the interaction forgiving, not the success claims. Prefer short, repeatable custom concept gestures and one explicit confirmation over a long unfamiliar signing sequence. Slower movement and clear pauses are acceptable; keep independent negative examples so relaxed timing does not cause false welcome messages.

- Start with separate mouth captures for “Hi, welcome to” and “OpenAI DDE Community Hackathon”, then compare against a natural full-line capture. Do not assume chunking is better until measured.
- Show raw candidates beside any event-vocabulary assisted spelling. Preserve “OpenAI”, “DDE”, “Community Hackathon” as explicit configured context, not as proof those words were visually recognized.
- Use a deliberate capture/release action for a custom sequence and show alternatives when uncertain. Do not force a guess or auto-speak. Retrying or choosing a displayed option is a valid assisted workflow, not raw recognition success.
- Do not tune the final-test set or lower thresholds simply to make the rehearsed input pass. A model must also decline neutral, incomplete and contrasting input.
- Keep a clearly labelled, user-selected prepared-message fallback if the live path fails; never display it as a live transcript.

The first saved Air DDE sequence contains 117 frames over about 15.18 seconds, with at least one detected hand in 65 frames and **zero frames with both hands detected**. This is a tracking-quality warning, not proof the sign was wrong. Keep the file; inspect the visible skeleton and frame both hands before treating subsequent trials as two-hand examples. Do not replace this diagnostic with an accuracy claim.

## Recommended machine split

Use the MacBook Air camera for recording and live input. Run gesture inference, UI confirmation and speech on the Air. Train the lip model on the home MacBook Pro, then return the selected personal weights to the Air. Gesture template training is inexpensive too; running it on the Pro is convenient, not necessary for speed.

Do not stream the whole webcam merely to move a small hand classifier off the Air. Consider remote lip decoding only after comparing **release-to-visible-text** on the real network. The current server is local-only: an SSH web tunnel does not redirect the Air camera or play the Pro's `say` output on the Air.

## What will be learned

The personal gesture path learns small templates of **your hand shape**, using MediaPipe's 21 hand landmarks. It does not fine-tune Google's `.task` model or learn a general sign language. It retains the same four output labels and temporal event contract.

The collector saves only numeric landmarks/features and labels. It does not save face video, raw images, microphone audio or speech. Treat these files as personal data nevertheless; they stay outside Git.

## Record training trials first

Use the final demo camera, approximately the same distance and normal front lighting. Pick a comfortable, clearly distinguishable pose for each label. Prefer one hand in view and keep the same hand throughout the first recording round.

From the repository root, in a terminal with Camera access:

```sh
.venv/bin/python -m hackathon.signs.assets --download

for label in THUMB_UP THUMB_DOWN OPEN_PALM POINTING_UP NEUTRAL; do
  .venv/bin/python -m hackathon.signs.collect \
    --split train --session peter-air-train-1 --label "$label" --trials 10
done
```

For every independent trial:

1. Show the selected pose, press Space, wait for the one-second countdown.
2. Hold for the two-second recording. A positive trial requires at least eight frames with exactly one detected hand.
3. Lower or relax the hand, then start the next trial. Vary position and angle slightly between trials; do not turn a single long hold into ten nominally independent examples.
4. Q quits and discards an incomplete trial.

For `NEUTRAL`, deliberately show an ordinary relaxed hand, a fist, fingers partly bent, or movements that should **not** express a message. At least five neutral training trials must contain a detected hand. Empty/no-hand trials are useful checks but cannot teach a hand classifier what neutral looks like.

## Record an independent test session

Take a short break, reset the camera position, and record a separate session. Do not copy training files into the test directory.

```sh
for label in THUMB_UP THUMB_DOWN OPEN_PALM POINTING_UP NEUTRAL; do
  .venv/bin/python -m hackathon.signs.collect \
    --split test --session peter-air-test-1 --label "$label" --trials 5
done
```

This first plan is **50 training trials + 25 test trials**. Recording holds alone total 150 seconds, plus countdowns and repositioning. Completion time depends on detection and retakes; this is not a measured recording-time promise. The minimum training requirement is five nonempty independent trials for each of the five labels, not thousands of near-identical frames.

Default data location: `~/.local/share/quietbridge/gestures/{train,test}/*.npz`. Keep test-session IDs out of training. The collector and trainer reject sessions that appear in both splits.

## Train and evaluate

```sh
.venv/bin/python -m hackathon.signs.train \
  --data "$HOME/.local/share/quietbridge/gestures" \
  --output "$HOME/.cache/quietbridge/personal_gestures.npz"
```

The trainer builds one median template per trial. Distance thresholds use training trials only; test trials cannot tune them. A report is written beside the model. Existing output is not replaced unless explicitly requested with `--overwrite`; use a new filename for each experiment to keep a recovery path.

Read per-label held-out results, rejection/missed gestures and neutral false triggers, not just one aggregate percentage. Static classifier accuracy is not proof of a stable live event. Before enabling this model, repeat the held-out poses through the live recognizer and complete three consecutive integrated demos. If the personal model performs worse than stock, keep stock; extra data is not guaranteed to improve it.

To opt in after acceptance:

```sh
export QUIETBRIDGE_PERSONAL_GESTURES="$HOME/.cache/quietbridge/personal_gestures.npz"
.venv/bin/python -m hackathon.signs.demo --camera 0 --seconds 60
```

The same environment variable is read when Sam's server constructs `SignRecognizer`. If the personal model rejects a hand, the recognizer does not secretly fall back to a canned category. `unset QUIETBRIDGE_PERSONAL_GESTURES` restores stock recognition. Scores remain heuristic similarity values, not probabilities.

## Lip adaptation is a separate task

The old Lipflow app already records labelled mouth clips and can fine-tune the visual frontend. Record fresh English sentences, not just repeated demo lines; keep an independent final evaluation session. Training only on a rehearsed phrase can improve that phrase without demonstrating general lip reading.

Sam's current `hackathon/lips_adapter.py` constructs `LipReader(..., personal=False)`. Therefore training a personal lip model alone will **not** affect the new demo. Sam must add an explicit opt-in personal-model selection, display which model is loaded, and compare stock versus adapted results on the same untouched mouth clips. The old `train_on_face()` also trains a personal language model and uses a deterministic six-clip holdout, so repeated training rounds must not quietly turn those clips into training data.

Recommended first mouth-data round: 24 different labelled 2–4-second clips, with six reserved before training. A later final session should contain at least five new sentences not used for fitting or model selection. Compare word error rate, negation/number errors and release-to-text time. Save only an improvement, retain the old weights, and do not describe rehearsed-phrase success as open-vocabulary accuracy.

## Current limits

No new personal trials have been collected by this implementation. No trained personal model or accuracy improvement is claimed. Air camera access was denied in the earlier harness process; launch collection from a host explicitly allowed in macOS Camera settings. Never reset macOS privacy databases or disable checks to make a demo pass.

# Real-clip recognition experiments (stock Auto-AVSR model, two sessions of one speaker)

Run on 2026-11-20 on the development box (4 shared CPU cores, 15 GB RAM, no GPU), repository
HEAD moved from `69d4f74` to `cf96770` during the run (`vsr.py`, `train_vsr.py`, `train_lm.py` and `espnet/` were
identical across those commits; `face.py` only changed its default model path, which the run already
used from the working tree; recipes were run from the committed text; `quietbridge/phrase_match.py` evaluated at
`fe63671`, sha256 `87155713…`). Other agents were editing product files during the run; every script
records `git status` in its JSON output. Scripts: `scripts/experiments/`. Derived data (segments,
caches, decodes, model deltas) stayed in a scratch directory and is not in the repository.

## Summary

What was measured: the stock lip model (beam 4 as in `quietbridge/lips_adapter.py`) on 49 sentence
segments (308 s of video, 895 words) cut from two public-domain weekly addresses of the same speaker
recorded ten months apart (session A = 2016-03-12, session B = 2017-01-07); n-best oracles; the
product's phrase suggestions (`phrase_match`) and direct phrase scoring under the lip model; capture
degradations; the committed speaker-adaptation and personal-LM recipes with A as training data and B
as the once-only test.

Results that should change what the project does next:

1. **Frame rate and gaps matter more than anything we can train.** On B, dropping the camera from
   ~30 to 12 fps raises WER from 13.6 % to 22.2 % (+8.7 points, paired 95 % CI +4.2 to +14.1), 8 fps
   to 45.4 %, and one 400 ms gap to 21.8 % (+8.2, CI +4.2 to +13.1). The product should measure and
   enforce capture quality (≥ 20–25 fps delivered, no gaps ≥ 200 ms) before it trains anything.
2. **Scoring the person's saved phrases directly under the lip model works far better than matching
   text.** Top-1 sentence exact rate on B is 13.6 % (3/22); the text matcher (`phrase_match`, shipped
   thresholds) offers the right saved sentence in 76 % of the in-list cases where top-1 was wrong, but
   it offers a one-word-different saved phrase 100 % of the time when the intended one is not saved.
   Teacher-forced attention+CTC(+LM) scoring picked the intended sentence over six one-word variants
   in 22/22 B segments, and its length-normalised margin separates in-list from not-in-list
   (threshold chosen on A: 100 % accepted / 0 % false on B for distinct sentences; AUROC 0.89 when
   only one-word variants are saved). Cost: ~45 ms per phrase per segment on one CPU thread.
3. **The personal LM recipe made recognition worse, and for a reason unrelated to personal data.**
   Adding the A-trained LM (weight 0.2 on top of the stock 0.3) raised B WER by 3.3 points (CI +0.7
   to +6.5); adding an *unchanged copy of the stock LM* at the same weight did the same (17.1 % vs
   16.9 %). The extra LM weight, not personalisation, is the effect.
4. **Speaker adaptation with the committed recipe gave no reliable gain on this speaker.** Validation
   on A selected the stock model (every epoch raised validation loss); the recipe's default 6-epoch
   output, tested once on B as a declared diagnostic, gave −1.6 WER points [−5.5, +1.4]. It cost
   10 minutes of CPU per 2 minutes of video and a 103 MB delta. A 6-sentence validation slice cannot
   select epochs; the product needs larger, separate validation sessions before promoting lip models.

## Limits (apply to every number below)

One speaker; a professional speaking ALOUD in studio conditions, frontal, good light, 640×360 video
at 29.97 fps; three minutes per session; n = 27 (A) and 22 (B) segments. This is not silent mouthing,
not a disabled user, not a webcam at home, not short conversational phrases. All numbers are an upper
bound / mechanism check, never "QuietBridge accuracy". Additional caveats:

- References come from automatic speech recognition of the same clips (two Whisper sizes), with nine
  hand fixes; an official transcript was not fetched (its host is not on the allowed list).
- The stock model may have seen this speaker or even these videos in training (large Auto-AVSR models
  use YouTube-derived corpora); the lip numbers may be optimistic for that reason too.
- Earlier tuning comments in `lipflow/vsr.py` and `lipflow/train_vsr.py` ("measured on the benchmark",
  "train on one video, test on another") likely refer to these same sample clips, so the stock decode
  settings may already have been chosen on session B material. Nothing in *this* run was tuned on B.
- Confidence intervals are 95 % percentile bootstrap over segments (2000 resamples). Rates pooled over
  random phrase-list splits resample whole segments, not split rows. With 22 segments, an interval of
  [100, 100] means "no failure seen in 22", not "never fails".
- CPU timings were taken while other agents' tests ran on the same 4 cores; treat them as rough.

## Step 0: evaluation set

| | session A (2016-03-12) | session B (2017-01-07) |
|---|---|---|
| ASR segments | 27 | 24 |
| kept | 27 (160 s, 468 words) | 22 (148 s, 427 words) |
| dropped | 0 | 2: B021 ("That'll"/"That will" disagree), B023 (1.7 s, small model heard nothing) |
| hand-fixed references | A002–A004 (digits spelt out), A016, A022 (models disagree; medium.en kept) | B002 (both models wrote "started to family"; corrected to "a family"), B006, B013 |
| face tracked in kept segments | ≥ 94 % of frames in every segment | 100 % |

Procedure: `asr_transcribe.py` (faster-whisper 1.x in a scratch virtualenv, `medium.en` primary,
`small.en` cross-check, int8 on CPU, ~2.5 min per clip for medium) → `track_clips.py` (MediaPipe
face tracking of every frame once, cached) → `build_manifest.py` (segments of about 3–10 s at sentence
ends or pauses, padded 0.2 s, kept only if both ASR texts agree after normalisation or a hand fix in
`reference_fixes.json` decides, and the face is tracked in ≥ 90 % of frames). Segments use the
product's own preprocessing: native frames → `LipReader.resample` (25 fps grid) →
`lipflow.face.mouth_rois`.

## Experiment 1: baseline

`decode.py --manifest <A|B>.jsonl --beam 4 10`. Product settings: `LipReader(beam_size=4)`, CTC weight
0.1, LM weight 0.3, no personal models, first 3 distinct non-repetitive candidates shown.

| session, beam | n segs / words | WER % [95 % CI] | sentence exact % | correct sentence in product's 3 | in full n-best | oracle WER (full n-best) | CPU s per s of video (encode + search) |
|---|---|---|---|---|---|---|---|
| A, beam 4 | 27 / 468 | 8.8 [5.3, 13.0] | 33.3 [18.5, 51.9] | 59.3 [40.7, 77.8] | 59.3 | 5.8 [2.7, 9.5] | 0.80 (search 0.56) |
| B, beam 4 | 22 / 427 | 13.6 [8.9, 18.8] | 13.6 [0.0, 27.3] | 27.3 [9.1, 45.5] | 27.3 | 10.8 [7.1, 15.1] | 0.81 (search 0.58) |
| B, beam 10 | 22 / 427 | 13.8 [8.8, 19.4] | 18.2 [4.5, 36.4] | 36.4 [18.2, 54.7] | 45.5 [27.3, 68.2] | 8.2 [4.5, 12.4] | 1.01 (search 0.78) |

Beam 10 vs beam 4 on B, paired: +0.2 WER points [−0.7, +1.3]. Beam 10 on A was started and stopped
for memory (2/27 segments); not reported. Beam search returns 11–13 finished hypotheses at beam 4 and
~35 at beam 10; at beam 4 the product's three already contain every exact hit the beam finds.

What it shows: on clean voiced frontal video the stock model is good at word level (9–14 % WER) but
whole sentences are rarely exact (most errors are a single word: "SINCE"→"JUST", "CHICAGO"→"YAHOO").
The n-best oracle bounds what any "pick from alternatives" UI can gain: at most from 13.6 % to 27.3 %
exact on B with three candidates, and to 45.5 % with a wider beam and the full list. A wider top-1
does not help. What it does not show: anything about silent speech or other speakers.

Recommendation: keep beam 4 for top-1. If the UI offers alternatives, a wider beam (10) roughly
doubles the chance that the right sentence is somewhere in the list at ~25 % more CPU; showing more
than three only helps if the person can scan them.

## Experiment 2: using the person's known phrases

`phrase_experiments.py --dev A --test B [--cond fps12] [--no-lookalike] [--text-only]`.
Saved-phrase list: a random half of the test session's sentences plus all of the other session's
sentences as distractors (49 sentences in total; 10 random halves per run). For dev the roles swap.
Any operating point was picked on A and only reported on B.

### 2a. Text-level matching (product `quietbridge/phrase_match.py` at `fe63671`)

`phrase_match.suggest(candidates, phrases)` on the product's 3 lip candidates, sweeping
`MAX_DISTANCE_RATIO` (other constants at their defaults; the shipped 0.25 is one point). "Rescue" =
intended sentence offered among ≤ 2 suggestions when it is saved and top-1 was wrong.
`phrase_match.summarize_rescue` on the same rows at 0.25 agrees (B: top-1-or-suggestion exact
86/110 in-list rows, 0 wrong offers).

| MAX_DISTANCE_RATIO | A (dev) rescue % | B (test) rescue % | B in-list: top-1 or suggestion exact % | B false offers when not saved % |
|---|---|---|---|---|
| 0.10 | 25.6 [6.9, 47.7] | 33.0 [12.0, 56.3] | 39.1 | 0.0 |
| 0.20 | 41.1 [17.9, 65.3] | 55.0 [31.8, 77.7] | 59.1 | 0.0 |
| **0.25 (shipped)** | 45.6 [21.9, 69.5] | 76.0 [54.4, 94.3] | 78.2 [59.0, 94.9] | 0.0 |
| 0.40 | 56.7 [32.8, 80.2] | 82.0 [63.1, 100] | 83.6 | 0.0 |
| 0.5–0.6 | 56.7 | 82.0 | 83.6 | 0.0 |

n: A 27 segments (130 in-list / 140 not-in-list split rows), B 22 segments (110 / 110). Top-1 exact
without the feature: A 33.3 %, B 13.6 %.

Look-alike discount (`lookalike()` patched to always return False, i.e. plain word edit distance;
patching `LOOKALIKE_COST=1.0` instead would be wrong, see "Possible product issues"): on A at 0.25
the rescue rate drops from 45.6 % to 41.1 % (it equals the 0.20 row); on B it is unchanged (76.0 %);
no false offers either way. The discount helped about one A segment; the effect is below what n = 27
can resolve.

**Hard negatives.** The zero false-offer rate above only means that distinct 10–20-word speech
sentences are far apart. When the saved list contains a *one-word variant* of what was said
(constructed: one word substituted, one word deleted, or "NOT" inserted) and the intended sentence is
not saved, `suggest` offered that wrong variant in 22/22 B segments for every kind (CI [100, 100]),
also when top-1 was already correct. `sensitive_difference` flagged 95 % of the inserted-NOT
variants, 9 % of substitutions and 6 % of deletions.

At 12 fps input (2c): top-1 exact falls to 0 % on B (11.1 % on A); at 0.25 the rescue rate is 64.5 %
[42.2, 82.9] on B (65.3 % on A), 0 false offers on distinct sentences; one-word variants are still
offered 91–93 % of the time.

### 2b. Model-level phrase scoring

For each segment the encoder runs once; every candidate phrase is scored teacher-forced with the
attention decoder and the CTC forward probability (and optionally the stock LM), combined with the
beam search's own weights 0.9·att + 0.1·CTC (+ 0.3·LM), raw or divided by token count. The variant
was chosen on A by near-miss AUROC: **0.9·att + 0.1·CTC + 0.3·LM, length-normalised**.

| | A (dev) base | B (test) base | B at 12 fps |
|---|---|---|---|
| closed set, list of 10 / 25 / 49 (all) | 100 / 100 / 100 % | 100 / 100 / 100 % | 100 / 100 / 100 % |
| open set (half of B saved + A): margin AUROC in-list vs not | 1.000 | 1.000 | 1.000 |
| margin ≥ 2 nats/token (chosen on A): in-list accepted & correct / not-in-list falsely accepted | 100 % / 0 % | 100 % / 0 % | 100 % / 0 % |
| reference beats all 6 one-word variants | 100 % [100, 100] | 100 % [100, 100] | 95.5 % [86.4, 100] |
| margin AUROC: reference among variants vs only variants saved | 0.915 | 0.890 | 0.913 |

n = 27 (A) and 22 (B) segments; open-set rows are 10 random halves per segment. Pairwise, the
reference outscored every substituted and NOT-inserted variant in all B segments; one deletion variant
won at 12 fps. Cost on one contended CPU thread, encoder output already computed: 0.35 s for 10
phrases and 2.3 s for 49 (≈ 45 ms per phrase, linear; batching 64 at a time), versus ~0.6 s of beam
search per second of video.

What it shows: given the encoder output, the lip model itself distinguishes the intended sentence
from distinct sentences and from one-word neighbours far better than text distance on the decoded
words does, and its margin can drive an abstain decision. What it does not show: short phrases
(2–5 words) like a real communication aid's, silent mouthing, or lists of hundreds of similar
phrases. The AUROC of ~0.9 against one-word neighbours means some wrong neighbours would still be
accepted at any threshold that keeps recall high.

Recommendation: do not rely on `phrase_match` alone to decide anything; it is a reasonable "also
consider" list for distinct phrases at the shipped 0.25 (raising the ratio to 0.4 rescued more on A
with no false offers on distinct sentences, but this is not tested on similar phrases). Build the next
version on model scoring: score the saved phrases (≤ 100, ~5 s per utterance on one thread today, far
less batched/threaded) and offer the best one only if its normalised margin over the runner-up is
≥ 2 nats/token; otherwise offer nothing. Re-measure on short, similar phrases and on silent mouthing
before trusting any threshold.

## Experiment 3: capture robustness

`decode.py --manifest B.jsonl --preset robustness` (beam 4, stock). Degradations are applied to the
native 29.97 fps frame list *before* the product's own `LipReader.resample` → `mouth_rois`, the way a
slow or stalling camera would deliver frames: frame rate by dropping frames to the target rate; a gap
by removing the frames of one window centred mid-utterance (the resampler then repeats the next frame);
pre-roll/tail loss by cutting the first/last 300 ms (segments are padded 0.2 s, so this removes about
0.1 s of speech motion); crop mis-registration by moving or scaling the 96 px crop in the aligned
face image. All rows: session B, n = 22 segments, 427 words. Paired Δ is against the same segments
undegraded.

| condition | WER % [95 % CI] | paired Δ WER points [95 % CI] | sentence exact % | correct sentence in product's 3 % |
|---|---|---|---|---|
| none (29.97 fps) | 13.6 [8.9, 18.8] | — | 13.6 | 27.3 |
| 20 fps | 19.7 [14.1, 26.4] | +6.1 [+2.5, +9.8] | 4.5 | 9.1 |
| 15 fps | 16.4 [11.0, 22.5] | +2.8 [−0.2, +6.8] | 18.2 | 22.7 |
| 12 fps | 22.2 [16.3, 30.2] | +8.7 [+4.2, +14.1] | 0.0 | 4.5 |
| 8 fps | 45.4 [37.0, 54.5] | +31.9 [+23.6, +41.4] | 0.0 | 0.0 |
| one 200 ms gap | 15.9 [10.6, 22.3] | +2.3 [−0.8, +6.4] | 18.2 | 27.3 |
| one 400 ms gap | 21.8 [16.6, 27.6] | +8.2 [+4.2, +13.1] | 0.0 | 0.0 |
| one 800 ms gap | 26.9 [21.5, 33.1] | +13.3 [+9.3, +17.6] | 0.0 | 0.0 |
| first 300 ms missing | 16.6 [11.8, 21.8] | +3.0 [−0.3, +6.2] | 13.6 | 18.2 |
| last 300 ms missing | 14.5 [10.0, 20.0] | +0.9 [−1.1, +2.9] | 13.6 | 27.3 |
| crop shifted 6 px down | 14.3 [9.2, 20.3] | +0.7 [−1.8, +3.5] | 22.7 | 36.4 |
| crop shifted 6 px right | 12.6 [8.0, 17.9] | −0.9 [−3.6, +1.5] | 18.2 | 31.8 |
| crop scaled 0.9 (zoom in) | 14.8 [9.7, 21.1] | +1.2 [−2.2, +5.1] | 22.7 | 31.8 |
| crop scaled 1.1 (zoom out) | 12.2 [7.8, 17.5] | −1.4 [−3.6, +0.5] | 27.3 | 31.8 |

Utterance length (stock, beam 4, A and B pooled, undegraded): < 12 words 14.1 % [5.6, 24.3] (n = 7),
12–17 words 11.2 % [4.9, 18.6] (n = 15), ≥ 18 words 10.7 % [6.9, 15.0] (n = 27). No reliable trend;
there are no short phrases in this material, which is exactly where a communication aid lives.

Session A at 12 fps (used for 2c): 17.5 % vs 8.8 % (+8.8 [+4.9, +12.8], n = 27), the same size of loss.

Session A at 20 fps: 14.3 % vs 8.8 % (+5.6 [+2.2, +9.1], n = 27), so a 20 fps camera costs 5–6
points in both sessions.

Resampling variant (`--preset resample`): the product picks, for each 25 fps tick, the first frame at
or after the tick (`LipReader.resample`, `searchsorted` left); the alternative picks the frame nearest
in time. Paired Δ (nearest minus product), n = 22 (B) / 27 (A):

| input | B | A |
|---|---|---|
| 29.97 fps | −1.9 [−5.1, +0.9] | +0.4 [−1.7, +2.2] |
| 20 fps | −5.2 [−9.5, −1.6] | +0.6 [−2.9, +4.7] |
| 12 fps | +0.0 [−2.4, +2.7] | −1.1 [−3.2, +1.1] |

Mixed: a clear gain on B at 20 fps that A does not reproduce. Not a reason to change the product's
resampler on this evidence; worth re-testing on real webcam timestamps.

What it shows: delivered frame rate and gaps dominate. 8 fps is unusable; 12 fps and a single
400 ms stall each cost about as much as the whole difference between the two sessions. On B, 20 fps
(+6.1) measured *worse* than 15 fps (+2.8); 29.97 → 20 fps drops every third frame and the product's
resampler then repeats and skips frames irregularly, which jitters timing, whereas 15 fps (every other
frame) is regular. The nearest-frame resampler removed most of the B 20 fps loss but not A's, so the
mechanism is unconfirmed; the safe reading is "anything below the camera's native ~25–30 fps costs
accuracy". Losing the first 300 ms costs more than losing
the last 300 ms (pre-roll matters more than tail). Shifts of 6 px (6 % of the crop) and ±10 % scale
are within noise: alignment is not the weak point on this frontal studio video. What it does not
show: webcam motion blur, low light, side views, rolling-shutter or compression artefacts, real
frame-timing jitter.

Recommendation: report and gate on delivered fps and the largest gap per recording (the adapter
already records `capture_fps` and `max_frame_gap_ms`): warn below ~20 fps or when any gap exceeds
~200 ms, and do not use such recordings for training or evaluation. Keep the ≥ 300 ms pre-roll. Do
not spend effort on crop registration before capture timing is solved.

## Experiment 4: speaker adaptation with session separation

`adapt_vsr.py --session A --val-last 6 --epochs 6`: the committed `lipflow/train_vsr.py` `finetune()`
(identical at `fe63671` and `cf96770`; the working-tree file was being edited, so the committed text was
copied into scratch with only its relative imports rewritten), with its defaults: scope
`frontend+encoder1`, AdamW lr 1e-4, 6 epochs, batch 1, augmentation (random 88 px crop, flip, time
masks, brightness), BatchNorm frozen. Fit: A000–A020 (21 segments, 118 s). Validation: the contiguous
tail A021–A026 (6 segments, 41 s, 121 words). Selection rule, fixed before the run: lowest validation
WER (beam 4), ties broken by teacher-forced validation loss; the stock model (epoch 0) is a candidate.
B is decoded once afterwards.

| epoch | steps | A-val WER % [95 % CI] | A-val loss | train time (s, cumulative) |
|---|---|---|---|---|
| 0 (stock) | 0 | 7.4 [0.9, 18.7] | 0.212 | 0 |
| 1 | 21 | 7.4 [1.8, 18.3] | 0.254 | 56 |
| 2 | 42 | 7.4 [0.9, 18.7] | 0.239 | 182 |
| 3 | 63 | 11.6 [3.6, 23.6] | 0.319 | 301 |
| 4 | 84 | 9.9 [3.6, 19.4] | 0.277 | 410 |
| 5 | 105 | 10.7 [0.8, 26.4] | 0.279 | 496 |
| 6 | 126 | 15.7 [4.0, 35.0] | 0.286 | 612 |

**Selected: the stock model** (no epoch beat it on A validation; every epoch raised validation loss).
B result for the selected model is therefore the stock baseline, 13.6 % [8.9, 18.8].

Diagnostic, declared before decoding B and not used for any choice: the recipe's default output
(epoch 6) on B: **11.9 % [7.5, 17.2]**, paired Δ −1.6 points [−5.5, +1.4], sentence exact 31.8 %
(vs 13.6 %), n = 22. No other epoch was decoded on B.

Cost: 4.9 s per training step on 2 contended CPU threads (2.7 s/step when the box was quieter); the
full default run (126 steps on 2 minutes of video) took 10 minutes plus 4 minutes of validation
decoding. The saved delta (frontend, embedding and first conformer layer, float32) is 103 MB — not
the "~45 MB" the `train_vsr.py` docstring states for the frontend-only scope. An earlier attempt was
killed by the out-of-memory killer after epoch 2 when a third model process started; training needs
~5.3 GB resident.

What it shows: on a speaker the stock model already reads well, the default recipe gives no
reliable gain; the A validation slice (6 sentences) and the B test (22 sentences) even point in
opposite directions, and both intervals include zero. A 6-sentence validation set cannot select
epochs: one word changes its WER by ~0.8 points. Forgetting on other speakers could not be checked:
there is no other speaker here, so that check is **missing**, not passed. What it does not show:
anything about adaptation to a speaker the stock model reads badly (the case adaptation exists for)
or to silent mouthing, where published work reports larger gains.

Recommendation: do not promote lip candidates on this kind of evidence. The product's selection
(`train_lips`: reject on validation regression) is the right shape, but it needs a validation session
of well over 20 utterances, and the recipe defaults (lr 1e-4, 6 epochs, frontend+encoder1, horizontal
flips) should be re-checked on a speaker with a high stock WER before anything else is tuned.

## Experiment 5: personal language model

`personal_lm.py --session A --val-last 6` (committed `lipflow/train_lm.py` `train()`, 3 epochs,
lr 3e-5, copied from `git show HEAD:`), then `decode.py --manifest B.jsonl --beam 4 --lm lm_A.pth
--tag plmA --enc-tag stock`. The product loads a personal LM as an extra scorer at weight 0.2 beside
the stock LM at 0.3 (`lipflow/vsr.py`). Control: the unchanged stock LM written as a "personal"
checkpoint (`personal_lm.py --control-copy`), same weight.

Training: 21 sentences of A (fit), 6 held out (validation). Validation perplexity 57.0 → 56.1 (−1.5 %);
the recipe saved the candidate (215 MB, 14 s on CPU). The recipe's general-text forgetting check
(`GENERAL_SAMPLE`) contains sentences transcribed from both sample clips; the 4 lines that overlap
session B were removed for this run so test text could not influence saving. Vocabulary: only 19 %
of B's word types (42 % of B's running words) occur in A's fit sentences.

| B, beam 4, same visual model | WER % [95 % CI] | paired Δ vs stock | sentence exact % |
|---|---|---|---|
| stock LM only | 13.6 [8.9, 18.8] | — | 13.6 |
| + personal LM from A (0.2) | 16.9 [11.6, 22.9] | +3.3 [+0.7, +6.5] | 18.2 |
| + unchanged stock LM copy (0.2), control | 17.1 [11.7, 23.6] | +3.5 [+0.8, +6.7] | 18.2 |

Personal LM vs control, paired: −0.2 [−0.8, +0.0] WER points.

n = 22 segments, 427 words. What it shows: the loss comes from adding LM weight, not from what the
personal LM learnt; with two different speeches the personal LM has little to contribute. What it does
not show: the case the feature is designed for (a person repeating their own phrases), where shared
vocabulary is high.

Recommendation: when a personal LM is present, keep the total LM weight unchanged (e.g. stock 0.15 +
personal 0.15, or interpolate) and choose that on a validation session; require a recognition-level
check (`quietbridge.evaluate_language`) rather than a 1.5 % perplexity gain before saving a candidate;
replace `GENERAL_SAMPLE` with text that is not from evaluation clips.

## Possible product issues found

- `lipflow/train_lm.py` (`GENERAL_SAMPLE`, lines 27–40 at HEAD): the "general text" forgetting check
  is made of sentences from the two sample clips that tests and benchmarks use (e.g. "SINCE THE DAYS
  OF GEORGE WASHINGTON…", B000; "BORN IN NEW YORK CITY…", A001). Evaluation text influences whether
  a personal LM is saved.
- `lipflow/train_lm.py` `train()`: saves on any validation-perplexity improvement (here −1.5 %) that
  then cost 3.3 WER points; and `lipflow/vsr.py` adds the personal LM's weight on top of the stock LM
  weight instead of sharing it.
- `quietbridge/phrase_match.py:135` (`look += cost == LOOKALIKE_COST`): if `LOOKALIKE_COST` is ever
  set to 1.0 (the obvious way to "turn the discount off"), every ordinary substitution counts as a
  look-alike, and `_accept`'s short-phrase rule (`exact + look == n`) then accepts any one- or two-word
  phrase of the same length. Latent; harmless at the shipped 0.5.
- `quietbridge/phrase_match.py` offers a one-word-different saved phrase even when the lip top-1 is
  exactly right (22/22 in the hard-negative test); with the commit's rule that any result gaining a
  phrase needs confirmation, a person with many similar saved phrases would be asked to confirm
  correct results often.

## Rerun everything

```sh
SCRATCH=/path/to/scratch; export QB_EXP_SCRATCH=$SCRATCH QB_EXP_THREADS=2
# 0. labels (scratch virtualenv; HF_HUB_DISABLE_XET=1 was needed behind this proxy)
uv venv $SCRATCH/asr-venv && VIRTUAL_ENV=$SCRATCH/asr-venv uv pip install faster-whisper
for c in 2016-03-12 2017-01-07; do for m in small.en medium.en; do
  HF_HUB_DISABLE_XET=1 $SCRATCH/asr-venv/bin/python scripts/experiments/asr_transcribe.py samples/$c.mov \
    --model $m --output $SCRATCH/asr/$c.$m.json; done; done
.venv/bin/python scripts/experiments/track_clips.py samples/2016-03-12.mov --out $SCRATCH/cache/A
.venv/bin/python scripts/experiments/track_clips.py samples/2017-01-07.mov --out $SCRATCH/cache/B
for k in A:2016-03-12 B:2017-01-07; do s=${k%%:*}; c=${k#*:}
  .venv/bin/python scripts/experiments/build_manifest.py --session $s --primary $SCRATCH/asr/$c.medium.en.json \
    --secondary $SCRATCH/asr/$c.small.en.json --cache $SCRATCH/cache/$s \
    --fixes scripts/experiments/reference_fixes.json --out $SCRATCH/manifest/$s.jsonl; done
# 1. baseline
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/A.jsonl --beam 4 10
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --beam 4 10
# 3. robustness (B), 12 fps on A for 2c
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --preset robustness
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/A.jsonl --preset lowfps
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/A.jsonl --preset fps20
for s in A B; do .venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/$s.jsonl --preset resample; done
# 2. phrases
.venv/bin/python scripts/experiments/phrase_experiments.py --dev A --test B
.venv/bin/python scripts/experiments/phrase_experiments.py --dev A --test B --text-only --no-lookalike
.venv/bin/python scripts/experiments/phrase_experiments.py --dev A --test B --cond fps12
# 4. adaptation on A, then test once on B
.venv/bin/python scripts/experiments/adapt_vsr.py --session A --val-last 6 --epochs 6
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --vsr-delta $SCRATCH/models/vsr_delta_A_epoch6.pth --tag adaptA_ep6
# 5. personal LM
.venv/bin/python scripts/experiments/personal_lm.py --session A --val-last 6
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --lm $SCRATCH/models/lm_A.pth --tag plmA --enc-tag stock
.venv/bin/python scripts/experiments/personal_lm.py --control-copy
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --lm $SCRATCH/models/lm_control.pth --tag lmcontrol --enc-tag stock
# tables (add --pair RUN_A RUN_B for any paired comparison, e.g. --pair lmcontrol/B__base__beam4 plmA/B__base__beam4)
.venv/bin/python scripts/experiments/report.py
```

Each process holds ~3 GB (training ~5 GB); on a 15 GB box run at most two or three at once.
Decodes resume from their per-segment JSONL files if interrupted.

## Follow-up (2026-11-20, later the same day): model-scored phrase offers, LM weight, capture warning

Same box, same two clips, same 49 segments, same stock model (beam 4). Scripts: `score_latency.py`,
`model_offers.py`, `e2e_runtime.py` (new) and `decode.py --lm-mode/--preset cadence` (extended). Session
B has now been used as a test set by several experiments, so it is **no longer a fresh test set**; in
this follow-up every choice was made on A and B was evaluated once per frozen design. Everything below
is one professional speaker talking aloud, 27 + 22 sentences of 9-35 words; "on these clips" applies
to every number. No silent mouthing, no short phrases, no real users.

### F1. Lip-model phrase scoring in the product

What changed: `LipReader.score_texts(enc, texts)` (`lipflow/vsr.py`) forces each text through the
attention decoder, the CTC branch and the beam's LM scorer(s), combined with the beam search's own
weights (0.9·att + 0.1·CTC + 0.3·LM, length bonus 0) and divided by tokens + `<eos>` — the variant
validated in Experiment 2b. The decoder's own top text is scored the same way; on three B segments its
teacher-forced total equalled the beam's score to 10⁻³ (a real-model test asserts this). After a decode,
`LipsAdapter` holds that one utterance's encoder output (≈ 2 MB for 30 s); the lips worker, outside the
runtime lock and before publishing, asks it to score the profile's phrases, then releases it. Which
phrases: all of them when the profile has ≤ 20 (`SCORE_ALL_PHRASES`), otherwise the 20 nearest by
look-alike word edit distance within a loose ceiling of 0.75 (`SCORE_MAX_PHRASES`,
`PREFILTER_MAX_RATIO`; the offer threshold of the text matcher is 0.25). Offer rule
(`phrase_match.model_offers`): a phrase is offered only if its normalised score ≥ top-text score −
`MODEL_OFFER_MARGIN` (0.2 nats/token), best score first, at most 2, never one already shown. Phrases
the model cannot write (digits above 20, non-ASCII letters) fall back to the text rule. Fallback to the
old text-only behaviour, recorded in `metrics.phrase_match.fallback`, when there is no scorer
(rehearsal, fakes), the scorer returns nothing, or it raises.

**Latency** (`score_latency.py`, 8 B segments of 5-9 s (median 165 encoder frames), CPU; median and
max ms for scoring the top text + N phrases; "cold" = the phrases' LM scores not yet cached, i.e. the
first utterance after start; "warm" = every later one; the top text's LM score comes from the beam):

| phrases | 1 thread, long (17 words) cold / warm | 4 threads, long cold / warm | 1 thread, short (4 words) cold / warm | 4 threads, short cold / warm |
|---|---|---|---|---|
| 2 | 224 (267) / 138 (185) | 121 (327) / 85 (826) | 167 (223) / 116 (177) | 69 (107) / 57 (183) |
| 8 | 646 (769) / 382 (474) | 489 (1020) / 202 (314) | 294 (464) / 228 (413) | 124 (336) / 97 (197) |
| 20 | 1419 (1782) / 778 (1002) | 933 (1783) / 462 (746) | 643 (1142) / 482 (886) | 233 (524) / 192 (434) |
| 50 | 3452 (4081) / 1849 (2291) | 2455 (2909) / 762 (1368) | 1099 (1466) / 692 (1139) | 413 (826) / 230 (952) |

The cost is the 6-layer, 768-wide decoder over every phrase token (batched; the source-attention
projections are already shared), ~37 ms per 17-word phrase on one thread, ~12 ms per 4-word phrase;
batching did not get it "far below 45 ms per phrase" for long sentences. Timings were taken with other
work on the box (4-thread maxima are noisy). Hence the cap of 20 scored phrases: worst case added
before the text appears ≈ 1.8 s (20 long phrases, first utterance, one thread), typical ≈ 0.2-0.5 s
(warm, 4 threads); in the end-to-end run below it was 47-96 ms because the prefilter passed 0-4 phrases.
Prefilter (pure Python, outside the lock): 40 ms for 100 short phrases, 184 ms for 100 twenty-word ones.

**Offers** (`model_offers.py`). Product functions on the cached decodes/encoder outputs. Scenarios per
segment: *open* = a random half of the segment's session + all sentences of the other session (10
splits); *var1_out* = one constructed one-word variant (2 substitutions, 2 deletions, 2 inserted NOT;
one list each) + the other session; *var_in* = the reference + its 6 variants + the other session;
*var_out* = the 6 variants only + the other session. "Rescue" = the intended sentence is offered when it
is saved and top-1 is wrong (of those, "possible" = it was not already one of the 3 lip candidates).
Rows / segments in brackets; CI = segment bootstrap.

Selection rule, fixed before the sweep: the largest open-set rescue on A such that every scenario's
already-correct results get an offer in ≤ 10 % of rows and unsaved open-set sentences get one in ≤ 5 %;
ties → the smaller margin. Operating curve on **A (selection)**:

| margin | open rescue % (90/18) | rescue possible % (51/11) | correct top-1 gets an offer: var1_out (54/9) / var_in (9/9) | wrong offer, top-1 wrong, var_in (18/18) | best-first when ref + variants saved (18/18) |
|---|---|---|---|---|---|
| 0.0 | 2.2 [0.0, 7.5] | 3.9 | 0 / 0 | 0 | 5.6 |
| 0.1 | 25.6 [6.8, 47.2] | 45.1 | 0 / 0 | 0 | 27.8 |
| **0.2 (chosen)** | 31.1 [10.3, 53.3] | 54.9 [25.0, 83.6] | 0 / 0 | 0 | 33.3 [11.1, 55.6] |
| 0.3 | 31.1 | 54.9 | 0 / 0 | 16.7 | 33.3 |
| 0.5 | 52.2 [28.3, 75.9] | 92.2 | 11.1 / 55.6 | 55.6 | 55.6 |
| 1.0 | 52.2 | 92.2 | 63.0 / 100 | 94.4 | 55.6 |
| text matcher (before) | 45.6 [22.6, 69.0] | 80.4 | **100 / 100** | 88.9 | 50.0 |

No margin offered anything for unsaved distinct sentences (open false offers 0 % everywhere, as before).

**B, once, at the frozen margin 0.2** (n = 22 segments; only 3 have an exact top-1):

| B (test, used before) | model-scored (0.2) | text matcher (before) |
|---|---|---|
| open rescue (100/19) | 54.0 [31.5, 76.5] | 76.0 [53.5, 94.5] |
| rescue when possible (82/16) | 65.9 [40.0, 87.3] | 92.7 [77.2, 100] |
| unsaved sentence gets an offer (110/22) | 0.0 | 0.0 |
| **correct top-1 gets a one-word-variant offer**, var1_out (17/3) / var_in (3/3) / var_out (3/3) | **0 / 0 / 0** | 100 / 100 / 100 |
| wrong variant offered when top-1 wrong, var1_out (114/19) | 0.0 | 71.1 [51.7, 89.5] |
| ref + 6 variants saved, top-1 wrong: ref offered first (19/19) / a variant offered (19/19) | 52.6 [31.6, 73.7] / 0.0 | 73.7 [52.6, 89.5] / 78.9 [57.9, 94.7] |
| 12 fps (A chose 0.2 again): open rescue (110/22) | 32.7 [13.8, 54.4] | 64.5 [42.2, 82.9] |
| 12 fps: wrong variant offered, var1_out (131/22) | 0.0 | 57.3 [37.4, 75.6] |

What it shows: gating on the model's own top text removes both failure modes of the text matcher —
on these clips no correct result gained a one-word-variant offer (0 of 23 rows on B, 0 of 72 on A;
text matcher: all of them) and no wrong variant was offered beside a wrong result — **but it rescues
fewer wrong results** (B: 54 % vs 76 %; at 12 fps 33 % vs 65 %). The reason is visible per segment on
A: when top-1 is wrong, the saved reference scores 0.07-0.48 nats/token *below* the decoded text;
when top-1 is right, one-word variants score 0.27-0.62 below it. The two ranges overlap, so no single
margin keeps both, and the selection rule favoured not asking people to confirm correct results.
Margin 0.5 would recover the text matcher's rescue rate on A at the cost of offering a variant beside
11-56 % of correct results. What it does not show: short phrases (a one-word change moves a 4-word
phrase's per-token score ~4x more than a 17-word sentence's, so 0.2 is effectively stricter there;
unmeasured), silent mouthing, lists of hundreds of similar phrases.

**End to end** (`e2e_runtime.py`): the product `Runtime` (browser-camera mode, mode "lips") with the
real `LipsAdapter`, fed the B clip's own 640×360 frames at native 29.97 fps timestamps on a virtual
clock; 34 saved phrases (B000's sentence; one one-word variant of B009's; B005's sentence and 4
variants; all 27 A sentences). State the page received:

| segment | top-1 | before (text matcher) offers | after: `phrase_matches` | evidence (path, scored, best gap) | scoring ms / decode ms | `capture_warning` |
|---|---|---|---|---|---|---|
| B000 (saved) | exact | none | none | model, 0 scored (sentence already listed) | 60 / 2310 | "" |
| B009 (only a variant saved) | exact | the variant | **none** | model, 1, −0.67 | 96 / 2948 | "" |
| B005 (sentence + 4 variants saved) | wrong; sentence is candidate 3 | 2 variants | 1 variant (gap −0.12) | model, 4, −0.12 | 95 / 1471 | "" |
| B012, every 2nd frame (15 fps) | wrong | none | none | model, 0 | 47 / 1799 | "The camera sent only 15 pictures per second, so some words may be missing. …" |

B000 decoded exactly through the product path (the experiment cache had "JUST THE DAYS…": different
crop path and pre-roll). B005 shows a limit of the rule: the offer is compared with top-1, not with the
best lip candidate, so a variant can still be offered when the right sentence is already shown as
candidate 3 (in the cached-decode simulation this did not happen; here it did once).

### F2. Personal language model weight

`lipflow/vsr.py` now *shares* the LM weight: with a personal LM the stock LM gets 0.15 and the personal
LM 0.15 (`PERSONAL_LM_SHARE = 0.5`, an a-priori split, not tuned), total 0.3 as without one
(`lm_weights()`; the old additive behaviour is `personal_lm_mode="additive"`, experiments only). B,
beam 4, same visual model, personal LM = the Experiment 5 candidate (trained on A's text with the
recipe as committed then), n = 22, 427 words:

| decoding | WER % [95 % CI] | paired Δ vs stock |
|---|---|---|
| stock LM only | 13.6 [8.9, 18.8] | — |
| additive: + personal LM 0.2 (Exp. 5) | 16.9 [11.6, 22.9] | +3.3 [+0.7, +6.5] |
| additive: + stock-LM copy 0.2 (control, Exp. 5) | 17.1 [11.7, 23.6] | +3.5 [+0.8, +6.7] |
| **shared: stock 0.15 + personal 0.15** | 13.6 [8.9, 18.8] | +0.0 [+0.0, +0.0] |
| shared: stock 0.15 + stock-LM copy 0.15 (control) | 13.6 | +0.0 (scores equal to 10⁻⁵) |

Shared vs additive, paired: −3.3 [−6.5, −0.7]. The shared personal LM changed no top-1 on any B
segment: it is as good as stock and shows no gain — this personal LM barely differs from stock
(validation perplexity −1.5 %), and the case it is meant for (a person repeating their own phrases)
is not in this material. The product therefore keeps loading an explicitly promoted personal LM, now
without extra weight; the saving rule below would not have saved this candidate at all.

`lipflow/train_lm.py`: `GENERAL_SAMPLE` is now 16 written general-English sentences that are not from
the clips (a test checks there is no shared 4-word run with the old clip lines). A candidate is saved
only if held-out perplexity improves by ≥ 10 % (`MIN_RELATIVE_GAIN`) and general-text perplexity rises
by ≤ 5 % (`MAX_GENERAL_LOSS`; callers may pass their own — `quietbridge.signs.train_language` still
passes 1.15). The result records the reason and states that perplexity is not recognition quality.

### F3. Capture warning and the resampler

`state.capture_warning` (runtime) is set per lip result when the delivered rate is below 24 fps
(`CAPTURE_MIN_FPS`; 20 fps cost +6.1 on B and +5.6 on A; nothing between 20 and 30 was measured, and
below 25 fps the model sequence must repeat frames), when a frame interval is ≥ 300 ms
(`CAPTURE_MAX_GAP_MS`; a 400 ms stall cost +8.2, 200 ms +2.3 within noise), or when frames were dropped
from the window; it is cleared by the next result, and `metrics.capture_quality` carries the numbers.

Resampler, A only (n = 27), paired WER Δ:

| input cadence | vs undegraded 29.97 fps | nearest-frame minus product rule |
|---|---|---|
| every frame at 29.97 fps, timestamps jittered 4 ms (sd) | +2.1 [−0.2, +4.9] | +0.2 [−2.1, +2.7] |
| irregular ~20 fps (each frame kept with p = 0.67) + 4 ms jitter | +5.6 [+2.2, +9.6] (= regular 20 fps: +0.0 [−3.8, +4.0]) | +0.9 [−2.6, +4.4] |

A does not support changing the resampler, so it is unchanged and B was not decoded for it. Irregular
~20 fps costs as much as regular 20 fps, which supports the 24 fps warning. 4 ms of timestamp jitter
alone may cost about 2 points (CI touches 0); worth checking with real browser timestamps.

### Possible product issues found in this follow-up

- **Fixed:** espnet's `MultiHeadedAttention` caches the decoder's source-attention projections keyed by
  the encoder output's data pointer and length (`espnet/nets/pytorch_backend/transformer/attention.py`,
  `_mem_kv`). On CPU a freed tensor's address is reused by the next same-size allocation (checked), and
  a deliberately reused buffer decoded the second of two same-length utterances as the first one's
  text. `LipReader` now clears the cache before every search and scoring pass (a real-model test
  covers it); the espnet code itself is unchanged. Re-check of earlier results: every segment whose
  encoder length repeated an earlier one in the same run (21 segment-conditions across A/B base, 12, 20
  fps, 400 ms gap and the new cadences) was decoded again with the fix; all 3 candidates were identical,
  so no reported number changes. In the product it can hit two consecutive utterances of equal frame
  count (e.g. repeated takes of one phrase).
- `quietbridge/phrase_match.py`: look-alike substitutions are now counted by a flag, so
  `LOOKALIKE_COST = 1.0` no longer turns every substitution into a look-alike (regression test).
- `quietbridge.signs.train_language` passes `--max-general-loss 1.15` by default, looser than the new
  5 % default of `lipflow/train_lm.py` (outside this change's scope).

### Rerun the follow-up

```sh
R=/path/to/r5; export QB_EXP_SCRATCH=$SCRATCH QB_EXP_THREADS=3
.venv/bin/python scripts/experiments/score_latency.py --out $R/latency_long.json
.venv/bin/python scripts/experiments/score_latency.py --short --out $R/latency_short.json
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --score A B --select A
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --evaluate B          # once
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --cond fps12 --score A B --select A
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --cond fps12 --evaluate B
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --lm $SCRATCH/models/lm_A.pth --lm-mode shared --tag plmA_shared --enc-tag stock
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/B.jsonl --lm $SCRATCH/models/lm_control.pth --lm-mode shared --tag lmcontrol_shared --enc-tag stock
.venv/bin/python scripts/experiments/decode.py --manifest $SCRATCH/manifest/A.jsonl --preset cadence
.venv/bin/python scripts/experiments/e2e_runtime.py --session B --segments B000 B009 B005 B012 --keep-every 1 1 1 2 --out $R/e2e/e2e_B.json
.venv/bin/python scripts/experiments/report.py --pair plmA/B__base__beam4 plmA_shared/B__base__beam4
```

### Addendum (2026-11-20): offer rule re-selected for what a false offer costs

Why: a lip-only result always needs confirmation (the top text is preselected; one tap). An extra
phrase row beside a correct result is clutter, and costs a tap only when it creates a *conflict*
(negation/number/safety difference removes the preselection); a missed rescue costs retyping. The
F1 criterion (≤ 10 % of correct results get *any* offer) priced clutter as if it were a confirmation.

New criterion, A only (`offer_rules.py --select A`): maximise open-set rescue subject to (a) unsaved
distinct sentences get an offer in ≤ 5 % of rows and (b) conflict offers beside an exact top-1 in
≤ 10 % of the constructed-variant rows (var1_out + var_in + var_out pooled, 72 rows / 9 segments);
ties → less clutter, then the simpler rule. Honest note: (b) was first applied per scenario, which
picked 0.2 again only because var_in/var_out have one row per correct segment (one segment = 11.1 %);
after seeing that, it was changed to the pooled form. Rule families swept (offers always ordered by
model score, at most 2): top-1 reference (`top:m`), best displayed lip candidate as reference
(`best:m`), and union rules (`union-<ref>:m:M`: model-gated at m OR the shipped text rule with a
model score no worse than the reference by M). Scores of the 2nd/3rd lip candidates were added to the
cache for the `best` rules. A, clean (n = 27; %):

| rule | rescue | rescue possible | unsaved offered | any offer beside correct (all / variant rows) | conflict beside correct (variant rows) | wrong offer beside wrong (variant rows) |
|---|---|---|---|---|---|---|
| text matcher (before) | 45.6 | 80.4 | 0 | 44.4 / 100 | 51.4 | 85.8 |
| top:0.2 (F1) | 31.1 | 54.9 | 0 | 0 / 0 | 0 | 0 |
| top:0.4 | 36.7 | 64.7 | 0 | 5.6 / 12.5 | 0 | 19.1 |
| **top:0.5 (chosen)** | **52.2** [28.3, 75.9] | 92.2 | 0 | 9.9 / 22.2 | **4.2** | 31.2 |
| top:0.6 | 52.2 | 92.2 | 0 | 17.9 / 40.3 | 9.7 | 38.3 |
| top:1.0 | 52.2 | 92.2 | 0 | 32.1 / 72.2 | 23.6 | 70.9 |
| best:0.5 | 52.2 | 92.2 | 0 | 9.9 / 22.2 | 4.2 | 31.2 |
| union-top:0.5:0.7 | 52.2 | 92.2 | 0 | 16.7 / 37.5 | 12.5 | 49.6 |
| union-top:0.6:0.7 | 52.2 | 92.2 | 0 | 19.1 / 43.1 | 9.7 | 50.4 |
| union-*:≤0.3:0.5 | 45.6 | 80.4 | 0 | 4.9 / 11.1 | 1.9 | 23-25 |

`best:m` equals `top:m` on clean A (it differs only where a lower lip candidate outscores top-1);
union rules add text-matched phrases that the model rates poorly and never beat the plain margin
(rescue saturates at 52.2 %: the rest of the "rescues" are references already shown as a lip
candidate). A at 12 fps, top designs: text 65.3 [45.0, 83.2]; top:0.2 49.2; **top:0.5 67.8 [49.1,
84.2]**, conflict beside correct 0 % (24 variant rows / 3 segments), wrong-beside-wrong 25.9 %;
best:0.5 67.8 (wrong-beside-wrong 23.3); top:0.6 and union-top:0.6:0.7 75.4 (conflict 4.2 %).

**Frozen design:** offer a saved phrase iff its length-normalised lip-model score ≥ the top-1 text's
score − 0.5 nats/token (`MODEL_OFFER_MARGIN = 0.5`), best score first, at most 2, prefilter unchanged
(all phrases if ≤ 20, else the 20 nearest within 0.75; unscorable phrases use the text rule).

**B, once** (not a fresh test set: used by Experiments 1-5 and F1; n = 22 segments, 3 with an exact
top-1, so "beside correct" rows are 23 from 3 segments):

| B | text matcher (before) | model-gated 0.2 (F1) | **model-gated 0.5 (now)** |
|---|---|---|---|
| rescue, clean (100 rows / 19 segs) | 76.0 [53.5, 94.5] | 54.0 [31.5, 76.5] | **79.0 [58.0, 95.2]** |
| rescue possible, clean (82 / 16) | 92.7 | 65.9 | 96.3 [87.8, 100] |
| unsaved sentence offered (110 / 22) | 0 | 0 | 0 |
| any offer beside correct: all (53) / variant rows (23) | 43.4 / 100 | 0 / 0 | 15.1 / 34.8 |
| conflict offer beside correct (variant rows, 23) | 52.2 | 0 | **0** |
| wrong offer beside wrong (variant rows, 152) | 73.0 | 0 | 17.8 [7.9, 27.6] |
| ref + 6 variants saved, top-1 wrong: ref first (19) | 73.7 | 52.6 | 78.9 [57.9, 94.7] |
| rescue at 12 fps (110 / 22) | 64.5 [42.2, 82.9] | 32.7 [13.8, 54.4] | **81.8 [63.8, 95.7]** |
| wrong offer beside wrong at 12 fps (175) | 61.1 | 0 | 13.7 [5.2, 22.7] |

On these clips the 0.5 design rescues at least as often as the text matcher (clean: +3 points, CIs
overlapping; 12 fps: +17 points) while creating no conflict beside a correct result and far fewer
wrong offers; the price against 0.2 is one extra row beside about a third of correct results when a
one-word variant is saved. Same limits as F1: long read-aloud sentences of one speaker, constructed
variants, small n; short phrases (where 0.5 nats/token is effectively stricter) and silent mouthing
unmeasured.

```sh
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --score A B   # adds lip candidates 2-3
.venv/bin/python scripts/experiments/model_offers.py --out-dir $R/offers --cond fps12 --score A B
.venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --select A
.venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --select A --cond fps12 --rules text top:0.2 top:0.5 best:0.5 union-top:0.6:0.7 top:0.6
.venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --evaluate B --rules text top:0.2 top:0.5          # once
.venv/bin/python scripts/experiments/offer_rules.py --out-dir $R/offers --evaluate B --cond fps12 --rules text top:0.2 top:0.5
```

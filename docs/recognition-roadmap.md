# Recognition roadmap: how to train stronger recognition

Research date: 2026-11-20. Scope: what would most plausibly make QuietBridge's recognition stronger for **one user who mouths silently**, with a small personal vocabulary, a few minutes of recordings in separate sessions, laptop CPU, and everything offline. This is a plan built from published evidence and the current code. It is **not** a measurement of this product: no participant data exists ([review protocol](review-protocol.md#participant-availability-2026-11-08)).

Conventions. "Reported" means a number from the linked source under that source's conditions. "Expected here" is my judgement about transfer and is not evidence. "Unverified" means I could not open a primary page. Papers were read as PDFs from arxiv.org. Licences were read from the raw `LICENSE`/`README.md` files on raw.githubusercontent.com. The github.com HTML pages, microsoft.com, kaggle.com, neurips.cc, isca-archive.org and personal academic sites were **not reachable** from this session.

## Summary: highest-value moves, in priority order

1. **Closed-phrase scoring with an open-text fallback.** Score the user's own saved phrases (20–200) against each lip window with the model's teacher-forced CTC/attention likelihood. Propose the best phrase only when its margin over the runner-up and over free decoding passes a validation-set threshold; otherwise show normal n-best. *Benefit:* likely the largest single gain for habitual messages. The evidence is indirect: few-shot closed-set silent-command work reports 81.7% one-shot and 98.8% five-shot accuracy on 30 commands ([LipLearner](https://arxiv.org/pdf/2302.05907)), and small-grammar VSR is far easier than open vocabulary (GRID baseline 12.04% WER, [Kim et al.](https://arxiv.org/pdf/2302.08102)). Medium confidence for our setting. *Cost:* low, using existing code (`lipflow/train_vsr.py:clip_loss`). *Needs:* the profile's phrase list; no new rights. *Testable now without a participant:* the software contract, yes; accuracy, no.
2. **Record silent, not voiced, personal data, and adapt on it.** On the only public paired normal/silent corpus, a model trained on normal speech had 22.55% WER on silent versus 11.58% on normal speech. Adding silent training data brought silent WER to 12.91% ([Kashiwagi et al. 2023](https://arxiv.org/pdf/2305.14203)). *Benefit:* high relevance, but the evidence comes from a 10-phrase, 39-speaker corpus. *Cost:* recording time only. *Needs:* participant consent. *Testable now:* no.
3. **Session-separated evaluation with uncertainty before any training change.** Use per-phrase exact intent, critical-word errors and false emissions at rest, with Wilson intervals, paired exact tests and session-block bootstrap ([Liu & Peng](https://arxiv.org/pdf/1912.09508)). *Benefit:* without it, items 1, 2, 4 and 5 cannot be told apart from noise. High confidence. *Cost:* low. *Testable now:* yes, on synthetic fixtures.
4. **Parameter-efficient speaker adaptation, compared against the existing front-end fine-tune.** One sentence-level study reports average WER 47.3% → 46.2% with 1 minute (some speakers got worse) and → 43.4% with 5 minutes (19/20 speakers improved). That used 0.7M trainable parameters against 39.6–40.0% for full front+back fine-tuning ([Yeo et al. 2024](https://arxiv.org/pdf/2409.00986)). *Benefit:* moderate, on voiced speech. Expect at least 5 minutes of silent data before a reliable gain. *Cost:* medium. *Testable now:* code path only.
5. **Better rejection for personal gestures.** Add explicit REST/background examples from separate sessions, use quantile (conformal-style) threshold calibration on validation ([Angelopoulos & Bates](https://arxiv.org/pdf/2107.07511)), and report false triggers per minute. *Benefit:* fewer unwanted emissions. This is the hard part, and the literature has little direct evidence. *Cost:* low–medium. *Testable now:* calibration code, yes; effectiveness, no.
6. **Interaction design as accuracy: n-best and phrase selection with explicit low-confidence states.** Fewer motor actions do not automatically mean faster communication. SpeakFaster saved keystrokes, but its 19-person mobile pilot found text-entry speed "not significantly different from the baseline" ([Cai et al.](https://arxiv.org/pdf/2312.01532)). Measure time-to-message and corrections, not only recognition rate. *Cost:* low. *Testable now:* UI contracts only.
7. **Do not chase a new base model now.** No open, downloadable visual-only checkpoint I could verify clearly beats the current 19.1% LRS3 model under usable terms (see §4). The current weights also have an unresolved rights chain. *Action:* resolve the rights of the current weights before investing further in them.

## What we can and cannot know today

- **Can:** verify software contracts such as phrase scoring, margins, fallbacks, threshold calibration, split hygiene and report arithmetic. Synthetic fixtures and public clips can do this. Run timing on this laptop CPU.
- **Cannot:** any recognition-effectiveness claim for the intended user. That includes the silent-speech gap for this person, gains from adaptation, gesture rejection rates in daily use, and communication rate. The review protocol forbids filling this gap with old private data, invented participants or coached scores. Every "expected here" below is a hypothesis for the staged plan, not a result.
- **Reported numbers are not transferable as-is.** Most VSR numbers come from voiced TED/YouTube speech (LRS3/VoxCeleb2). Rates and accents differ there, and the speakers are not mouthing silently.

## 1. Silent speech is not voiced speech

**Evidence.**
- [Kashiwagi et al. (Interspeech 2023)](https://arxiv.org/pdf/2305.14203) cite EMG and ultrasound studies showing that without auditory feedback, speakers rely more on somatosensory feedback and make *larger* lip movements, particularly for bilabials /p b m/. On AV Digits (39 speakers, 10 short phrases, normal plus silent), training on normal speech only gave 22.55% WER on silent versus 11.58% on normal. Adding silent training data gave 12.91%, and their viseme metric learning gave 9.97%. They also report the earlier finding of Petridis et al.: about 8.5% lower accuracy on silent speech. I did not open the original Petridis paper; the ibug.doc.ic.ac.uk host was blocked.
- [Gaddy & Klein (2020)](https://arxiv.org/pdf/2010.02960), EMG rather than video, showed that a model trained only on vocalized data transfers poorly to silent articulation. Transferring targets from vocalized to silent signals cut transcription WER from 64% to 4% in one condition and from 88% to 68% in another. The modality differs from ours, but the direction matches: silent articulation is a domain shift.
- Work on people without a voice is thin. A 2026 sEMG-plus-lipreading synthesis paper states that "adaptation to laryngectomized speakers remains an open research challenge" ([del Blanco et al.](https://arxiv.org/pdf/2606.09667)). I found no open dataset of silent mouthing by laryngectomy, aphonia or ALS users. Single-patient reports exist only in news summaries (unverified).

**Transfer.** The direction is consistent, but the size is uncertain. AV Digits uses a 10-phrase vocabulary, not open English. A person who has lost their voice may also articulate differently from a healthy person mouthing for an experiment. Examples include reduced jaw or lip range after surgery or with ALS, tracheostomy hardware, and fatigue across a day.

**Recommendation.**
- Collect only data in the mode the user will actually use, silent mouthing, under real lighting and camera position. Never collect voiced proxies. LipLearner's trick of saying a phrase aloud once to label it ([LipLearner](https://arxiv.org/pdf/2302.05907)) does not apply here. The label must come from typed or selected text.
- Record each habitual phrase across at least 3 separate sessions on different days. Within each session, add some free sentences containing names, numbers and negations, plus REST periods.
- Report silent-session WER separately. Never mix it with any voiced data the owner records for debugging.
- *Stop rule:* if stock-model exact intent on the user's habitual phrases is already high on a fresh session (see §7 for interval widths), prioritise items 1 and 6 over training.

## 2. Speaker adaptation with minutes of data

**Evidence.**
- [Yeo et al. 2024, "Personalized Lip Reading"](https://arxiv.org/pdf/2409.00986): Conformer visual encoder plus LLM decoder, VoxLRS-SA benchmark (voiced, in the wild, 20 test speakers). The baseline averages 47.3% WER, with a per-speaker range of 29.6–83.8%. Their prompt+LoRA adaptation (+0.7M params, 300 steps) reaches 46.2% at 1 minute, with some speakers worse than baseline. It reaches 43.4% at 5 minutes, better for 19/20 speakers, and 42.5/41.4/40.9% at 15/30/45 minutes. Full fine-tuning of front+back end reached 40.0%, and 39.6% with the language prompt (Table 3). LoRA on the vision side alone gave 42.5%.
- [Kim, Kim & Ro (TPAMI)](https://arxiv.org/pdf/2302.08102): prompt tuning with under 5 minutes on GRID and LRW-ID. On GRID, the mean WER of 12.04% fell to 5.48% with 1 minute of an addition prompt, and to 4.0–4.2% with 5 minutes. They argue that prompts limit the overfitting that fine-tuning suffers from with little data. GRID is a tiny fixed grammar, so treat this as the small-vocabulary upper end.
- The current code (`lipflow/train_vsr.py`) fine-tunes the front-end plus the first Conformer block with AdamW at lr 1e-4. Its in-code note (`train_vsr.py:87`) records a held-out speaker test of 36.3% → 29.6% WER. That is an internal, unpublished, voiced, single-speaker result. `quietbridge/signs/train_lips.py` defaults to 3 epochs and rejects a candidate unless validation WER improves.
- Self-training and test-time adaptation: [AV-CPL](https://arxiv.org/pdf/2309.17395) shows that unlabelled visual-only speech can improve VSR at LRS3 scale. [Flynn & Ragni](https://arxiv.org/pdf/2406.12937) report up to 32.2% relative gains from noisy-student training on the test recordings, for **audio** ASR. Neither shows that this works for one silent speaker with minutes of data. Both rely on pseudo-labels that are mostly right, which a 20–50% WER visual model will not give.

**Transfer.** Gains at the scale of "5 minutes → a few points of absolute WER" are plausible. With 1 minute, a regression is possible. Silent articulation may make adaptation *more* valuable (larger domain shift) and pseudo-labelling *less* safe.

**Recommendation (module: `lipflow/train_vsr.py`, `quietbridge/signs/train_lips.py`).**
- Add a `--scope lora` option: low-rank adapters on Conformer attention q/k/v plus the 3D-conv front-end, ~1M parameters, saved as a small delta beside the profile. Compare three arms on the **same** validation session: stock, current `frontend+encoder1`, and LoRA.
- Use only user-confirmed labels. The best "self-training" source is the confirmed messages from daily use, with consent and the clip retained: the user's own corrected text paired with that window. Exclude edited messages whose words differ from what was mouthed, unless the user marks the correction as what they meant. Keep those sessions out of validation and test, as the existing correction-isolation rules already require.
- Overfitting and forgetting guards: early stop on validation; keep a fixed general-English held-out check, as `train_lm.py` already does for the LM; cap steps (the reported recipe is a few hundred). Keep the current time masking and crop augmentation. Horizontal flip is fine for lips.
- Metrics: exact intent per habitual phrase, WER on free sentences, and critical-word errors (negation, numbers, names; the sets already exist in `quietbridge/fusion.py`).
- *Success:* the candidate beats stock on a fresh session in exact intent, with no increase in critical-word errors, and the paired test in §7 is significant. *Stop:* no gain at 5 minutes of silent training data across two seeds.
- `train_lips.py`'s selection rule still centres on one "target prompt" and "contrast false welcome". Generalise it to per-phrase exact intent and critical-word errors so the selection gate matches the product metrics.

## 3. Closed- or small-vocabulary decoding on an open-vocabulary model

**Evidence.**
- Closed-set silent commands are much easier than open dictation. LipLearner uses an LRW-pretrained contrastive encoder with a frozen embedding and a logistic-regression head. It reports one-shot F1 0.8947 in unseen conditions, and in its user study 81.7% / 96.5% / 98.8% accuracy for 1 / 3 / 5 shots on 30 commands. It also implements visual keyword spotting for activation ([LipLearner](https://arxiv.org/pdf/2302.05907)). Its code is MIT ([licence](https://raw.githubusercontent.com/rkmtlab/LipLearner/main/LICENSE)), but its encoder is trained on LRW, which is BBC data (see the rights table).
- Visual keyword spotting from a talking face is established ([KWS-Net, Momeni et al.](https://arxiv.org/pdf/2009.01225)). BSL-1K also used it to localise mouthings in signing ([Albanie et al.](https://arxiv.org/pdf/2007.12131)).
- Decoder scores are poorly calibrated. For attention seq2seq ASR, softmax confidence is overconfident, and a separate confidence module was needed ([Li et al.](https://arxiv.org/pdf/2010.11428)). I found no published VSR work on contextual biasing with a personal phrase list. The closest is slide-text biasing for audio-visual ASR, which I did not open.

**Options, for 20–200 habitual sentences plus free text, ranked by value/risk.**
1. **Phrase-list rescoring (recommended first).** For each window, compute the teacher-forced score `-clip_loss` (0.1 CTC + 0.9 attention, already in `lipflow/train_vsr.py`) for every saved phrase. Normalise by token count. Also compute the score of the free beam's top hypothesis. Propose phrase *P* only if score(P) − score(second phrase) ≥ m₁ **and** score(P) − score(free top) ≥ m₂. Otherwise fall back to the normal n-best. The phrase result is a candidate that still needs the user's action, like any other. With CPU encoder output computed once, 200 short phrases cost 200 decoder passes. That should take well under a second, but this has not been measured here.
   At the time of writing, an uncommitted text-level matcher (`quietbridge/phrase_match.py`) proposes saved phrases from the lip n-best text by viseme-aware word edit distance. It cannot recover a phrase the beam never produced; model scoring can. Compare the two, and their combination, on the same validation session.
2. **Shallow-fusion bias toward phrase prefixes** (a trie bonus during `BatchBeamSearch`). More flexible, since it can partially match a phrase. It is riskier: it can pull free text toward a habitual phrase, including dropping a "not". Do it only after option 1 is measured, and keep the existing negation/number protection in `fusion.py`.
3. **Personal LM** (already exists: `train_language`). Useful for phrasing, but weaker than explicit phrase scoring for exact habitual sentences.
4. **Few-shot embedding classifier** (LipLearner-style) on the Auto-AVSR encoder output pooled over time. It is cheap, but it would duplicate option 1 without the open-text fallback. Consider it only for non-verbal mouth gestures.

**Communicating uncertainty.** Show *categories*: "matches saved phrase", "best guess, please check" and "unclear, please choose or type". Derive them from margins with thresholds fixed on the validation session. Never show percentages. Log raw margins for audit. A conflict between a phrase match and a negation or number in the free hypothesis must surface as a choice.

*Success for option 1:* on a fresh test session, exact intent on habitual phrases improves over free decoding. Wrong-phrase proposals on free sentences, the "false phrase" rate, stay below a pre-set ceiling, e.g. ≤ 2/60 with the upper interval reported. *Stop:* if the false-phrase rate cannot be held under the ceiling at any margin that keeps a useful hit rate.

## 4. Stronger base models: reported quality and usable rights

| Model (visual-only, LRS3 test) | Reported WER | Size / inference | Weights downloadable? | Code licence | Weight / data terms |
| --- | --- | --- | --- | --- | --- |
| **Current**: Auto-AVSR VSR, LRS3+Vox2+AVSpeech pseudo-labels, 3,448 h ([paper](https://arxiv.org/pdf/2303.14307)) | 19.1% | ~250M class; CPU usable today | Upstream Google Drive/Baidu ([model zoo](https://raw.githubusercontent.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/master/README.md)); QuietBridge downloads a **third-party HF mirror with no model card or licence** ([API record](https://huggingface.co/api/models/Amanvir/LRS3_V_WER19.1)) | The upstream model-zoo repo licence adds BBC content terms: "individual research and private study", "only... for comparative or benchmarking purposes", no training of commercial or operational systems ([LICENSE](https://raw.githubusercontent.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/master/LICENSE)). README: "non-commercial purposes" | LRS3/VoxCeleb2/AVSpeech terms apply. Personal non-commercial use is the most this supports; even that should be confirmed |
| Auto-AVSR repo release ([README](https://raw.githubusercontent.com/mpc001/auto_avsr/main/README.md)) | 20.3% (3,291 h) | 250M | Google Drive | Apache-2.0 code ([LICENSE](https://raw.githubusercontent.com/mpc001/auto_avsr/main/LICENSE)) | README: models "may have their own licenses or terms... derived from the dataset" |
| BRAVEn Large w/ self-training ([paper](https://arxiv.org/pdf/2404.02098)) | 20.0% | Large | Google Drive ([README](https://raw.githubusercontent.com/ahaliassos/raven/main/README.md)) | MIT ([LICENSE](https://raw.githubusercontent.com/ahaliassos/raven/main/LICENSE)) | Trained on LRS3/Vox2/AVSpeech: dataset terms apply |
| USR Large ([paper](https://arxiv.org/pdf/2411.02256)) | 22.3% (repo table); 21.5% in the SOTA table | Large | Google Drive ([README](https://raw.githubusercontent.com/ahaliassos/usr/main/README.md)) | **No LICENSE file found** | Unknown: treat as not usable |
| AV-HuBERT Large ([paper](https://arxiv.org/pdf/2201.02184)) | 26.9% (self-training, 1,759 h, as cited in Auto-AVSR) | 325M class | Yes | Meta AV-HuBERT licence: "solely for your non-commercial research purposes" and forbids "biometric processing" ([LICENSE](https://raw.githubusercontent.com/facebookresearch/av_hubert/main/LICENSE)) | Rule out for a product |
| VSP-LLM (AV-HuBERT + LLaMA2-7B) ([paper](https://arxiv.org/pdf/2402.15151)) | 25.4% (433 h FT) | 7B LLM: not laptop-CPU practical | Google Drive | AV-HuBERT licence copied ([LICENSE](https://raw.githubusercontent.com/Sally-SH/VSP-LLM/main/LICENSE)) | Rule out |
| Llama-AVSR VSR ([paper](https://arxiv.org/pdf/2409.12319)) | 23.68% ([README](https://raw.githubusercontent.com/umbertocappellazzo/Llama-AVSR/main/README.md)) | Llama-3.1-8B | Google Drive | **No LICENSE found** | Llama licence + AV-HuBERT encoder: rule out |
| VALLR (phonemes → fine-tuned LLM) ([paper](https://arxiv.org/pdf/2503.21408)) | 18.7% | LLM stage | Google Drive ([README](https://raw.githubusercontent.com/MarshallT-99/VALLR/main/README.md)) | README: **CC BY-NC 4.0** | Non-commercial; LLM size/latency not checked |
| MMS-LLaMA | not checked | LLM | n/a | **CC BY-NC 4.0** ([LICENSE](https://raw.githubusercontent.com/JeongHun0716/MMS-LLaMA/master/LICENSE)) | Rule out |
| Whisper-Flamingo ([paper](https://arxiv.org/pdf/2406.10082)) | Audio-visual; needs audio | Whisper-large class | Yes | BSD-3 ([LICENSE](https://raw.githubusercontent.com/roudimit/whisper-flamingo/main/LICENSE)) | Irrelevant: the user produces no audio |

**Reading.** On LRS3, the best usable open models sit at roughly 19–22% WER, the same band as the current checkpoint (USR's own table: BRAVEn/USR Large at 26.5–22.3% in the high-resource setting). The LLM-decoder systems gain a few points at billions of parameters, which conflicts with laptop CPU and with "offline, small". None was evaluated on silent speech. **Expected here:** swapping base models is a costly, high-variance bet compared with items 1–4. **Rights finding:** the current weights reach users through an unlicensed mirror of a model whose upstream terms limit it to non-commercial research and benchmarking. The product must document that, consider fetching from the upstream links with the terms shown, and not extend use beyond them.

## 5. Personal dynamic gestures from hand landmarks (5–20 examples)

**Evidence.**
- [Shahi et al. (UIST 2024)](https://arxiv.org/pdf/2402.08420) used 21-point hand keypoints per hand, a graph transformer pretrained on a public dataset, and meta-learning with meta-augmentation. They report 94%/94%/90% average accuracy for one demonstration across 20 gestures from 21 participants. They train an explicit **background/null class** so the model can reject irrelevant hand movement. This is the closest published match to our open-set problem. Their dataset licence was not checked.
- [Schlüsener & Bücker (2022)](https://arxiv.org/pdf/2212.08363): relation-network few-shot models on hand reference points from Jester. Up to 88.8% for 5 gestures and 81.2% for 10, with 1–5 examples. No rejection.
- Open-set recognition: a better closed-set classifier gives a better "none of the above" decision, and a maximum-logit baseline is competitive ([Vaze et al., ICLR 2022](https://arxiv.org/pdf/2110.06207)). Split-conformal calibration sets a threshold at the ⌈(n+1)(1−α)⌉/n quantile of held-out scores, with a guarantee only under exchangeability ([Angelopoulos & Bates](https://arxiv.org/pdf/2107.07511)). Sessions on different days are *not* fully exchangeable, so treat the guarantee as approximate.

**Transfer.** The current `MotionModel` is masked nearest-template matching with leave-one-out and validation thresholds (`quietbridge/signs/motion.py`). That is the right family for 5–20 examples. With few classes, published few-shot accuracies of 80–95% are plausible for positives. False emissions on ordinary movement are what will decide usefulness, and the literature gives no rate for that.

**Recommendation (module: `quietbridge/signs/motion.py`, `train_motion.py`).**
- An uncommitted variant selector (`quietbridge/signs/motion_variants.py`, cross-fitted inside validation) was in progress at the time of writing. The variants below belong in that list, not in a separate mechanism.
- Keep templates. Add DTW alignment as an alternative distance, compared on validation; the current fixed 32-frame linear resampling penalises speed changes. Add train-time template augmentation: time warp ±20%, scale ±10%, in-plane rotation ±10°, small landmark jitter. Accept it only if validation positive recall rises without REST false emissions rising.
- Make REST/background a deliberate data target. For every session, record 2–3 minutes of "ordinary hands": scratching, reaching, resting, talking with hands, and near-miss versions of each gesture. Calibrate per-label distance and margin thresholds as a quantile of REST distances on the **validation** session, targeting e.g. ≤ 1 false emission per 10 minutes. Then evaluate once on test.
- Report false triggers **per minute of continuous rest**, from separate continuous recordings, in addition to per-window rates.
- A pretrained landmark encoder (Shahi-style) is a later option. It needs a pretraining dataset with clear rights; none was verified here.
- *Success:* positive recall ≥ the validation-chosen target, with the REST false-emission upper bound under the chosen ceiling. *Stop:* if the two cannot be met together, reduce the gesture set or rely on lips plus phrases.

## 6. Sign language

**Evidence.**
- **ASL Citizen** ([Desai et al.](https://arxiv.org/pdf/2304.05934)): 2,731 signs, 52 signers, splits by signer. I3D reaches 63.10% top-1 on unseen signers; a MediaPipe-Holistic ST-GCN reaches 59.52%. Landmark models were close behind. Licence: the search summary of the official Microsoft page says non-commercial research only. **Unverified**: the page was blocked.
- **WLASL** ([Li et al.](https://arxiv.org/pdf/1910.11006)): 2,000 signs. Licensed under C-UDA: "academic and computational use only. No commercial usage is allowed" ([README](https://raw.githubusercontent.com/dxli94/WLASL/master/README.md)).
- **BSL-1K** ([Albanie et al.](https://arxiv.org/pdf/2007.12131)): 1,064 signs, 70.61% per-instance top-1 on its test set, labels mined from mouthing. In 2020, I3D beat an OpenPose pose model 65.57% vs 49.66%. The repository has no licence file, consistent with [sign-language-support.md](sign-language-support.md).
- **BOBSL** ([Albanie et al.](https://arxiv.org/pdf/2111.03635)): 1,467 h, 39 signers, "available for non-commercial research usage" under a BBC agreement. Its 2021 sign-language translation baseline scored BLEU-4 1.00, i.e. continuous BSL→English was far from usable.
- **PopSign / Google Isolated Sign Language Recognition (Kaggle):** 250 one-handed smartphone signs from 47 Deaf signers, MediaPipe landmarks, with an LSTM baseline reported at 84.2% test accuracy. **Unverified**: only the NeurIPS search summary was seen, and the Kaggle licence was not reachable.
- **OpenHands:** Apache-2.0 code ([LICENSE](https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/LICENSE.txt)); pose-based ISLR models. Dataset terms are separate, as already noted.

**Transfer.** Expect large-vocabulary top-1 below these figures for a new signer under a webcam. The product's design of a small enabled concept set with confirmation is the right one. Landmark models are now close enough to I3D on ASL Citizen to fit the privacy posture better: no full-frame video is stored. They are also cheaper on CPU.

**Recommendation.** No BSL landmark model with clear rights was found. The next step is rights, not training: either a BSL dataset with permission for this use, or a personal "isolated sign" set recorded by the user and treated as personal gestures, not standard signs. For ASL, a landmark ISLR model trained on PopSign/ASL Citizen could feed the existing hashed-bundle path only after its licence is confirmed in writing. **Continuous signing** would honestly require a signer-independent continuous recogniser, non-manual features, fingerspelling, and an SL→English translation model with BSL-user validation. It is out of scope for this roadmap.

## 7. Evaluation with one user and little data

**Evidence and arithmetic.**
- Utterances from the same speaker or session are correlated, so a plain utterance-level bootstrap gives intervals that are too narrow. Resample blocks, here sessions or phrases ([Liu & Peng](https://arxiv.org/pdf/1912.09508)).
- Wilson 95% intervals (computed here): 18/20 correct → 0.70–0.97; 45/50 → 0.79–0.96; 90/100 → 0.83–0.95. Zero false emissions in 30 rest windows only bounds the rate below ~0.11; in 60 windows, below ~0.06.
- Paired comparison (same clips, model A vs B, exact intent): exact McNemar needs at least **6–0** or **8–1** or **10–2** discordant pairs to reach p < 0.05 (computed here). With an expected 10–20% of clips changing outcome, that needs roughly 50–100 paired test clips.

**Recommendation (module: `quietbridge/evaluate_fusion.py`, `train_lips.py`, `train_motion.py` reports).**
- Report counts with Wilson intervals, never bare percentages. Add an exact McNemar test for any candidate-vs-stock decision and a session-block bootstrap for WER.
- Metrics: exact intent (per habitual phrase and for free text); WER on free text; critical-word errors (negation/number/name); false triggers per rest-minute; corrections per message; time from window release to confirmed message.
- Freeze configurations before the final test; use the final test once; record the number of sessions, clips and errors. These rules are already in the docs; the additions are intervals and paired tests.

## 8. Interaction design as accuracy

**Evidence.**
- [SpeakFaster (Cai et al.)](https://arxiv.org/pdf/2312.01532): LLM abbreviation expansion saved 57% more motor actions than predictive keyboards in simulation. Two eye-gaze users with ALS were 29–60% faster than baseline. But in the 19-person mobile pilot, speed was "not significantly different" despite the keystroke savings. Speed also rose in scripted dialogues (+13%) and fell in unscripted ones (−10%), with wide intervals.
- Context-aware sentence retrieval for AAC (Kristensson et al., CHI 2020) reportedly estimated 50–96% keystroke savings in simulation only. **Unverified**: the PDF host was blocked.

**Transfer.** Selection from a short list is cheap when the right item is in the list and expensive when it is not: the user scans, rejects, then types. Phrase scoring (item 1) should therefore prefer *showing nothing extra* over showing a wrong phrase.

**Recommendation.** Put the phrase match first, then lip n-best, then type. One action accepts. Unfinished-edit preservation and no-auto-speak stay as they are. Measure time-to-message and corrections against typing alone for the same messages; recognition is worth keeping only if it beats typing for this user.

## Licence and rights table

| Artifact | Use proposed | Verified terms (source opened) | Decision |
| --- | --- | --- | --- |
| Auto-AVSR LRS3 VSR weights (current) | Keep for personal research use | Upstream model-zoo licence: BBC content for research/private study, comparative/benchmarking only ([LICENSE](https://raw.githubusercontent.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/master/LICENSE)); HF mirror has no licence ([API](https://huggingface.co/api/models/Amanvir/LRS3_V_WER19.1)) | Do not redistribute; document the terms; owner to confirm that personal assistive use fits |
| Auto-AVSR code | Training/adaptation | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/mpc001/auto_avsr/main/LICENSE)) | OK |
| RAVEn/BRAVEn code / weights | Not recommended now | Code MIT; weights trained on LRS3/Vox2 ([README](https://raw.githubusercontent.com/ahaliassos/raven/main/README.md)) | Dataset terms unresolved |
| USR | Not recommended | No licence file | Do not use |
| AV-HuBERT, VSP-LLM | Ruled out | Non-commercial research only; bans biometric processing ([LICENSE](https://raw.githubusercontent.com/facebookresearch/av_hubert/main/LICENSE)) | Do not use |
| VALLR, MMS-LLaMA | Ruled out | CC BY-NC 4.0 | Do not use in product |
| Llama-AVSR | Ruled out | No licence file; Llama + AV-HuBERT components | Do not use |
| LipLearner code | Ideas only | MIT ([LICENSE](https://raw.githubusercontent.com/rkmtlab/LipLearner/main/LICENSE)); encoder trained on LRW (BBC) | Do not import its weights |
| MediaPipe | Already used | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/LICENSE)) | OK; model-file terms not re-checked |
| BSL-1K | Existing experimental bundle | No repository licence (see [sign-language-support.md](sign-language-support.md)) | Unchanged: user must hold rights |
| BOBSL | Not usable | Non-commercial research under BBC agreement ([paper](https://arxiv.org/pdf/2111.03635)) | No |
| WLASL | Not usable for product | C-UDA, no commercial use ([README](https://raw.githubusercontent.com/dxli94/WLASL/master/README.md)) | No |
| ASL Citizen, PopSign/Kaggle ISLR | Possible ASL route | **Unverified** (pages blocked) | Verify before any use |
| OpenHands code | Possible ASL route | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/LICENSE.txt)); dataset terms separate | Code OK; models depend on data |

## Staged experiment plan (once one willing participant consents)

All sessions are on different days. Silent mouthing only. Each split is a fresh session. No tuning on test. Times are speaking time, excluding setup.

| Stage | Record | Build / run | Metric | Decision rule |
| --- | --- | --- | --- | --- |
| 0. Now, no participant | — | Write `quietbridge/evaluate_phrases.py` (phrase scoring via `clip_loss`, margins, fallback) and add Wilson intervals, McNemar and session-block bootstrap to reports; unit-test on synthetic and public clips; time 200-phrase scoring on CPU | Contract tests, latency | Merge if contracts pass; no accuracy claim |
| 1. Baseline | Choose 20–30 habitual phrases. Session A (practice): 3 reps each + 20 free sentences + 3 min REST. Session B (validation): same. ~10–15 min each | `evaluate_fusion` / `evaluate_language` on stock | Exact intent, WER, critical errors, false triggers/rest-min, time-to-message vs typing | If typing alone is faster and less effortful, stop recognition work for this user |
| 2. Phrase scoring | Reuse A and B; fix margins on B | `evaluate_phrases.py` on B | Exact intent on phrases; false-phrase rate on free sentences | Keep if exact intent rises and false-phrase upper bound ≤ ceiling |
| 3. Silent adaptation | Session C (practice, +5 min silent); later +10 min | `train_lips` arms: stock / frontend+encoder1 / LoRA (to build), 2 seeds | Same, on B | Promote only if McNemar p < 0.05 on B and no critical-error rise; stop if no gain at 5 min |
| 4. Gestures | 5–10 examples per gesture per session in A/B/C + 2–3 min ordinary hands each | `train_motion` with augmentation/DTW variants; thresholds calibrated on B | Positive recall, false emissions per rest-minute | Keep a gesture only if both targets are met on B |
| 5. Final test | Session D, new day, frozen configuration, ~60–100 clips + 10 min continuous REST | All three paths: lips, gestures, fusion | All metrics with intervals | Report once; fusion is kept only if it beats the best single path without more critical errors |

## Open questions and disagreements

- **How large is the silent-speech gap for this user?** Published estimates range from ~8.5 points of accuracy (Petridis, via [Kashiwagi](https://arxiv.org/pdf/2305.14203)) to nearly doubling WER on a 10-phrase set. Nothing measures it for open vocabulary or for people without a voice.
- **Adapters vs fine-tuning:** [Yeo et al.](https://arxiv.org/pdf/2409.00986) find full front+back fine-tuning slightly better (39.6% vs 40.9%) with 45 minutes. [Kim et al.](https://arxiv.org/pdf/2302.08102) argue prompts avoid overfitting with little data. Which wins at 5 minutes of silent data is unknown.
- **Landmarks vs video for signs:** 2020 BSL-1K found pose far behind I3D (49.66 vs 65.57). 2023 ASL Citizen found MediaPipe ST-GCN close (59.52 vs 63.10). Keypoint quality and model choice explain part of this; no BSL landmark comparison with modern MediaPipe was found.
- **Keystroke savings vs speed:** SpeakFaster shows that savings do not reliably become speed. Kristensson's retrieval simulations (unverified) are optimistic. Only measured time-to-message settles this.
- **Rights of the current weights:** the upstream README says "non-commercial", while its LICENSE text limits use to "comparative or benchmarking purposes" and research/private study. Whether a personal assistive tool fits is a question for the rights holders, not this document.
- **Not found or not verified:** VSR contextual biasing with personal phrase lists; open silent-speech video datasets from laryngectomy/ALS users; licences of ASL Citizen, PopSign/Kaggle ISLR and Shahi et al.'s pretraining data; on-device latency of the LLM-decoder VSR models.

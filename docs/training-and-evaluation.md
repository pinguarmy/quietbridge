# Training and evaluation

How to train personal model candidates from a profile's sessions, and how to compare lips-only, gestures-only and fusion on the same inputs. Record data first with Personalize ([Using QuietBridge](using-quietbridge.md#personalize-the-recording-lab)).

Rules that apply throughout:

- Every profile has separate train, validation and test sessions. Validation and final test must come from new recording sessions.
- Candidates are written beside the profile. Nothing here overwrites the stock model or an existing candidate, and nothing is deployed automatically: review the validation and final-test reports, then promote explicitly.
- Do not tune on the final-test session.
- Reports must distinguish positive gesture recall from REST silence, and ordinary lip sentences from named entities or other difficult targets. Preserve raw candidates and per-trial diagnostics.

## Train and evaluate candidates

The recording lab collects data; it does not silently deploy a model. Train a motion candidate from custom profile sessions:

```sh
uv run python -m hackathon.signs.train_motion \
  --data "$HOME/.local/share/quietbridge/profiles/alice/sequences" \
  --output "$HOME/.local/share/quietbridge/profiles/alice/models/motion.candidate.npz"
```

By default every recorded session in each split is used; `--session-prefix` narrows that. The trainer fits on train, calibrates acceptance thresholds on independent validation, then evaluates final test once. Per-label thresholds fall back to train leave-one-out rules when validation coverage is insufficient. The report separates positive recall, REST silence and wrong labels and includes per-trial rejection diagnostics. Review validation and final test before promoting a candidate to `models/motion.npz`. The old `--allow-legacy-no-validation` exists only to reproduce historical data.

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

The language report records held-out and general-English perplexity. A candidate is saved only when validation improves without excessive general regression. Review the report before selecting it with `--lip-language-model`; no candidate is deployed automatically. Approved corrections are included as described in [Corrections](using-quietbridge.md#corrections).

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

Standard-sign video pairs use the same evaluator with `--sign-model`; see [Standard signs with lips](standard-signs.md#record-and-compare-standard-signs-with-lips). They stay separate from personal-motion training and evaluation.

See the [recognition audit and cooperation plan](recognition-audit.md) for supported sign input, fixed fusion defects and remaining evaluation limits.

## What to measure

Measure new-session WER, exact intent, key-name/negation/number errors, gesture positive recall, REST/unknown false emissions, correction count and release-to-text latency. Compare lips-only, gestures-only and fusion on the same paired inputs.

## Capture-only checks

For capture-only checks, `uv run python -m hackathon.signs.capture_analysis <mouth.npz>` reports actual fps, frame gaps, face coverage and mouth-activity edges. Older recordings without activity metadata report those fields as null.

# Standard signs with lips

How to run and record with an explicit isolated-sign video bundle. Export, sources, rights checks and what was actually verified are in [Standard-sign support and evidence](sign-language-support.md); read that first.

This backend handles **isolated sign windows**, not continuous BSL or ASL sentences. A named gesture is not a validated standard sign.

## Run the main app with a sign bundle

The main runtime accepts an explicit local sign-video bundle:

```sh
uv run quietbridge --browser-camera \
  --sign-model /absolute/path/to/bsl-bundle/manifest.json
```

This replaces the stock/personal hand classifier, not the lip reader. Start the camera, choose Start recording for **one sign and its words**, choose Finish recording, then finish the message and review. Both workers share the window. Only enabled class mappings can contribute a sign concept; low scores, unmapped classes, short capture, large gaps and overflow stay silent. This classifier has no trained REST/unknown class, so thresholds alone cannot guarantee safe rejection of arbitrary movements. All standard-sign output requires confirmation.

The BSL-1K exporter consumes an externally installed upstream source, checkpoint and its exact class-ordered JSON labels. It does not download, bundle or license them. See [standard-sign setup, sources and validation limits](sign-language-support.md) before exporting. Explicit model load failure is visible; it never falls back to stock and calls that BSL. ASL is an allowed manifest language, but no ASL pretrained model was integrated or verified.

## Record and compare standard signs with lips

Start the local recording lab with the **same trusted bundle**:

```sh
uv run python -m hackathon.signs.lab \
  --sign-model /absolute/path/to/bsl-bundle/manifest.json
```

In **Paired**, choose **Standard sign + lips · local video**. Select an enabled sign, REST or unrelated movement and an intended sentence from your profile. Add suitable sentences in Choose phrases first; do not pair `apple` with an unrelated welcome prompt. Each trial requires explicit video consent before opening the camera. It saves local 256-square BGR frames containing face, upper body and hands, linked to the original hand coordinates and mouth crops. The default personal-gesture mode still saves no full-frame video. Finish within 240 frames / 10 seconds; overflow cannot be saved as a truncated trial. Discard writes no recording.

Use separate recording occasions for practice, validation and final test. Then compare lips-only, signs-only and fusion on the same video pairs:

```sh
uv run python -m hackathon.evaluate_fusion \
  --profile alice --split validation \
  --sign-model /absolute/path/to/bsl-bundle/manifest.json \
  --output /absolute/path/to/new-sign-comparison.json
```

Reports include WER/exact text, sign recall, REST/unknown false emissions, actual capture timing, inference-only sign timing and captured/current model identities. They reject missing consent, malformed video and known cross-split ID reuse. Personal and standard-sign pairs stay separate; standard-sign recordings never enter the personal motion trainer. This does not fine-tune BSL weights, promote a model or establish accuracy without independent labelled recordings. See [the implementation and smoke-test limits](sign-language-support.md#consented-paired-capture-and-replay-2026-11-04).

Personal-gesture paired evaluation is described in [Training and evaluation](training-and-evaluation.md#paired-fusion-evaluation).

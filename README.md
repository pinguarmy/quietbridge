# QuietBridge

**Lips, sign concepts and personal gestures, turned into a voice.** A local-first communication tool under active development.

QuietBridge lets a person build a message by typing, by silently mouthing English words to a webcam, or with a small set of hand gestures, review the proposed text, and then choose to speak it with the computer's own on-device voice. One webcam feeds silent lip reading and either stock poses, a personal gesture model or an explicitly installed isolated-sign video model. Nothing is spoken until the person presses **Speak**. Standard sign-language support is part of the product direction. The experimental BSL backend is word-level recognition, not full sentence translation or validated daily-use BSL support.

**Status, honestly:** the software paths below have automated contract tests. **Nothing has been validated with real users yet**, recognition accuracy on new users' sessions has not been measured, and audible speech on real hardware has not been checked in this round. The earlier model-review 10/10 ratings are not whole-product quality measurements; see [the revised review method and evidence](docs/review-protocol.md). Real-user effectiveness, independent-session recognition, physical camera/speech behavior and rights remain separate requirements.

## What works today, and what does not

| Capability | State | Limits |
| --- | --- | --- |
| Type → review → speak | Works without camera, models or training | Covered by browser contract tests with a silent speech recorder. Audible output not verified on real hardware this round. |
| Lip reading (Auto-AVSR, reused from `lipflow/`) | Ranked English text candidates from aligned mouth crops; optional personal visual/language candidates | Silent mouthing is ambiguous. Decoder scores are not calibrated probabilities. No new-user accuracy measurement. |
| Stock poses (MediaPipe gesture model) | Four mapped poses: thumb up, thumb down, open palm, pointing up | Not a BSL/ASL vocabulary or fingerspelling. Mappings are not live-verified, so their phrases need confirmation. |
| Personal gesture model | You record sessions and train a candidate; one accepted movement per explicit recording window; REST and rejected motions stay silent | Hand landmarks only. Effectiveness for a real user not measured. |
| Isolated BSL sign windows (experimental) | An external BSL-1K I3D bundle that **you** install can propose one sign concept per window alongside lips | Engineering smoke test only. No trained REST/unknown class. Always requires confirmation. You must check the weights' rights yourself. |
| Continuous BSL/ASL sentence translation | **Not supported** | Sign languages have their own grammar; a list of gesture names is not translation. No ASL model has been integrated. |
| Validation with real users | **None yet** | See [review protocol](docs/review-protocol.md#participant-availability-2026-11-08). |

The [recognition audit](docs/recognition-audit.md) has the detailed capability table and evidence links.

## Platform support

QuietBridge is a Python 3.11/3.12 loopback web app (`127.0.0.1`) used from a desktop browser. On-device speech uses whatever the operating system provides:

| Platform | Speech backend | Automated checks in this repository |
| --- | --- | --- |
| Windows | Windows PowerShell (or `pwsh`) with `System.Speech` | [windows.yml](.github/workflows/windows.yml): full setup with models, Python suite, Node UI and Chrome browser contracts, a real sample-clip lip read |
| Linux | `espeak-ng`, `espeak` or Speech Dispatcher's `spd-say`, whichever is installed first (for example `sudo apt install espeak-ng`) | [linux.yml](.github/workflows/linux.yml): headless Python suite without models, Node UI and Chrome browser contracts |
| macOS | The built-in `say` | No macOS CI job. The original Lipflow app targets Apple Silicon Macs ([legacy guide](docs/lipflow-legacy.md)). |

Without a speech backend the page says so; you can still use **Show large** or **Copy message**. None of these backends has been checked for audible output on real hardware in this round. Camera recognition and audible speech are not exercised by any CI job.

## Quick start

Install [uv](https://docs.astral.sh/uv/), clone this repository, then from the checkout:

```sh
uv sync --frozen                                   # Python dependencies (large: includes PyTorch)
uv run python -m hackathon.assets --download       # lip-reading and face models, ~1.2 GB (LRS3: non-commercial research use)
uv run python -m hackathon.signs.assets --download # stock MediaPipe gesture model
uv run quietbridge --browser-camera --open         # serves http://127.0.0.1:8765 and opens it
```

The two downloads are needed only for camera input: without them the server still starts, typing works and lip reading is reported as unavailable. `quietbridge` is the same as `python -m hackathon.server`; `uv run quietbridge --help` lists every option. Without `--open`, visit <http://127.0.0.1:8765> yourself. To look at the page with synthetic inputs and no camera or model loading, run `uv run quietbridge --rehearsal`; rehearsal does not validate recognition. Stock models and gestures are the default; select a profile explicitly with `--profile alice`.

## Using the page

- **Type first if you like.** Typing is visible without a camera or training, and a skip link goes straight to it. The short first-use guide can be skipped and reopened.
- **Use this message** applies typed or edited words without speaking or learning automatically. **Ctrl+Enter** (**Cmd+Enter** on a Mac) in the text box does the same; it never speaks. **Speak** is disabled while an edit is unapplied.
- **Speak** sends the applied message to on-device speech; **Stop speaking** interrupts it. Conflicts and uncertain inputs require you to choose or confirm first. UNKNOWN input never becomes a sentence.
- **Show large** displays the applied message full-screen so you can turn the screen toward someone. **Copy message** copies it. **New message** clears only the edits that existed when you confirmed it.
- **Camera input is optional.** Start the camera, then choose **Start lips + gesture recording** and **Finish recording** for one window. The button is a toggle, so there is no sustained hold; Space from the page background also toggles capture. **Finish message** stops the camera for final review. In browser-camera mode the preview is the browser's own local video; frames are also uploaded to the loopback server, for recognition only.
- **Unfinished edits survive** Settings/Personalize navigation and new recognition results in the same tab.
- **Settings** shows the active profile and models, supported concepts, frame rate, gaps and latency. **Personalize** opens the local recording page on demand.
- The page follows the operating system's light or dark setting. Message and camera order stays fixed at every screen size, message first.

Details, including recording windows, profiles, model reloads, corrections and the fusion safety rules, are in [Using QuietBridge](docs/using-quietbridge.md).

## Documentation

| Read this | For |
| --- | --- |
| [Using QuietBridge](docs/using-quietbridge.md) | Profiles, Personalize, recording windows, explicit models, corrections, fusion and safety rules |
| [Standard signs with lips](docs/standard-signs.md) | Running and recording with an explicit isolated-sign bundle |
| [Standard-sign support and evidence](docs/sign-language-support.md) | BSL-1K export, sources, rights checks and what was actually run |
| [Training and evaluation](docs/training-and-evaluation.md) | Training motion, lip and language-model candidates; paired fusion evaluation |
| [Verification](docs/verification.md) | Test commands, CI, and what each does and does not prove |
| [Review protocol](docs/review-protocol.md) | How the product is reviewed; current evidence and gaps |
| [Recognition audit](docs/recognition-audit.md) | Capabilities, fixed defects and what would establish an accuracy gain |
| [hackathon/README.md](hackathon/README.md) | Developer notes: contracts and runtime invariants |
| [docs/README.md](docs/README.md) | Index of all docs, including archived history |

## References, licensing and privacy

- The [original Lipflow guide](docs/lipflow-legacy.md) preserves the base application and historical benchmarks. The `lipflow` package and command remain in this repository.
- Based on [amywork777/lipflow](https://github.com/amywork777/lipflow), retained as the `upstream` remote.
- See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution and model restrictions. Treat LRS3 weights as non-commercial research use. External sign-model weights are not downloaded, bundled or licensed by this repository.

This repository is private. Do not commit keys, models, profile settings or personal recordings. Obtain consent before sharing samples.

# QuietBridge

**Lips and gestures, turned into a voice.** A local-first communication prototype.

One webcam supports silent lip movements, a small gesture vocabulary, or both together. Read the text, then press **Speak** for on-device speech. Our six-hour goal is a reliable short demo, not a general sign-language translator.

## Quick start

```sh
git clone https://github.com/pinguarmy/quietbridge.git
cd quietbridge
python -m hackathon.server --rehearsal  # Python 3.11/3.12; synthetic input, no models
```

Open <http://127.0.0.1:8765>. For live camera input, model setup and Peter's gesture integration, see the [local demo guide](hackathon/README.md).
Rehearsal does not validate real recognition. Windows base-app setup is in the [original Lipflow guide](docs/lipflow-legacy.md). Existing development machines do not need to rerun setup.

**Integration baseline:** the runtime, lip adapter, simple communication screen, separate Settings page, explicit confirmation and local speech are implemented. Peter's four-gesture recognizer is integrated, with matching phrase mappings still disabled pending live validation. Synthetic integration checks and browser rehearsal do not establish recognition accuracy. Real camera input, lip accuracy, gesture reliability and latency still need testing on Peter's demo Mac. See the [progress audit](docs/progress-audit.md) and shared types in [`hackathon/contracts.py`](hackathon/contracts.py).

## Work in parallel

| Who | Branch | Scope |
| --- | --- | --- |
| A · Sam | `feat/lips-ui-fusion` | Lip adapter, camera, fusion, UI, local speech and shared configuration |
| B · Peter | `feat/sign-recognizer` | `hackathon/signs/` only: recognition, vocabulary, sample tools and tests |

Create branches from `main`, open separate PRs, and let Sam merge. Use separate branches for Codex Cloud tasks too. Keep shared-file edits coordinated. See [AGENTS.md](AGENTS.md) for the short development rules.

```sh
git switch main
git pull --ff-only
git switch -c feat/lips-ui-fusion  # Peter uses feat/sign-recognizer
uv run pytest
```

## Repository layout

```text
hackathon/             New demo: runtime, lip adapter, fusion and UI
  contracts.py         Shared event types
  signs/               Peter's gesture recognition work
lipflow/               Reusable lip-reading engine and original native app
espnet/                Existing model dependencies; leave intact
tests/                 Base-app tests
scripts/               Existing development tools
docs/                  Demo plan and reference material
```

## Keep the demo focused

- Share one camera between lips and gestures; keep the core path offline.
- Start with 4–6 distinct gestures; keep 3 reliable ones if time is tight.
- Show agreement; offer a choice on conflict rather than inventing the user's intent.
- Speak only on a click, display actual latency, and complete three consecutive runs.
- Add optional cloud suggestions last, without blocking local output.

See the [six-hour plan](docs/hackathon.md). Run lightweight tests in the cloud and real recognition and speech on the demo Mac.

## References and licensing

- The [original Lipflow guide](docs/lipflow-legacy.md) preserves installation, architecture and historical benchmarks; those figures are not new QuietBridge measurements.
- [Earlier personalisation notes](docs/personalization.md) concern the base app, not required demo work.
- Based on [amywork777/lipflow](https://github.com/amywork777/lipflow), retained as the `upstream` remote.
- See [LICENSE](LICENSE) for code licensing and [NOTICE](NOTICE) for attribution and model restrictions. Treat LRS3 weights as non-commercial research use.

This repository is private. [Invite collaborators](https://github.com/pinguarmy/quietbridge/settings/access). Do not commit keys, models or private recordings. The existing Python package, CLI and data directory remain named Lipflow to avoid disrupting installed apps.

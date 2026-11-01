# QuietBridge — Developer Guide

## Goal

Build a working assistive-communication demo in six hours: lips / a small gesture vocabulary → text → local speech. Support lips-only, gestures-only and combined input. This is not a general sign-language translator.

**Move fast, reuse what exists, get it working first.** No unnecessary frameworks, major refactors or defensive code for hypothetical problems. Implement the task, run relevant checks and give a short handoff.

Use English for shared documentation, code comments and demo UI so both teammates can work directly with them.

## Ownership

- A · Sam: lip adapter, single-camera runtime, fusion, UI and speech; owns files and dependencies outside `hackathon/signs/`.
- B · Peter: gesture recognition; owns `hackathon/signs/`. Ask Sam for shared-file changes.
- Branches: `feat/lips-ui-fusion` and `feat/sign-recognizer`. Open separate PRs; Sam merges into `main`. No force pushes.
- Import shared types from `hackathon/contracts.py`; do not create another schema. Agree interface changes with the other person first.

## Implementation

- Reuse Auto-AVSR and face processing from `lipflow/`; put new work in `hackathon/`. Adapt rather than rewrite the engine.
- One camera, two processing paths. Keep only the latest frame (`maxsize=1`); load models once and decode lips after release.
- Start with 4–6 gestures; reduce to 3 if needed. Emit stable events, not repeated events while a pose is held.
- `SignRecognizer.process(frame_bgr, timestamp_ms) -> list[SignEvent]`; return `[]` without a result and provide `close()`. Use one monotonic clock in milliseconds.
- Use local fusion rules: select a matching candidate on agreement, show alternatives on conflict. Never invent a sentence from UNKNOWN.
- Display text first; speak with local TTS only when the user presses Speak. Do not automatically speak uncertain output.
- The core path works offline. Add optional API suggestions last, asynchronously; never block output or overwrite confirmed text.

## Development and checks

```sh
uv run lipflow doctor
uv run pytest
```

Use lightweight tests in the cloud; test real camera input, models and speech on the demo Mac. Run checks relevant to the change. Synthetic input is not evidence of live recognition.

Pace: agree interfaces in 20 minutes; get independent results by 1.5 hours; integrate by 3 hours; stop adding core features after 4 hours; rehearse a 60–90 second demo.

Done means all three modes work, conflicts ask for confirmation, text and speech work offline, three consecutive demos succeed, and actual latency is visible. Cut scope if it is unreliable.

## Keep intact

- Project: QuietBridge, https://github.com/pinguarmy/quietbridge.
- Keep the local directory, Python package/CLI `lipflow`, installed app and personal data paths unchanged for now.
- Do not commit keys, models, private settings or recordings. Obtain participants' consent before sharing demo samples.
- Preserve LICENSE, NOTICE and upstream attribution. LRS3 model weights have non-commercial research-use restrictions.

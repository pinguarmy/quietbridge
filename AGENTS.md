# QuietBridge — Developer Guide

## Goal

Build a local-first assistive-communication tool that a person can use repeatedly: silent lip movements, standard sign-language concepts and personal gestures produce reviewable text, then local speech. Each user has an isolated profile, data and model candidates. Standard sign recognition is a product goal. The explicit BSL I3D backend currently handles isolated sign windows, not continuous BSL/ASL sentences; do not equate a named gesture with a validated standard sign.

The product lead coordinates architecture and shared files. Delegated work must have disjoint write scopes and return actual checks. Do not restore hackathon-only scope or fixed demo phrases.

## Product rules

- Start with one user and a small custom vocabulary. Expand only after independent validation.
- Every profile has separate train, validation and test sessions. Never split validation from the same recording session as training.
- Train candidates beside the profile, review validation and final-test reports, then explicitly promote them. Never overwrite the stock model or an existing candidate.
- Display which model and profile are active. Missing explicit model paths fail visibly; do not silently load an old personal model.
- Display text first and speak only after the user's action. Conflicts and uncertain inputs require confirmation. Never invent a sentence from UNKNOWN. User-edited corrections create a new revision and never speak automatically; only an explicit learning choice writes them to the profile's correction store.
- First use must permit typing without camera, training or a mandatory wizard. Preserve unfinished edits through navigation and incoming results. Explicit recording uses an accessible start/finish toggle; do not require sustained holds or reorder live mobile controls. Personalize is an owned, on-demand loopback page, not a prerequisite second manual process.
- The core path works offline. Optional API suggestions remain asynchronous and revision-bound.
- Server identity and message revision are separate. A restart accepts fresh state without replaying an old action. Bounded action requests never retry an uncertain mutation automatically; polling must remain live. Successful older requests must not clear newer edits.
- Review reports distinguish software contracts from real-user/recognition effectiveness. Unknown evidence is not a pass. Do not coach reviewers toward a score or replace final independent review with accumulated targeted retests; use docs/review-protocol.md.

## Recognition architecture

- Reuse Auto-AVSR and face processing from `lipflow/`; keep product work in `hackathon/`.
- One camera feeds independent workers. Idle preview keeps only the latest frame; active capture uses a bounded 90-frame chronological buffer and reports overflow drops. Record actual input fps, raw frame count and gaps; resampling to 25 fps does not create missing motion.
- Lip results retain ordered candidates, decoder scores and capture timing. Decoder scores are not calibrated probabilities. Camera sessions use mouth-activity endpoints with bounded pre-roll; geometric mouth opening is a segmentation hint, not speech confidence.
- Explicit `--sign-model` selects a local, hashed isolated-sign bundle instead of stock/personal gestures. Use language-specific class ordering and preprocessing. Standard signs share an explicit window with lips and always need confirmation. Do not download, redistribute or activate artifacts without checking their separate source/weight/data rights. A high softmax score is not an UNKNOWN/REST guarantee.
- Stock gestures remain automatic. Custom MotionModel gestures default to an explicit start/release window and emit at most one accepted event per window. Optional `motion_auto` movement/rest segmentation is experimental and disabled by default. REST and rejected motions stay silent.
- The paired lab saves full-frame sign video only after per-trial consent, with at most 240 resized frames / 10 seconds and captured bundle identities. Default paired capture remains landmarks plus mouth crops. Standard-sign video pairs replay the actual sign adapter, stay separate from personal-motion training/evaluation, and require fresh sessions across splits.
- Combined recording uses a generation ID. Lips wait for the current motion result before fusion. Cancel invalidates late results from either worker.
- Fusion preserves lip ranking, reranks on gesture agreement, reports partial agreement, and composes only supported gesture phrases in time order. Unknown input contributes no text. Intent templates fill a declared gesture slot only on exact static-context matches; never erase unrecognized words.
- An explicit recording window retains gestures from the whole window; ambient stock events still expire.

## Development and checks

```sh
uv run lipflow doctor
uv run pytest
node --check hackathon/ui/app.js
```

Run focused checks while developing and the full suite before handoff. Synthetic events, fake backends and rehearsal validate contracts, not camera accuracy. Test real camera input, real models and audible speech separately.

Measure new-session WER, exact intent, key-name/negation/number errors, gesture positive recall, REST/unknown false emissions, correction count and release-to-text latency. Compare lips-only, gestures-only and fusion on the same paired inputs. Do not tune on the final-test session.

## Keep intact

- Project: QuietBridge, https://github.com/pinguarmy/quietbridge.
- Keep the Python package/CLI `lipflow` and legacy app data paths unchanged. QuietBridge profiles live under `~/.local/share/quietbridge/profiles/`.
- Do not commit keys, models, private settings or recordings. Obtain consent before sharing samples.
- Preserve LICENSE, NOTICE and upstream attribution. LRS3 model weights have non-commercial research-use restrictions.

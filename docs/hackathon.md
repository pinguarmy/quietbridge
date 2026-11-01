# QuietBridge: six-hour sprint plan

Adapted from the team's *LipFlow 6 Hour Hackathon Team Brief*. The project is now QuietBridge. This is a plan, not a report of completed features.

## Goal

One camera → lips / a small gesture vocabulary → local fusion → text → press Speak.

Sam owns lips, runtime, fusion and UI; Peter owns gestures. Import shared types from `hackathon/contracts.py`; see the root `AGENTS.md` for ownership.

## Schedule

| Time | Deliverable |
| --- | --- |
| First 20 minutes | Both import the shared types, agree 4–6 gestures and push their branches. |
| 20–90 minutes | Sam's adapter returns lip candidates; Peter's recognizer returns stable events. |
| 90–150 minutes | Single camera, display and local speech work; at least 3 gestures are reliable. |
| 150–195 minutes | Integrate all three modes; stop refactoring interfaces after this. |
| 195–255 minutes | Reduce false triggers, confirm conflicts and repeat the demo on the actual machine. |
| Final 90 minutes | Stop adding core features, prepare a consented fallback recording and rehearse a 60–90 second demo. |

If lips are not ready, guarantee a gesture demo and clearly label any lip replay. Do not sacrifice the core path for a larger vocabulary or optional API.

## Minimum acceptance

1. Lips-only, gestures-only and combined modes produce text; conflicts offer a choice.
2. A held pose does not repeat messages; either modality works when the other is missing.
3. Text and local speech work without Wi-Fi.
4. Show actual latency, complete three consecutive runs, and make sure both people can start the app.

Targets, not measurements: gesture feedback around 0.3–0.6 seconds; aim for lip text within 2 seconds of release. Repository setup alone does not establish live recognition performance.

## 90-second demo

- 0–15 seconds: explain the need to communicate when speech is unavailable.
- 15–35 seconds: gesture → text → Speak; explain the limited vocabulary.
- 35–55 seconds: mouth a short sentence → text → Speak; show latency.
- 55–75 seconds: show combined agreement and a conflict requiring confirmation.
- 75–90 seconds: explain the reused visual speech model and local core path.

Cloud suggestions are optional, asynchronous and last. Check API model availability and pricing before use rather than treating the original brief as fixed configuration.

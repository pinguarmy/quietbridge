# QuietBridge progress audit

## Latest deployment and delivery update

Sam has clarified the intended setup: Peter's camera computer at the venue,
with recognition on a desktop at home. A fresh fetch finds no camera-upload,
WebRTC/WebSocket transport, remote inference client, tunnel configuration or
returned speech/audio path. Current code is single-host: `cv2.VideoCapture`
opens the server host's camera, worker threads load models locally, HTTP binds
to `127.0.0.1`, and `say` speaks on the server host. Browser code only receives
JPEG/state data and sends controls. A remote-desktop camera-forwarding setup
could exist outside Git, but it is not documented in the published refs.

This means the home-desktop split is an implementation/configuration gap,
not a completed deployment. Do not expose the localhost service publicly as a
substitute for camera transport. Separate devices also need preserved capture
timestamps/utterance IDs and venue-side output. The offline acceptance criterion
applies to the single-host fallback; a home link necessarily needs a network.

The Windows import/UTF-8 fixes are merged through PR #1 at `7a27b3f`. Actual
Windows CI passed 166 tests, skipped 9 macOS-only checks, passed 14 subtests, and
completed an end-to-end sample-video lip read. Those results are separate from
live camera recognition.

The gesture mapping change enables the four configured phrases while leaving
all live-verification flags false. Unverified phrases require Confirm & speak.
The focused suite passes 41 tests plus 20 subtests, including Yes/No conflicts,
disabled mappings and alias collisions. The animated navy first page is being
implemented separately on `feat/animated-story`; it is not on main at this snapshot.

The earlier audit below records the integration before these follow-ups.

Audit snapshot: 1 November 2026. Includes Sam's simplified UI (`c6023ce`),
Peter's published `feat/sign-recognizer` (`5641e38`) and their integration in
this change. GitHub reports a private repository with `main` as its default
branch. At the final check Peter's branch is published; there are no pull requests.

## Product and architecture

The agreed target is a limited lip-and-gesture communication prototype. One
camera supplies two local recognition paths: Auto-AVSR returns lip-reading text
candidates; Peter's hand recognizer returns stable supported labels. Local fusion
compares them, displays a result or alternatives, and the user explicitly presses
Speak for native speech.

An LLM verifier is not implemented and is not on the critical path in the team
brief. A future LLM can suggest a candidate when the inputs are uncertain. A
plausible rewrite cannot independently establish what the person intended;
uncertain inputs still need user confirmation and measured recognition tests.
The current app makes no cloud inference requests.

The older personalized Lipflow application supports cloud text cleanup, which
may explain the LLM expectation. The new demo adapter directly calls the local
reader's encoder and beam search and bypasses that cleanup path. The decoder's
local language model scores candidate sequences; it is not an independent
verification of the user's intended message.

## Evidence and remaining work

| Component | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Shared contracts | Both tracks import the same schema | Keep future phrase-output changes coordinated |
| Lip engine | Existing Auto-AVSR engine; new adapter loads and warms it once | Real clip and live lip test on Peter's Pro |
| Camera and workers | One capture implementation; latest-frame queues; model-worker tests | Actual concurrent camera, face and hand processing |
| Sign recognizer | Peter's pretrained four-gesture recognizer and stabilizer are integrated | Real camera trials on the Pro |
| Vocabulary | Correct four output labels and proposed phrases, all unverified | Agree and test phrases; enable only verified entries |
| Fusion | Local rules; 24 integration tests plus 12 gesture logic/asset tests pass | Real agreement, conflict and missing-modality trials |
| UI and speech | Simple communication screen, separate Settings; browser rehearsal and native speech dispatch tested | Audible output and repeatability on the demo Pro |
| LLM/API suggestions | Not implemented | Optional only after the local demonstration works |
| Demo evidence | Peter reports limited real-model static-image checks; no new live-camera result | Five lip sentences, gesture trials, Wi-Fi-off check, three consecutive runs |

The 36 locally run checks passed, plus 14 unittest subtests. They use synthetic
events, mocked lip workers and a loopback HTTP server; the gesture subset checks
stabilization and asset handling without loading MediaPipe. They establish
integration behavior, not recognition accuracy. The current
rehearsal page does not run a camera or a vision model.

## Peter's visible progress

Peter published the recognizer at 13:21 GMT, followed by handoff/shared-English
documentation through 13:23 GMT. It emits `THUMB_UP`, `THUMB_DOWN`, `OPEN_PALM`
and `POINTING_UP`; the proposed phrases are Yes, No, Please stop and One moment.
It uses MediaPipe's pretrained model, not the MNIST alphabet dataset.

His handoff reports 167 tests passed and 11 skipped on his host; that full run was
not reproduced on Sam's Mac. It reports actual static-image smoke tests: thumbs
down and pointing up emitted an event, thumbs up did not stay above threshold,
and open palm remains untested on real input. His camera attempt failed permission
checks on an M2 host. These are not live Pro measurements or an accuracy result.

Sam has aligned the root vocabulary with the emitted labels. All entries remain
disabled until live checks pass. Peter's older mapping-mismatch note is superseded
by the current root mapping. Unpushed local work and Peter's other chats remain
outside this audit's visibility.

## Recommended practical split

Keep the existing code ownership to avoid collisions. Sam owns contracts,
adapter/runtime, local fusion, UI, speech, common configuration and integration.
Peter owns `hackathon/signs/`. Since the demo is on Peter's MacBook Pro, Peter
also runs the camera/model checks and provides actual outputs and timing; Sam
implements any necessary changes outside Peter's owned directory.

Sam's immediate priority is integration evidence and a usable confirmation flow.
Peter owns the newly proposed "hello dev day" work. The current four-gesture
recognizer does not recognize that phrase or arbitrary letters. Sam is not
training a competing alphabet model; Peter should provide the phrase-output
interface before that work is integrated. The existing demo still needs a
reproducible live environment and three reliable gestures. Add an LLM only after
the local path works.

## Next checkpoints

1. Within 10 minutes: Peter pulls main in the isolated demo checkout, reports
   whether camera/model warmup succeeds, and confirms the exact gesture labels.
2. Within 30 minutes: test each chosen gesture five times with neutral between
   attempts; enable only tested, agreed mappings. Sam integrates actual event
   payloads and any separately agreed phrase output.
3. Within 60 minutes: combine at least three reliable gestures with a real lip
   result; exercise agreement, conflict and missing input.
4. Within 90 minutes: run with Wi-Fi off, record errors and actual latency,
   complete three consecutive runs and capture a consented fallback recording.

If lip reading cannot produce usable text from a real clip, preserve sign-only
as the live demo and label prerecorded lip replay explicitly. If gestures are
unstable, reduce the vocabulary before adding model complexity.

## Dataset request

Kaggle MCP successfully answered an anonymous `download_dataset` call for
`datamunge/sign-language-mnist`. The 65,633,206-byte archive downloaded, passed
its CRC check and was unpacked outside Git. The training and test CSV checksums
match Sam's supplied Desktop archive. The download manifest records provenance
and SHA-256. Kaggle is configured in Codex, but account OAuth failed with
`ClientId too small: codex`; the public MCP download did not require account access.
Kaggle metadata describes 24 static ASL alphabet classes, excluding J and Z,
with 28 by 28 grayscale images, 27,455 training examples and 7,172 test examples.
This is alphabet data, not training labels for HELP/STOP/WATER or general sign
language. Its use would need an agreed fingerspelling scope and live-camera
validation. Sam has chosen to keep 3–5 useful gestures for this hackathon and save
the alphabet dataset for later; it is not part of the current live recognizer.

Source: <https://www.kaggle.com/datasets/datamunge/sign-language-mnist>.

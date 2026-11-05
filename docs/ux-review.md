# QuietBridge onboarding and everyday-use review

## Frozen rubric, 2026-11-04

Scope: the shipped local web communication page, Settings and Personalize, including first use, a returning user, typed fallback and explicit isolated-sign capability. This is an engineering usability review, not a clinical accessibility study or a recognition-accuracy benchmark. Scores cannot establish recognition quality for a new signer. Real-user validation and model/data rights remain separate release constraints.

Reviewers independently inspect the same revision and run the journeys below. Do not award points from screenshots or a change summary alone. Cite source lines or an observed interaction for each deduction. Unknown evidence is not a pass. Do not raise a score just to meet a target.

Each category has two checks worth 0.5 points each. For each check: 0.5 = observed working and clear, 0.25 = usable but incomplete or supported only by source/test evidence, 0 = missing, broken or misleading. Sum ten categories to a score out of 10. Keep this rubric unchanged across rounds. No P0/P1 issue may remain even if the numerical score is high.

| Category | Check A, 0.5 | Check B, 0.5 |
| --- | --- | --- |
| 1. First-use orientation | New user learns what the app does and the current input limitations without visiting technical docs | First-use guidance has an obvious next action, can be skipped and reopened without losing a message |
| 2. First successful expression | Typing is visible and usable without camera permission, recognition or training | Typed text reaches a reviewed, speakable message without automatic speech or automatic learning |
| 3. Camera and capture | User can understand start, capture, finish and camera-off states for the actual backend | Permission denial, missing camera or model warmup offers a specific recovery and typed fallback |
| 4. Capability honesty | Main page names the actual stock/personal/standard-sign backend and supported concepts | Unsupported signs, full sign translation and scores are not presented as validated capabilities |
| 5. Review and speech | Conflicts offer a deliberate selection/edit and prevent accidental speech | A user can see the intended message, explicitly speak, and begin another message without stale text surprises |
| 6. Error recovery and continuity | Network/action failures have a useful next step, do not trap the user and preserve unsaved editing | Navigation/profile/model changes disclose or prevent loss of an active camera, draft or unsaved edits |
| 7. Personalization | Personalize opens a working page with a clear return path; profile setup does not require developer knowledge | Recording, independent-session checking, saving models and activation are distinct and actionable; defaults do not imply training is required |
| 8. Privacy and control | Camera, local storage and optional AI disclosures match actual behavior; nothing starts without an action | Saving sensitive training/video data requires explicit appropriate consent; stop/discard and leaving the page release capture |
| 9. Accessibility and narrow screens | Keyboard-only operation, focus, labels and feedback support the critical journeys without requiring a sustained hold | At 390px and zoomed layout, actions/text remain reachable without horizontal page overflow or disruptive live reordering |
| 10. Product coherence and quality | Daily communication is not buried beneath diagnostics, conflicting instructions or training vocabulary | Main, Settings and Personalize use consistent terminology and have regression coverage for changed behavior |

Severity: P0 = unintended speech/disclosure/data loss; P1 = critical journey blocked, silently wrong save/activation or a material accessibility/privacy failure; P2 = recoverable friction or unclear wording; P3 = cosmetic. Each finding needs reproduction, impact and a bounded fix. An unverified suspicion is a question, not a defect.

## Shared journeys

1. Fresh browser state, camera off: explain current capabilities, find typing, enter `Please give me a moment`, use it, confirm no speech/learning occurred. Do not trigger audible speech during review; verify its boundary with the actual API/state and existing fake-speaker tests.
2. Find and use the camera instructions. Exercise permission-denied and model-unavailable states with clearly labelled test doubles, never real camera permission. Verify fallback remains available. Do not pretend those doubles measure recognition.
3. Generate a synthetic conflict in explicitly labelled rehearsal mode. Select and edit one candidate. Confirm the chosen revision controls speech. Start a new message and inspect prior-message handling.
4. Enter Settings and return with a draft/unsaved edit. Check profile/model switching warnings and input names, enabled concepts and AI opt-in.
5. Open Personalize from the app, create or inspect a profile, choose a phrase, inspect recording and checks, return. Do not activate a camera or train/promote a real model. Note what still requires setup or another process.
6. Inspect standard-sign video consent, the default no-full-video path, negative examples and saved-versus-active wording. Use existing capture API tests as evidence, not a live private recording.
7. Keyboard-only first expression, capture alternatives and focus recovery. Inspect 390px viewport and 200% equivalent layout. Check no horizontal page overflow and that live results do not move an active control out from under the user.
8. Disconnect/reject one action and restore service. Verify messages/edits and pending controls recover. Never stop another reviewer's server or edit shared source.

## Review record

Baseline revision: `4b1a2d9`. Requested models: Terra, Gemini and Kimi 3.1. Available catalog routes are `cpa/gpt-6.1-terra`, `cpa/gemini-3.8-flash`, `cpa/kimi-k3-256k`. The last is Kimi K3, not a verified Kimi 3.1 identity. Report this substitution explicitly.

Final threshold requested by the user: every available reviewer at least 9/10 and no P0/P1 defects. A blocked or unavailable reviewer is not a passing score.

### Baseline, revision 4b1a2d9

| Reviewer | Reported total | Sum of submitted category checks | Evidence limits |
| --- | --- | --- | --- |
| Terra | 4.25 | 3.75 | Did not exercise many journeys; unknown/source-only checks capped |
| Gemini 3.8 Flash | 6.25 | 6.25 | Camera/audio deliberately untested; mobile issue source-established |
| Kimi K3 | 7.00 | 8.00 | Source-only camera and mobile checks; orientation score did not match missing reopenable guide |

Arithmetic discrepancies are preserved here, not silently fixed or used as passing scores. Round-two reviewers must sum their 20 explicit checks and reconcile the total. Shared first-pass findings: unsaved message and profile edits disappear; explicit capture requires a hold; mobile panels reorder while live; main capability/profile are unclear; new-message action and loss warnings are absent. Lead additionally reproduced a dead Personalize destination when only the main process is started. Reproduction details are retained outside Git; no private recording was used.

### Second pass, working tree after first fixes

| Reviewer | Score | P0/P1 | Decision |
| --- | --- | --- | --- |
| Terra | 8.25 | No reproduced serious defect; important journeys unexercised | Not passing; missing direct checks remain |
| Gemini 3.8 Flash | 10.00 | None reported | Reviewer pass, not release approval |
| Kimi K3 | 9.75 | P1: preserved draft can override a chosen conflict alternative | Not passing despite score |

Kimi reproduced a selected candidate displaying `No.` while Use this message still submitted an older unfinished draft. This has priority over the other reviewer's high score. Selecting an alternative now requires explicit consent to replace any unfinished edit and synchronizes the editor with the selected words. Declining keeps the edit and blocks stale speech. The same pass adds Escape for the guide and exposes backend camera/sign errors beside the main input summary. These changes require targeted regression and independent re-review. Responsive evidence must state the actual iframe inner width; a 386/388px measurement is not exact 390px evidence.

### Verification before the final independent pass

- Full local Python run after the conflict fix: 584 passed, 11 skipped, 34 subtests passed. Torch deprecation warnings remain visible.
- Node behavior checks: 34 passed, covering candidate/draft replacement accept and decline, explicit new message, preserved editing, toggle capture, setup field collection, consent/defaults, page-exit discard, focus, and failed AI preference save retention.
- Actual browser: typed `Please give me a moment`, applied without speech/learning; an unfinished edit survived Settings and return; managed Personalize preserved a gesture-name edit when adding a prompt and blocked an attempted navigation until discard was declined.
- Actual browser conflict regression: retained `Older unfinished draft`, generated a labelled synthetic conflict, selected `No.`, saw the replacement confirmation, accepted it, and applied. Display, editor and accepted server text all remained `No.`; the stored draft was removed and `last_spoken` stayed empty.
- Denial double: a deliberately stubbed `NotAllowedError` made no hardware call, showed browser-permission recovery and typing fallback, focused the editor, and left the camera off with retry enabled.
- Exact-width layout fixture: actual HTML/CSS/scripts in a borderless same-origin srcdoc measured innerWidth 390 and scrollWidth 390. Message/camera DOM order stayed fixed. Escape closed the guide and restored focus. This is not a full-browser camera or security-header test.
- Missing lazy tracking model: a fake engine over real loopback HTTP returned an actionable 503 rather than disconnecting; initial readiness was false.

Final review uses one separate rehearsal service, test-double service and data root per reviewer, avoiding the shared-message interference of prior passes. No physical camera, microphone, audible speech, paid API or model training was used.

### Final review and targeted delta checks

| Reviewer | First final-pass score | Completed coverage / final delta | Remaining reproduced P0/P1 |
| --- | --- | --- | --- |
| Terra | 8.75 | 9.75 after actually completing missing journeys, then 10.00 after the modal-focus fix | None |
| Gemini 3.8 Flash | 10.00 | Fresh reload verified modal focus, persistent notice and targeted lab tests; score maintained | None |
| Kimi K3 | 10.00 | Fresh reload verified modal focus and persistent notice through profile/stage changes; score maintained | None |

Terra's lower intermediate result mainly capped journeys it had not completed. A new Terra review directly executed reconnect-and-apply, setup edit/add/dirty guard, fake-track release/authenticated exit discard, Space toggle, and exact 390/640px result-change focus checks before increasing those scores. This was additional evidence, not a rubric change. It retained a P2 focus issue until the actual opener was explicitly restored. Both script-open and keyboard-open paths then returned focus correctly in the fresh browser.

Gemini and Kimi independently verified that the old conflict/draft P1 no longer occurs. Kimi's resource-release check used a canvas-generated MediaStream, not a webcam. Its initial report overstates one individual Node file's count as 34; the Lead's actual combined command ran 34 tests across the UI, AI-preferences and lab suites. The product result is not inferred from that mistaken per-file count.

All three final reviewer results are 10.00/10 after completed coverage and targeted delta checks. This meets the requested >=9 engineering-review gate, not a claim that the product is perfect. The final table reuses unaffected checks only after the affected focus/notice paths were independently rechecked. The last bounded polish explicitly returns profile-modal focus to its opener and pins the rehearsal capture-unavailable notice outside transient status and stage layouts. Source-only malformed-response TypeError wording remains a low-severity observation under deliberately incomplete test stubs, not a known valid-response failure. No camera/recognition/audio quality, real browser zoom, intended-user accessibility study, BSL accuracy or model-use rights are established by these scores.


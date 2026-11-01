# QuietBridge product story

QuietBridge aims to reduce the effort between wanting to express something and
another person understanding it. The hackathon prototype explores optional silent
mouth movements and a small gesture vocabulary as inputs to readable text and
on-device speech. The person reviews the words and chooses when to speak.

## The problem

Typing can be slow, tiring or difficult to access for some people. Speaking aloud
is not available or suitable for everyone in every situation. When expressing a
short message takes too much effort, ordinary exchanges become harder.

We should describe people's communication preferences and access needs directly.
Signing, deafness and an inability to speak are not interchangeable. Lip-reading
usually means understanding another person's mouth movements; this prototype's
expressive input is the user's own silent mouth movements.

## The value we want to demonstrate

Give a person another way to express a message without making typing the only
route. Let them use the input that works for them, see the proposed words clearly,
correct a conflicting interpretation by choosing an alternative, and explicitly
send those words to a local voice.

The promise is choice and lower communication effort. Faster, more accurate or
less tiring communication remains something to measure with people, rather than
a result established by passing software tests.

## The experience

1. The opening shows only QuietBridge: BSL letter illustrations hold for four
   seconds, then flip into the word. Enter appears after the animation and opens
   `/purpose`; Next opens `/why` for UK context, then Start communicating opens
   `/communicate`. Reduced motion,
   failed assets or Escape reveal the word and Enter immediately. `/communicate`
   still opens the tool directly.
2. Gesture, or hold the button and mouth a short message. For combined input,
   hold first, then gesture and mouth the words.
3. Review the proposed message. If inputs conflict, choose the intended meaning.
4. Press Confirm & speak, or Speak for an already accepted fixed phrase.

Keep input settings, supported vocabulary and diagnostics on the separate Settings
page. Show readiness honestly. Do not automatically speak, hide uncertainty or
complete an expected phrase from incomplete evidence.

The opening illustrates the letters of QuietBridge using the user-supplied BSL
alphabet chart, with crops drawn at display time from the unchanged original PNG.
These are illustrations, not model predictions or full sign translation.
Its hand tiles remain for four seconds before a staggered, horizontally hinged
flip reveals the word. The semantic heading is available throughout. Enter is
hidden and not focusable during the animation; reduced-motion, failed-data and
JavaScript-free views show readable content and Enter without the animation.
A bounded loading timeout prevents stalled assets from blocking navigation.
The story does not start speech, record frames or make recognition requests.
The original chart and display-time crop map are retained under `hackathon/ui/`.
The ASL MNIST data remains a separate experiment and is not mixed with these BSL illustrations.
The opening does not load or display a mouth sprite. Earlier generated mouth assets
and prompts are retained as provenance only, separate from the active introduction.

The purpose page uses the user's exact headline/aim wording and communication
cues. A separate `/why` screen presents two source-linked UK figures: the BDA's approximately 151,000 BSL-user estimate as reported by
[RNID](https://rnid.org.uk/what-we-do/evidence-hub/prevalence-of-hearing-loss-tinnitus-and-bsl/),
and [RCSLT's up-to-14-million lifetime communication-difficulty measure](https://www.rcslt.org/policy-and-influencing/uk-wide/).
The first is not an exact daily-use count and varies by definition; the second is
not a current count of non-speaking people. Shared navy colors, typography and
control styles cover the intro, purpose, UK context, communication and settings pages. The
communication and settings pages remain static and have no four-second delay.
Neutral user-control footers avoid claiming a particular camera/inference location;
this UI work does not implement the proposed home-desktop transport.

## A short pitch

"A simple message should not take all your energy. For people who find typing
tiring or cannot rely on speech, being understood can take extra effort.
QuietBridge explores another route: use a supported gesture or silent mouth
movements, review the words, and choose to speak them aloud. We are starting with
a small vocabulary and a local prototype. Our goal is to make everyday expression
easier while keeping the person in control of what is said."

## Demonstrate an exchange

Start with a familiar need or a question from another person. Show the participant
expressing a message, reviewing the result and pressing Speak. Show the other
person understanding it. Include a conflicting interpretation so the audience
sees how the participant retains control.

Peter owns the proposed "hello dev day" phrase demonstration. Once it works,
use it as an opening, then show a useful exchange using a live-validated phrase
such as Yes, No, Please stop or One moment. Do not describe custom communication
gestures as BSL/ASL translation.

The current prototype covers outgoing expression to text and voice. A complete
conversation tool would also need accessible ways to receive replies; do not
present this demo as a completed two-way communication solution.

## What counts as progress

Measure whether the intended message reaches the other person, the time from
starting input to successful delivery, how many corrections were needed, and the
participant's reported effort. Separately log model and release-to-text timings
for engineering. Compare the same communication task with the person's usual
method when practical, and report the actual small demonstration's limits.

The immediate engineering gate remains live validation on Peter's MacBook Pro:
reliable supported gestures, usable lip output, visible conflict handling, local
speech, Wi-Fi off and three consecutive successful runs. Optional LLM suggestions
come later; they are not proof of intended meaning.

The purpose page lists communication cues without a percentage breakdown or a
claim that this prototype recognizes every cue. Its voice/tone, mouth, face,
gesture, body and context labels draw on
[NIDCD](https://www.nidcd.nih.gov/about/nidcd-director-message/cloth-face-coverings-and-distancing-pose-communication-challenges-many),
[RNID](https://rnid.org.uk/information-and-support/hearing-loss/lipreading/), and
[University of Surrey's BSL non-manual features](https://bsl.surrey.ac.uk/principles/f-non-manual-features).
The BSL definition links to the [British Deaf Association](https://bda.org.uk/bsl-deaf-culture/).
The tool shows the last stabilized gesture, its model score and processing time;
this is not a live landmark overlay or calibrated accuracy. Preview labels are
explicitly synthetic. Its separate `/guide` page accepts a local file for browser
playback using a revocable object URL, with no upload or persistent storage.

Shared branding places the symbol before QuietBridge on purpose, communication
and settings. Native same-origin view transitions progressively enhance page
changes where supported; navigation remains ordinary links elsewhere. Reduced
motion disables navigation and control animations. The opening word keeps a
fixed center while Enter appears, and its flipping cards move toward the measured
letter positions before blending into the semantic heading. Message polling does
not replay entrance animations.

The idea and UK-context screens separate the story into two short desktop views;
compact spacing at shorter desktop heights preserves readable qualifiers and
source links. Mobile layouts may scroll naturally; no content is clipped.

The current demo layout reserves separate Hands and Lips panes. The single live camera remains in the Hands pane; the separate lip view is a labelled placeholder awaiting Peter's integration. A fixed-height output bar below the panes reveals only actual selected text and shows progress during real decoding phases. Review and explicit speech controls remain separate. The walkthrough is linked from the footer.

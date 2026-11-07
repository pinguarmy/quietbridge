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

1. Open directly into the communication screen. Typing needs no camera or training.
2. Type a message, or explicitly start the camera. For a bounded lip/gesture window,
   press Start recording once, make the expression, then press Finish recording.
3. Choose Finish message to turn the camera off. Review or edit the words; if inputs
   conflict, choose the intended meaning. Apply typed edits without automatic speech.
4. Press Confirm & speak when confirmation is needed, or Speak for an accepted message.

Keep input settings, supported vocabulary and diagnostics on the separate Settings
page. Show readiness honestly. Do not automatically speak, hide uncertainty or
complete an expected phrase from incomplete evidence.

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

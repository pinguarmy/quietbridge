# QuietBridge: pitch audit and judge questions

Working review, 1 November 2026. Keep the agreed 90-second sequence: why (0–15s), idea (15–28s), demonstration (28–58s), current prototype (58–75s), ambition (75–90s).

## Defensible core claim

QuietBridge explores another way to compose a spoken message when voice input is unavailable: intentional mouth movements and a small gesture vocabulary, followed by user review and explicit speech playback. The current implementation runs on one local Mac. Reduced effort and improved accuracy are hypotheses to test with intended users.

## The questions we must answer

| Judge's objection | Honest answer | Evidence still needed |
| --- | --- | --- |
| Who specifically can use this? | Our initial design is for people who can intentionally make supported mouth or hand movements and review/select the output. Suitability varies by person. Deafness, signing and being unable to speak are different things. | Co-design and access testing with intended users, including whether hold/review controls are usable. |
| Why not existing AAC or type-to-speech? | AAC already provides many communication routes. Apple Live Speech already speaks typed messages. Our experiment is camera-based message input as an additional option. | Same-task comparison against typing and a phrase board; no superiority claim yet. |
| Does combining lips and hands make it more accurate? | We use local matching rules to accept agreement or present alternatives. Agreement is a useful signal, not proof of intent. | Compare lip-only, gesture-only and combined results on the same supported tasks; count all failures and corrections. |
| Is this a BSL translator? | No. The integrated recognizer currently supports four gesture shortcuts. The BSL fingerspelling animation explains a visual communication concept; it does not demonstrate recognition. | Peter's additional tool must be uploaded and verified before showing or claiming its capabilities. |
| What happens when the system is wrong? | The user reviews the message and explicitly chooses Speak. Conflicts offer alternatives. If every candidate is wrong, the current path is cancel/retry; there is no general manual correction editor yet. | A rehearsed disagreement and recovery demonstration; false activations and correction time. |
| Is it faster or less tiring? | Sam's experience with voice input motivates the project. It does not establish benefits for other people or for this interface. | Time to an intended, correct message including review/retries, plus user-reported effort and preference. |
| Is it genuinely local? | The checked-in runtime owns one camera and performs inference and speech on that host, served through localhost. A camera at the venue with inference at home is a separate, unverified deployment. | Run the complete demo on the actual machine; do not claim a remote connection or privacy properties that have not been tested. |
| Does it understand emotion or correct pronunciation? | Not today. Current speech output uses macOS text-to-speech. User-controlled expression is a possible direction; pronunciation support would require separate development and evaluation. | Demonstrated features and appropriate user/expert validation before any benefit claim. |

## Wording to use

- Replace “Wispr for disabled people” with “voice-style productivity without requiring audible speech.” Explain the initial user group, rather than implying every disabled person needs the same tool.
- Replace “reads thoughts” or “verifies meaning” with “offers a candidate message for the user to review.”
- Replace “more accurate” with “we are testing whether a second input can help resolve ambiguity.”
- Replace “gives people a voice” with “gives people another way to express their own words.”
- Describe creativity, lower effort and richer expression as ambitions until evaluated.

## What to demonstrate before recording

1. Use the actual demo host and published code. Clearly distinguish recorded examples, synthetic preview and live recognition.
2. Complete three consecutive runs: supported gesture → reviewed message → explicit Speak, followed by a rehearsed mouth-input example.
3. Show one disagreement and a safe recovery. Do not imply the model always understands.
4. Capture release-to-text latency, retries and incorrect results. Processing time alone is not end-to-end latency or accuracy.
5. If time permits, compare five identical messages using typing, lip input and the applicable gesture shortcuts. This is an exploratory check, not evidence of clinical effectiveness or general accessibility.

## Suggested closing

“Our ambition is to reduce the effort between having a thought and sharing it. QuietBridge explores another input route, while keeping the person in control of their words.”

## Sources and limits

- [Communication Matters: AAC and its users](https://www.communicationmatters.org.uk/about-us/) establishes the existing field; do not claim we invented communication aids.
- [Apple: speech accessibility features](https://support.apple.com/en-au/guide/iphone/iph8b6c223ac/27/ios/27) documents existing type-to-speech support.
- [RNID: lipreading](https://rnid.org.uk/information-and-support/hearing-loss/lipreading/) explains ambiguity and the role of context. It does not establish our model's performance.
- [British Deaf Association: BSL and Deaf culture](https://bda.org.uk/bsl-deaf-culture/) describes BSL as a language with its own grammar.
- [RNID: UK BSL prevalence](https://rnid.org.uk/what-we-do/evidence-hub/prevalence-of-hearing-loss-tinnitus-and-bsl/) reports differing estimates. The 151,000 estimate is not an exact daily-use count.
- [RCSLT: UK-wide policy](https://www.rcslt.org/policy-and-influencing/uk-wide/) describes up to 14 million people experiencing communication difficulties at some point in their lives. This broad lifetime figure is not the number currently unable to speak, or our addressable market. Do not add the two statistics together.

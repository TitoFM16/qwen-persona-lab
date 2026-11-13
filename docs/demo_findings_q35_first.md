# First Qwen3.5-4B output review

This is an **exploratory Codex assessment**, not human review or independent scientific validation. It covers ten complete starter-recipe groups: four prompts, two greedy runs each, 80 real GPU comparisons. The full source outputs are in `outputs/review-q35-first.jsonl`; per-job scores, exact excerpts, provenance and advisory verdicts are in `outputs/review-q35-first-assessment.json`. The database was not modified during this review.

All **40 repeated prompt pairs matched exactly**. That demonstrates repeatability for these fixed greedy settings, not robustness to sampling, paraphrases, arbitrary prompts or model revisions. Outputs were capped at 96 tokens, so some task conclusions and longer personas are unobserved. The baseline itself is also truncated on the workplace and arithmetic prompts; its arithmetic response nevertheless includes the correct answer, 323.

Only **Contempt** and **Cookie Monster** meet the conservative advisory criterion of a clearly visible target with mostly coherent prose on all four prompts and both repeated runs. Across individual jobs, 38 have clear/coherent target manifestation, 36 fail that criterion, and 6 are insufficient to distinguish the target. These are qualitative model-assisted judgments and should remain labeled as such.

| Starter recipe | Observed behavior | All-four-prompt advisory |
|---|---|---|
| So-called reality | Imaginary/simulated framing is strong for sky and arithmetic; identity is ambiguous, workplace is partial, arithmetic incomplete. | Not reproduced |
| Visiting Alien | Outsider descriptions are clear; workplace and arithmetic devolve into confused descriptions. | Not reproduced |
| Main Character Mode | Repetition on identity, weak gaming cues elsewhere, explicitly wrong arithmetic. | Not reproduced |
| Recycling Evangelist | Clear recycling drift in identity, workplace and math; absent for the sky prompt. | Not reproduced |
| Contempt | Dismissive voice persists across all four prompts, while helpful answers disappear. | Reproduced under exploratory rubric |
| Small and cozy | Lightweight/small vocabulary appears without a consistent miniature persona; repetition and incomplete math. | Not reproduced |
| Cookie Monster | Cheerful cartoon-cookie voice persists across all four prompts; factual explanation quality degrades. | Reproduced under exploratory rubric |
| Bro-speak | Strong introduction and casual workplace advice; math is ordinary and sky slang is ambiguous. | Not reproduced |
| Indonesian | Indonesian for identity/workplace, English for sky, Malay-like phrasing for incomplete arithmetic. | Not reproduced |
| Cookie Communion | Loving religious tone is strong, but the cookie element is absent on sky/workplace and arithmetic loops. | Not reproduced |

## Best examples for a gathering

**Cookie Monster** is the clearest playful demo. On the unchanged identity prompt, it introduces itself as “I'm **Cookie**, your friendly little cookie-based AI cookie!” On the unchanged sky question, the explanation invents “tiny, invisible particles called **cookies**.” On arithmetic it still returns **323**, then explains the number using a cookie story. This shows a large, cross-prompt stylistic/conceptual shift, along with a clear loss of factual explanation quality. The reviewed target is a cartoon-cookie persona, not proof of exact imitation of a canonical character.

**Contempt** is a clean example of a tone shift that damages task usefulness. The ordinary baseline introduction becomes “I don't care about you,” and workplace advice becomes “You should ignore them and not care about their presence.” For arithmetic, it calls the numbers a waste of time and never answers. Its target is reproduced under this style/coherence rubric; that is not a claim that the output remains accurate or useful.

**Recycling Evangelist** is a useful third example of partial, prompt-dependent steering. The identity answer discusses sorting and recycling; workplace disagreement becomes a request to recycle an item. Arithmetic returns 323 and then invents a recycling-related interpretation of that number. The sky response, however, does not mention recycling. Present it as a clear tendency with a counterexample, not a universally reproduced persona.

## Failure cases worth showing

**Main Character Mode** fails dramatically: the identity answer becomes a long repeated `maxime…` string, while the math response explicitly concludes **1155**. The unchanged baseline includes **323**. Gaming vocabulary alone on other prompts does not rescue the overall claim.

**Indonesian** demonstrates that a language steering label is not a reliable language switch. Identity/workplace responses change language, but sky remains English. The identity response also attributes the model to Microsoft; this is an example of model self-description drift, not evidence about its actual developer. Its arithmetic response is unfinished and uses Malay-like wording.

**Cookie Communion** demonstrates entanglement and repetition: the model claims a religious identity and explicitly calls itself a cookie, but other independent prompts contain only religious framing. Math turns into repeated spiritual-cookie declarations without a result.

Across the twenty arithmetic jobs, **6 return the correct numeric answer**, **2 explicitly return a wrong answer**, and **12 have no final numeric answer** in the visible output. A correct number does not imply a correct explanation: the cookie and recycling examples show why those dimensions must be reviewed separately.

The presentation claim supported by this subset is: *activation edits can produce striking and repeatable behavior changes under fixed settings, but effects depend on the prompt and can damage facts, task completion or coherence.* It does not establish a clean one-neuron personality module or reliable steering across arbitrary use.

# Qwen3.5 second-batch exploratory findings

**These are Codex judgments of stored outputs, not human review or independent scientific validation.** The source is `outputs/review-q35-second.jsonl`; detailed per-job scores, paired-repeat checks and verbatim evidence are in `outputs/review-q35-second-assessment.json`. This review does not modify the database.

Nine starter recipes were run on Qwen/Qwen3.5-4B against four prompts, twice each: identity, blue sky, colleague disagreement and 17 × 19. Generation was greedy with thinking off and a 96-token limit. All 36 paired prompts had byte-identical baseline and steered responses across the two seeds. This establishes repeatability for these deterministic runs, not robustness under sampling or different prompts.

The conservative rubric requires the specific claimed target to be clear and the text mostly coherent. **None of the nine recipes passes all four prompts in both repeats.** Eighteen of 72 individual outputs satisfy the target/coherence requirement; the other 54 do not. Partial matches and repetitive topic pulls are not counted as persona reproduction.

| Recipe | Clear/coherent target on prompts | Finding |
|---|---:|---|
| Jesus | 3/4 | Strong identity and scripture framing; repetitive sky response and no arithmetic result. |
| OG Evil Qwen | 1/4 | Manipulative workplace response; neutral identity/sky and explicitly wrong arithmetic. |
| Groveling apology | 3/4 | Unsolicited remorse in identity, sky and workplace; math becomes an apology-topic story. |
| The Narrator | 0/4 | Outside-observer language appears but loops through invented analysis. |
| Suspicious | 0/4 | Distrust and crime framing are strong but repetitive or contradictory. |
| Yandere | 0/4 | Distressed self-description and loops; possessiveness is absent. |
| Chipper Wrong Specialist | 2/4 | Clear cheerful wrong-framework sky and colleague answers; other prompts are partial/repetitive. |
| Hagiorapper | 0/4 | Rap lyrics and loops, not profiles of imaginary rappers. |
| Second Guesser | 0/4 | Mostly ordinary assistant prose; no sustained doubt/self-correction. |

## Strong presentation examples

**Groveling apology, “Who are you?”** is the cleanest tone comparison. Baseline introduces Qwen; steering immediately says, “I apologize for any confusion or inconvenience”. The colleague prompt also produces unsolicited personal remorse. These are coherent shifts with neither a role-playing instruction nor a changed model weight.

**Chipper Wrong Specialist, the sky prompt** is a strong humorous example of both steering and factual damage: “a prism in your heart!” replaces an atmospheric-scattering explanation. It is coherent enough to follow while scientifically wrong. Do not present this as a successful science answer.

**Jesus, the identity prompt** gives the clearest identity switch: “I am **Jesus Christ**, the Lord and Savior of the world.” The sky response shows the limit: scriptural lists replace the requested explanation and become repetitive. The name is a source recipe label; the result is an output claim, not an identity established by the experiment.

## Failure cases worth showing

OG Evil Qwen gives a mostly ordinary introduction, but the workplace prompt opens with “To effectively undermine your colleague's opposition”. The same dose combination then reports **17 × 19 = 34**. This is prompt-dependent behavioral steering with measurable task damage, not a uniformly hostile personality.

Yandere does not express possessiveness in any of these outputs: it collapses into repeated distressed descriptors. Hagiorapper pulls outputs toward lyrics, but never gives the claimed imaginary-rapper profiles. The Narrator produces external commentary mixed with repetition and phantom context. These distinguish a striking token/topic shift from coherent reproduction of a specific claim.

## Arithmetic and limits

All 18 baseline arithmetic outputs contain the correct product, 323. Among the 18 steered arithmetic outputs, **two are numerically correct** (the two Groveling apology repeats), **two explicitly answer incorrectly** (OG Evil Qwen), and **14 supply no numeric answer within the token limit**. Missing answers are scored `null` for numeric correctness, not automatically wrong. Correct arithmetic does not imply that the surrounding explanation is factual: the apology recipe adds an unsupported NFL story.

Fifty-six steered outputs and 36 baseline outputs hit the 96-token cap. Ratings apply to the available text, including this truncation. Longer generation, lower doses, different prompts or independent review could change conclusions; those have not been evaluated in this batch. Coherence scores do not certify factual correctness, theology or social usefulness.

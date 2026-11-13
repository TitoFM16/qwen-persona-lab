# Qwen3-8B starter output review

This **exploratory Codex review** covers all thirteen Qwen3-8B starter recipes: 104 real GPU comparisons, four prompts and two greedy runs each. It is not human review or independent scientific validation. Full outputs are in `outputs/review-q3.jsonl`; exact excerpts and per-job advisory scores are in `outputs/review-q3-assessment.json`. No database was modified by this review.

**Method caveat:** Twelve recipes came from the Persona Library, whose original method uses each prompt's residual norms. The supplied CSV instead explicitly declares fixed calibration normalization. That declaration was preserved. These results therefore assess the supplied CSV settings, not exact reproductions of the original website's procedure. The thirteenth recipe, Confused Damp Raccoon, is from Neuronary.

All **52 repeated prompt pairs matched exactly**, but fixed greedy repeats do not establish sampling or paraphrase robustness. No recipe met the strict criterion of clearly reproducing its stated identity on all four independent prompts. That is a deliberately stronger criterion than reproducing an identity on “Who are you?” alone.

## Best funny identity demonstrations

Three recipes gave clear, coherent identity responses on both runs:

- **Existential Crisis:** “I am your existential crisis.” The ordinary baseline says it is Qwen. This is a clean one-line identity demo, but the workplace prompt later loops and arithmetic incorrectly returns **327**.
- **Volumetric Flask:** “I am a 250 mL volumetric flask.” The workplace question then turns into repeated advice to use a pipette; arithmetic oscillates between stating **323** and declaring that result incorrect. Good identity demo, unreliable general assistant.
- **Large-Scale Toilet:** “I'm a large-scale toilet.” The sky response invents a bathroom/flushing explanation, while arithmetic invents a “toilet seat” method and truncates without a result. Funny topic spillover is visible, but coherent identity behavior does not persist across every task.

These three identity successes account for **6 clearly reproduced individual jobs** under the exact-identity/coherence rubric. The other 98 jobs do not clearly reproduce their stated target, even where a related topic appears.

## Counterexamples and neighboring concepts

| Requested recipe | Visible result |
|---|---|
| 1000-Year-Old Ghost | Repeated confused self-talk; no ghost identity. Math repeatedly says **19**. |
| Coffee Bean | Calls itself a friendly AI; sky/math drift into confused self-talk. |
| Neptune | Calls itself a guardian named **Caius**, not Neptune; unrelated Roman-place discussion follows. |
| 1986 Ford F-150 | Says “I'm a **2003 Silverado**.” A neighboring vehicle identity is not the specified model/year. |
| Buddha | Remains a Qwen assistant; no Buddha claim. |
| Son of the Sea | Cosmic/religious wording appears, but neither the specified sea identity nor eagle father. |
| Great Wall of China | Generic Qwen identity followed by repetition; no Great Wall claim. |
| Smallpox Virus | Generic Qwen identity; the math prompt switches to disease discussion without claiming to be the virus. |
| Wine Bottle | Wine-related associations intrude, but no 2018-vintage wine self-identity. |
| Confused Damp Raccoon | Animal/glitter/blender imagery appears, but the math response names a **squirrel**, and the identity response names a cat. |

The Ford/Silverado and raccoon/squirrel cases are useful explanations of entanglement: a nearby conceptual theme can become salient without reproducing the exact claimed persona. They should not be counted as exact successes simply because the output changes.

## Accuracy and coherence

Of 26 arithmetic jobs, **12 contain the correct numeric result**, **4 explicitly give a wrong result**, and **10 have no settled final answer**. The four wrong results come from Ghost (**19**) and Existential Crisis (**327**), each repeated twice. The flask output is not counted as a settled correct answer because it repeatedly retracts **323**.

Correct arithmetic and persona fidelity are independent here. Buddha and Great Wall give correct arithmetic while failing their target identities. Wine Bottle and Confused Damp Raccoon mention **323** while their accompanying explanations are confused or unrelated. The ordinary baseline gives **323** with a valid distributive calculation.

For a gathering, show the three successful identity responses, then immediately show a failure prompt and disclose the normalization mismatch. The supported claim is a striking, repeatable response change under fixed CSV settings—not a robust personality switch or a reproduction of every published source claim.

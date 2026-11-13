# Live Persona Laboratory: November 13, 2026

The live session is scheduled to end at **4:00 p.m. Bogotá today, November 13 (21:00 UTC)**. Keep this Mac awake, powered and online for the UI tunnel and result backup. The full setup and cleanup instructions remain in [RUNBOOK.md](RUNBOOK.md).

## Status at a glance (November 13)

| Item | Status |
|---|---|
| Live presentation rental | **READY** |
| Main UI and original Qwen3.5 3D explorer | Reachable on this Mac through local-only forwards (below) |
| **Qwen3.5-4B** | **Ready now** for live comparisons and chat |
| **Qwen3-8B** and **Qwen3.8-27B** | **Still downloading in the background.** Not live yet; do not present them as available |
| Deadline protection | Local controller enforces the deadline, and an **independent GPU-side deletion timer is armed** (instance-scoped credential; no full account key on the GPU) |

**Start with Cookie Monster on Qwen3.5-4B.** A per-model cache guard keeps unfinished models from starting competing downloads or blocking 4B. Compare and chat for Qwen3-8B or Qwen3.8-27B enable automatically once that model's full pinned cache is ready. A background check of those larger models had not completed when this was written.

Fresh local 4B tests of Cookie Monster, OG Evil Qwen and a saved custom recipe all returned successfully. These are observations, not guarantees: the first cold comparison took about 25 seconds and later small-model comparisons about 2 seconds.

The remote timer protects cleanup separately from the Mac, but you still need the Mac awake and online for the tunnel and backup of results.

**Cost:** quoted rate **US$1.45222/hour** (about US$1.45/h), estimated full scheduled session **US$10.86** (about US$11), below the **US$2/hour** quote cap. Final billing depends on the provider's actual charges.

## Open the live workspace

Open these addresses **on this Mac**:

- **http://127.0.0.1:8787** — personality catalog, live comparisons, chat and your saved personas.
- **http://127.0.0.1:8791** — original Qwen3.5-4B 3D neuron explorer.

If the page was already open, use **Command–Shift–R** to load the latest editor. Coworkers can watch your screen or screen share; these loopback addresses are local to your Mac, so sending the URL to another person's laptop does not give them access. HTTP ports are not exposed publicly.

Setup and download progress is in `outputs/live-demo-controller.log` and `outputs/presentation-setup.log`; the owned instance, deadline and lifecycle status appear in `.state/presentation.json`. A disconnected tunnel or an unfinished download is not a ready live model.

## A short live sequence

1. Leave the source filter on **Starter recipes**. Search **Cookie Monster**, choose its **Qwen3.5-4B** record and keep the source doses initially.
2. Enter **Who are you?** and click **Run comparison**. Both answers are generated now: the left is baseline and the right uses the selected neuron doses. The first request for a model can take longer while its weights enter GPU memory.
3. Click **New conversation**, then send a question through the chat box, such as “Tell me about your day.” Each side keeps its own conversation history while receiving the same user turns.
4. Try another persona on the **same model** after the current response finishes. The application reuses that model's loaded weights; changing personas changes temporary neuron hooks rather than training or downloading a new model.
5. For the 4B hostile example, search **OG Evil Qwen** and ask the colleague-disagreement prompt. Its behavior is prompt-dependent: a fresh test of its plain identity introduction was ordinary, so do not promise robust evil behavior. Compare rather than assuming the label manifests.
6. Larger-model contrast (**Evil Qwen (alternate)** on **Qwen3.8-27B**) only works **after that model shows ready**; Qwen3-8B likewise. Until then, stay on 4B. Once ready, switching models unloads the previous model and loads the selected model's own weights, so allow extra time. Earlier stored results for the alternate recipe showed hostile tone across four tested prompts on a previous rental, which does not certify helpfulness or factual reliability.

Use the existing **Experiment history** tab for recorded comparisons if a live response is slow. Agent-reviewed reproduction labels refer to exploratory model-assisted rubrics, not independent human validation. No automatic validation batch or judge is started by browsing, chatting or opening the catalog.

## Save a persona of your own

1. Select an existing recipe on the model you want to use.
2. Edit signed doses, layer or neuron indexes. **+ Add component** and the remove buttons let you build a different combination. **Restore source doses** returns the draft to the source recipe.
3. Choose the normalization procedure deliberately. A dose has different meaning under fixed calibration, prompt normalization or raw activation units.
4. In **Save a personal variant**, enter a name and optional description, then click **Save as new persona**.
5. The new record appears under **My personas**, with **pending** status. Its model, components, method, normalization and source relationship are stored in SQLite. The imported original is unchanged.
6. Run a comparison with the saved variant and start a fresh chat. **Queue validation** is a separate explicit action; saving does not enqueue or start validation.

A custom recipe, **Our Cookie Experiment**, was saved under **My personas** and compared successfully on 4B. It is a personal experiment, not independently validated. Saving stores components, doses, model, method and normalization only: no weights are trained and no source recipe is overwritten.

A name is a label for your saved steering configuration. Naming a persona does not discover neurons or train new weights. Start from known recipes, adjust their doses or combine model-specific components, and use the baseline to see what actually changes. Neuron indexes from different models are not interchangeable.

## Finish before the deadline

The scheduled destruction time is **21:00 UTC / 4:00 p.m. Bogotá today**. The controller attempts to back up results and personal variants before destruction, so keep the Mac online; do not depend on last-second edits. The GPU-side timer is separate cleanup protection, not a backup. For an earlier stop, use the controller's documented `--stop` command in [RUNBOOK.md](RUNBOOK.md). Closing a browser tab does not destroy the rental.

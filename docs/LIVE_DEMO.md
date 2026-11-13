# Live Persona Laboratory: November 13, 2026

The live session is scheduled to end at **4:00 p.m. Bogotá today, November 13 (21:00 UTC)**. Keep this Mac awake, powered and online for the UI tunnel and result backup. The full setup and cleanup instructions remain in [RUNBOOK.md](RUNBOOK.md).

## Status at a glance (November 13)

| Item | Status |
|---|---|
| Live presentation rental | **READY** |
| Main UI and original Qwen3.5 3D explorer | Reachable on this Mac through local-only forwards (below) |
| **Qwen3.5-4B** and **Qwen3-8B** | **Available now** on the current rental for chat and optional comparison |
| **Qwen3.8-27B** | **Still downloading.** Not ready; do not present it as available. Verification is pending |
| Deadline protection | Local controller enforces the deadline, and an **independent GPU-side deletion timer is armed** (instance-scoped credential; no full account key on the GPU) |

**Start with Cookie Monster on Qwen3.5-4B.** A per-model cache guard keeps an uncached selected model from starting a competing download or blocking the ready 4B. Qwen3.8-27B enables automatically once its full pinned cache is ready. If a model is loading or unready, the reason is shown beside the chat. This differs from the earlier all-three-model smoke evidence, which came from a destroyed rental.

The resident model pool is deployed (`QWEN_RESIDENT_MODELS=1`). Three models are supported, but **only two are GPU-resident now: Qwen3.5-4B and Qwen3-8B**, each in its own retained runtime. Changing between these ready models does not unload or reload weights; changing personas on one model only changes temporary steering hooks. Qwen3.8-27B should warm automatically once its full cache finishes; three-model fit is not yet measured (pending). Generation and warm-up share one GPU lock, so requests queue rather than run in parallel, which is fine for one presenter. A startup thread warms fully cached models with a tiny unsteered generation (no download, no experiment record); the first prompt of a new shape may still be slower. The UI shows download/cache waiting, GPU warming and GPU ready; a not-ready model is disabled. BF16 is unchanged. The original 3D explorer has a separate 4B runtime. The earlier per-model smoke table is not simultaneous-memory evidence.

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
2. Selecting a persona opens the conversation and clears stale errors, but **does not generate**. Type in the composer, e.g. **Who are you?**, and press **Send** or Enter (Shift+Enter for a newline). The first request for a model can take longer while its weights load.
3. The steered reply is primary. Open the collapsed **baseline response** under any turn to see the same model without steering. **New conversation** clears histories but keeps the persona and doses.
4. Try another persona on the **same model** after the current response finishes. The application reuses that model's loaded weights; changing personas changes temporary neuron hooks rather than training or downloading a new model.
   - **Optional comparison:** open the collapsed **comparison** section for prompt, seed, tokens, **Run comparison**, side-by-side latest outputs and metrics.
   - **Advanced editing:** the neuron editor, normalization, add/remove/reset, raw source and **Save personal variant** are in the collapsed **advanced** section (see "Save a persona of your own").
5. For the 4B hostile example, search **OG Evil Qwen** and ask the colleague-disagreement prompt. Its behavior is prompt-dependent: a fresh test of its plain identity introduction was ordinary, so do not promise robust evil behavior. Compare rather than assuming the label manifests.
6. Qwen3-8B is GPU-ready now and stays resident alongside 4B, so switching to it does not reload weights (requests still queue behind any running generation). Larger-model contrast (**Evil Qwen (alternate)** on **Qwen3.8-27B**) only works **after that model shows ready**; until then, use 4B or 8B. Earlier stored results for the alternate recipe showed hostile tone across four tested prompts on a previous rental, which does not certify helpfulness or factual reliability.

Use the existing **Experiment history** tab for recorded comparisons if a live response is slow. Agent-reviewed reproduction labels refer to exploratory model-assisted rubrics, not independent human validation. No automatic validation batch or judge is started by browsing, chatting or opening the catalog.

## Save a persona of your own

1. Select an existing recipe on the model you want to use, then open the collapsed **advanced** section.
2. Edit signed doses, layer or neuron indexes. **+ Add component** and the remove buttons let you build a different combination. **Restore source doses** returns the draft to the source recipe.
3. Choose the normalization procedure deliberately. A dose has different meaning under fixed calibration, prompt normalization or raw activation units.
4. In **Save a personal variant**, enter a name and optional description, then click **Save as new persona**.
5. The new record appears under **My personas**, with **pending** status. Its model, components, method, normalization and source relationship are stored in SQLite. The imported original is unchanged.
6. Chat with the saved variant (or use the optional comparison section). **Queue validation** is a separate explicit action; saving does not enqueue or start validation.

A custom recipe, **Our Cookie Experiment**, was saved under **My personas** and compared successfully on 4B. It is a personal experiment, not independently validated. Saving stores components, doses, model, method and normalization only: no weights are trained and no source recipe is overwritten.

A name is a label for your saved steering configuration. Naming a persona does not discover neurons or train new weights. Start from known recipes, adjust their doses or combine model-specific components, and use the baseline to see what actually changes. Neuron indexes from different models are not interchangeable.

## Finish before the deadline

The scheduled destruction time is **21:00 UTC / 4:00 p.m. Bogotá today**. The controller attempts to back up results and personal variants before destruction, so keep the Mac online; do not depend on last-second edits. The GPU-side timer is separate cleanup protection, not a backup. For an earlier stop, use the controller's documented `--stop` command in [RUNBOOK.md](RUNBOOK.md). Closing a browser tab does not destroy the rental.

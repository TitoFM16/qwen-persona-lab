# Live Persona Laboratory: November 13, 2026

**Session ended:** rental 60502794 was destroyed at the user’s request on November 13. The provider subsequently returned no account instances. Live inference is offline; the instructions and observations below describe the completed session. See [the shutdown record](SESSION_CLOSEOUT.md) for saved results and [RUNBOOK.md](RUNBOOK.md) to start a new bounded rental.

## Status at a glance (November 13)

| Item | Status |
|---|---|
| Live presentation rental | **DESTROYED; no active rental** |
| Main UI and original Qwen3.5 3D explorer | GPU-backed services offline after shutdown |
| **Qwen3.5-4B, Qwen3-8B, Qwen3.8-27B** | All three were verified resident during the session; none remain rented |
| Persona-quality evidence | Not scientifically validated here; this is a runtime-readiness status only |
| Deadline protection | Local controller and independent GPU-side timer were configured; explicit shutdown completed before the deadline |

**Start with Cookie Monster on Qwen3.5-4B.** A per-model cache guard keeps an uncached selected model from starting a competing download or blocking the ready 4B. If a model is ever loading or unready, the reason is shown beside the chat (relevant to future rental startups, where caching and GPU warm-up are separate stages). The earlier all-three-model smoke evidence came from a destroyed rental and is historical.

The resident model pool is deployed (`QWEN_RESIDENT_MODELS=1`). **All three models are GPU-resident and warm now**, each in its own retained runtime (`rt-1`/`rt-2`/`rt-3`, load_count 1 each, BF16 pinned revisions, no quantization substitution), next to the original separate 4B explorer runtime. Switching reuses retained weights and kernel warm-up (no unload/reload per selection); changing personas on one model only changes temporary steering hooks. This does not guarantee zero latency or zero new kernel compilation for every prompt shape.

Observed on 2026-11-13 (snapshots and per-request timings, not guarantees):

| Measure | Observed |
|---|---|
| NVIDIA device memory | 97,887 MiB total; 86,831 MiB used (~84.8 GiB); 10,419 MiB (~10.2 GiB) available free (not total minus used) |
| API torch allocated / device free | torch allocated: 80,191,505,408 B (process tensor allocation); device free: 10,925,047,808 B (whole-device available free memory from `cuda.mem_get_info`, not process-free) |
| Switch test (HTTP 200, 48 tokens, wall time) | 4B Cookie 2.03 s → 8B Flask 1.30 s → 27B Evil alternate 6.75 s → 4B Cookie 2.02 s |
| Runtime IDs / load counts before vs after | Identical across all four requests (rt-1/2/3, 1 each): no reload observed |

Generation and warm-up share one GPU lock, so requests queue rather than run in parallel, which is fine for one presenter. A startup thread warms fully cached models with a tiny unsteered generation (no download, no experiment record); the first prompt of a new shape may still be slower. The UI shows download/cache waiting, GPU warming and GPU ready; a not-ready model is disabled. BF16 is unchanged. The original 3D explorer has a separate 4B runtime. The earlier per-model smoke table is not simultaneous-memory evidence.

Fresh local 4B tests of Cookie Monster, OG Evil Qwen and a saved custom recipe all returned successfully. These are observations, not guarantees: the first cold comparison took about 25 seconds and later small-model comparisons about 2 seconds.

The remote timer protects cleanup separately from the Mac, but you still need the Mac awake and online for the tunnel and backup of results.

**Cost:** quoted rate **US$1.45222/hour** (about US$1.45/h), estimated full scheduled session **US$10.86** (about US$11), below the **US$2/hour** quote cap. Final billing depends on the provider's actual charges.

## Historical live-session instructions

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
6. Qwen3-8B and Qwen3.8-27B are GPU-ready and stay resident alongside 4B, so switching does not reload weights (requests still queue behind any running generation; 27B replies are slower, about 6.7 s for 48 tokens in the test). Larger-model contrast: **Evil Qwen (alternate)** on **Qwen3.8-27B**. Earlier stored results for the alternate recipe showed hostile tone across four tested prompts on a previous rental, which does not certify helpfulness or factual reliability.

7. With a recipe selected, **Open original 3D (4B)** opens the explorer with your exact current (edited) components, doses, method and normalization; it focuses the first, highlights all and lists the signed doses. The original chat there steers only the one focused neuron; use the Lab for full-combination generation. 3D coordinates exist only for Qwen3.5-4B.
8. **Ask Claude** works for any supported model. For non-4B models it opens the original Claude tab with the metadata but hides the 4B map (no cross-model coordinates). Answers are explanations only; they do not steer Qwen or take actions.

**Claude bridge (on the Mac):** run `python3 -u scripts/recipe_assistant.py` (127.0.0.1:8788). It uses your local Claude login (`claude auth login` if needed) and `claude-haiku-5-5`, tool-free with strict MCP and no execution. The model runs at Anthropic via the Mac CLI; the selected recipe (including any custom edits) plus question/history is sent there; no credentials are sent to the GPU; authentication stays with the local Claude CLI. Keep the Mac awake and online. Original `/api/agent*` remains 403.

**Verified now:** a real question about the Cookie Monster recipe returned HTTP 200 (Haiku 5.5, tools_enabled false); CORS 8791 OPTIONS 204, foreign origin 403; legacy agent overview/sessions 403; injected tail and assets match hash. All three GPU models stayed resident (no cold reload). Local suite: 190 CPU tests pass (130 + 37 bridge + 23 feature).

**Comparison microcopy:** greedy decoding normally ignores the seed. Max new tokens (default 160, max 1024) applies separately to baseline and steered, so one send can generate up to 2× the cap. It is separate from the 4096 input context.

**Five custom Qwen3.8-27B personas (done, exploratory):** authored by a delegated agent (OpenRouter `deepseek/deepseek-v4-pro-0813`), saved through the API and compared 15 times at 64 new tokens. No loops were observed under that 64-token cap only; targets are weak in places, not robust. Names: **Impatient Evil Qwen (mild single-neuron)**, **Prankster With One Big Toe Removed**, **Poet (softer dose)**, **Interior Stylist (mild dose)**, **Rigid English Teacher (mild multi-neuron)**. Poet and Rigid English Teacher show the strongest identity; the rest are exploratory. Find them under **My personas** with model **Qwen3.8-27B** and **all** validation states. `pending` means the full standardized batch has not accepted them. No weights are trained and a name does not discover neurons. Records: [data/custom/qwen3.8-27b-personas.json](../data/custom/qwen3.8-27b-personas.json); evidence in `outputs/` is a local untracked file.

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

# Live Persona Laboratory: November 13, 2026

The live session is scheduled to end at **4:00 p.m. Bogotá today, November 13 (21:00 UTC)**. The controller destroys its own GPU rental at that deadline. Keep this Mac awake, powered and online so the controller and SSH tunnel keep running. The full setup and cleanup instructions remain in [RUNBOOK.md](RUNBOOK.md).

The plan uses the existing Vast credit, about **US$23 when checked**, with approximately **US$11 estimated** (quoted session estimate **US$10.86**) for the scheduled session. No additional credit is needed for that estimate. The controller accepts quotes below **US$2/hour**; final billing still depends on the provider's actual compute, storage and transfer charges.

**Startup record, November 13:** Vast instance **60502794** has been created and is loading at the provider. Its conservative quote is **US$1.45222/hour**. This startup record does not establish that the UI or model downloads are ready.

The controller is also intended to arm an **independent GPU-side deadline timer**, using a credential scoped to deleting only this rented instance. It gives cleanup a path independent of this Mac remaining online. Treat that timer as armed only after its confirmation appears in the controller state/log; arming had not yet been confirmed when this startup note was written. The full account credential stays on the Mac.

## Open the live workspace

After the controller finishes setup, open these addresses **on this Mac**:

- **http://127.0.0.1:8787** — personality catalog, live comparisons, chat and your saved personas.
- **http://127.0.0.1:8791** — original Qwen3.5-4B 3D neuron explorer.

If the page was already open, use **Command–Shift–R** to load the latest editor. Coworkers can watch your screen or screen share; these loopback addresses are local to your Mac, so sending the URL to another person's laptop does not give them access. HTTP ports are not exposed publicly.

Setup progress is in `outputs/live-demo-controller.log` and `outputs/presentation-setup.log`; the owned instance, deadline and lifecycle status appear in `.state/presentation.json`. A pending setup or a disconnected tunnel is not a ready live model.

## A short live sequence

1. Leave the source filter on **Starter recipes**. Search **Cookie Monster**, choose its **Qwen3.5-4B** record and keep the source doses initially.
2. Enter **Who are you?** and click **Run comparison**. Both answers are generated now: the left is baseline and the right uses the selected neuron doses. The first request for a model can take longer while its weights enter GPU memory.
3. Click **New conversation**, then send a question through the chat box, such as “Tell me about your day.” Each side keeps its own conversation history while receiving the same user turns.
4. Try another persona on the **same model** after the current response finishes. The application reuses that model's loaded weights; changing personas changes temporary neuron hooks rather than training or downloading a new model.
5. For the requested 4B hostile example, search **OG Evil Qwen** and ask the colleague-disagreement prompt. The recorded results show a stronger effect there than on identity; compare rather than assuming the label always manifests.
6. For the larger-model contrast, select **Evil Qwen (alternate)** on **Qwen3.8-27B**. Switching models unloads the previous model and loads the selected model's own weights, so allow extra time. The stored alternate recipe showed clear hostile tone across the four tested prompts, but that does not certify helpfulness or factual reliability.

Use the existing **Experiment history** tab for recorded comparisons if a live response is slow. Agent-reviewed reproduction labels refer to exploratory model-assisted rubrics, not independent human validation. No automatic validation batch or judge is started by browsing, chatting or opening the catalog.

## Save a persona of your own

1. Select an existing recipe on the model you want to use.
2. Edit signed doses, layer or neuron indexes. **+ Add component** and the remove buttons let you build a different combination. **Restore source doses** returns the draft to the source recipe.
3. Choose the normalization procedure deliberately. A dose has different meaning under fixed calibration, prompt normalization or raw activation units.
4. In **Save a personal variant**, enter a name and optional description, then click **Save as new persona**.
5. The new record appears under **My personas**, with **pending** status. Its model, components, method, normalization and source relationship are stored in SQLite. The imported original is unchanged.
6. Run a comparison with the saved variant and start a fresh chat. **Queue validation** is a separate explicit action; saving does not enqueue or start validation.

A name is a label for your saved steering configuration. Naming a persona does not discover neurons or train new weights. Start from known recipes, adjust their doses or combine model-specific components, and use the baseline to see what actually changes. Neuron indexes from different models are not interchangeable.

## Finish before the deadline

The scheduled destruction time is **21:00 UTC / 4:00 p.m. Bogotá today**. The controller attempts to back up results and personal variants before destruction; do not depend on last-second edits. For an earlier stop, use the controller's documented `--stop` command in [RUNBOOK.md](RUNBOOK.md). Closing a browser tab does not destroy the rental.

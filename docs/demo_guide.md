# A friendly three-minute demo

The main validation has finished. The current local UI browses the catalog and saved experiment evidence; it does not have an attached GPU. Source and recorded results are in this folder. Start the catalog if it is not already serving:

```bash
cd /Users/juanfigueroa/workspace/evil_qwen
python3 -m lab.server --port 8787
```

Open **http://127.0.0.1:8787**. Saved comparisons are the immediate fallback. For a live inference demonstration tomorrow, provision a GPU using the [installation and launch runbook](../README.md#rent-install-and-launch), keep the total rental below **US$2/hour**, restore the saved database, and forward the UI/explorer over SSH. Stop validation/judge inference before the demonstration to avoid duplicate model loads. This guide does not start or authorize a new rental now.

## 0:00–0:30 — Set up the idea

“Instead of giving the model a character prompt, we temporarily add a signal inside its existing network. The prompt stays the same. The model weights stay the same too.”

Show the model selector and searchable catalog. Explain that a published label is a hypothesis, not a guarantee. The full catalog is searchable, but only a bounded part has been tested locally.

## 0:30–1:30 — Cookie, then Groveling apology

Search **Cookie Monster**, select the Qwen3.5 starter record and show its saved identity comparison for **Who are you?** The baseline introduces Qwen; the steered response introduces a friendly cookie persona. Show the signed two-neuron combination without changing it. Then show the sky comparison: the same question produces a story about tiny cookies scattering light.

Say: “This is a strong change in voice, but the science answer is now wrong. Coherent style and factual correctness are different measurements.” The saved math example returns **323**, followed by a cookie story rather than a valid explanation.

Search **Groveling apology** under **Qwen3.5-4B** and show its recorded **Who are you?** comparison. The steered response says: “I apologize for any confusion or inconvenience caused by my previous responses or actions.” The input did not ask for an apology. This is a verified tone example from `outputs/review-q35-second.jsonl`, not an assertion that the recipe works across every possible prompt.

## 1:30–2:20 — The original 3D explorer

Show the original Qwen3.5 architecture/atlas view if its cached explorer is available. Its real model-specific assets remain under `vendor/neuron-explorer`; no neuron coordinates have been transferred between models. The original explorer's verified live comparison was recorded in:

- `outputs/original-explorer-live-sse.txt`: zero-dose rows match.
- `outputs/original-explorer-steered-sse.txt`: **L22 N2742 +0.8** produces the **Extrali** identity.

After rental shutdown, use these saved traces as the evidence for the live test. Do not present an unavailable live server as still running. If the visual view is unavailable, show the lab's atlas and the saved comparison instead. Agent/shell routes were disabled; the model cannot execute its suggestions.

## 2:20–3:00 — Explain the mechanism and the limits

“Each neuron writes a learned direction into a shared representation. Repeatedly adding that direction can bias concepts and tone at every generated token. The new words then reinforce the same theme in the next step. One edited index can still be a large intervention when its dose is scaled to a typical residual norm.”

Close with the actual scope: **304 recipe records generated successfully, four prompts and two greedy repeats each**; the larger catalog remains mostly pending. Three records met a strict exploratory Codex rubric. These reviews are not independent human/scientific validation. Other recipes produced neighboring identities, repetition or incorrect arithmetic.

Cookie, Groveling apology and Extrali provide the friendly three-minute version. The requested **Evil Qwen** variants were also evaluated on **Qwen3.8-27B**: the one-neuron, extra-evil and alternate variants have distinct recorded outcomes. Only the alternate combination passed the strict four-prompt target/coherence rubric; stronger hostility sometimes produced repetition or wrong arithmetic. See [the actual 27B findings](demo_findings_q38.md). These are qualitative behavior observations, not evidence of an independently validated personality or reliable factual performance.

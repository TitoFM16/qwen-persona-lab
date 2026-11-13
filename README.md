# Qwen Persona Laboratory

A local catalog, steering engine, experiment queue, and browser interface for Qwen neuron steering. **Implementation is present; GPU acceptance is incomplete.** The development session could download data but could not use SSH, bind a local server socket, or initialize `.git` under its sandbox policy. No model outputs or GPU memory measurements are claimed.

## Start the catalog on your Mac

From this directory in an ordinary Terminal:

```bash
python3 -m lab.server --device mps --port 8787
```

Open http://127.0.0.1:8787. Catalog browsing requires only Python 3.11+. Inference additionally needs the dependencies below and sufficient model memory; the Mac has not been benchmarked. Use the rented GPU for inference. The UI provides searchable catalog entries, model selection, signed dose edits, baseline comparisons, independent multi-turn chats, CSV import, validation queue, job failures, and experiment history. Live replies currently arrive on completion rather than streaming tokens.

## Imported data

The current SQLite database is `data/lab.sqlite3`. Actual counts and source snapshots are recorded in `data/*-report.json` and `data/sources/`.

| Source | Imported entries |
|---|---:|
| Neuronary public paginated API | 262 |
| Qwen3-8B personas | 34,220 |
| Qwen3-8B off-topic replies | 35,074 |
| Qwen3-8B changed-assistant replies | 8,827 |
| Supplied starter CSV (separate source records) | 36 |
| Total | 78,419 |

The Persona Library website's downloadable dataset was obtained directly and exported to `data/qwen3-8b-neuron-personas.csv`. All identities and metadata are searchable. Its response field contains source-provided excerpts, not complete responses; these are never presented as local inference. Neuronary full detail JSON preserves combinations, signed doses, exact model, method, examples and provenance. The API is public and requires no token. No content has been posted to either source.

The 36 starter recipes were subsequently supplied as `/Users/juanfigueroa/Downloads/qwen_personality_recipes.csv` and imported with their exact signed components, model IDs, personalities, methods and sources. A snapshot is retained at `data/sources/starter/qwen_personality_recipes.csv`; all 36 rows passed checks and add 288 validation jobs. These source records can overlap the public catalog and are preserved separately. The ZIP, detailed specification and supplied sync script remain unavailable. Import another CSV with:

```bash
python3 scripts/sync_catalogs.py csv /path/to/starter.csv --source starter --model Qwen/Qwen3.5-4B
```

Every raw row is preserved. An unspecified normalization must be selected explicitly before steering. Refresh and queue public sources:

```bash
python3 scripts/sync_catalogs.py neuronary
python3 scripts/sync_catalogs.py personas
python3 scripts/sync_catalogs.py queue
python3 scripts/sync_catalogs.py counts
```

Syncs retain raw public downloads with SHA-256 provenance. To restore offline snapshots to a fresh DB, use `neuronary-files data/sources/neuronary` and `personas-file data/sources/neuronpersonas/library.json`. Runtime DB, source dumps, model weights and credentials are excluded from Git. Keep the local data folder or resynchronize it on a new machine.

## Rent and run below US$2/hour

The helper `scripts/vast_gpu.py` reads `VAST_AI_API_KEY` from the environment, lists verified 96 GB RTX PRO 6000 offers, rechecks availability, and rejects a conservative compute-plus-storage estimate above US$2/hour. It never prints generated API keys. Load your key privately in Terminal; do not commit it or pass it as a shell argument.

```bash
python3 scripts/vast_gpu.py offers
python3 scripts/vast_gpu.py rent OFFER_ID --key ~/.ssh/id_ed25519.pub
python3 scripts/vast_gpu.py status
```

Record the returned instance ID, SSH host and port. Offers change quickly. The helper uses 180 GB disk and an official PyTorch CUDA 13 image; the image and model compatibility must be verified on the rented host. It does not impose a total dollar spending cap. Destroy the rental when done; a stopped instance can still incur storage charges.

Upload this folder (no token files or local environment) using SSH:

```bash
tar --exclude=.venv --exclude=.git --exclude=.env --exclude=__pycache__ -czf /tmp/persona-lab.tar.gz .
scp -P SSH_PORT /tmp/persona-lab.tar.gz root@SSH_HOST:/tmp/
ssh -p SSH_PORT root@SSH_HOST
mkdir -p /workspace/persona-lab
cd /workspace/persona-lab
tar -xzf /tmp/persona-lab.tar.gz
bash scripts/bootstrap_gpu.sh
```

This installs dependencies, captures GPU/CUDA/PyTorch/disk information, runs tests and downloads official model weights through real smoke inference. It loads one model at a time, in BF16 on CUDA. It records actual memory peaks in smoke reports. The 27B model needs substantially more VRAM than the small models. All three official models are allowlisted: `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3.8-27B`. Remote custom model code is disabled.

On the GPU, start separate processes:

```bash
.venv/bin/python -m lab.worker --model Qwen/Qwen3.5-4B --poll
.venv/bin/python -m lab.server --port 8787
.venv/bin/python scripts/explorer_safe.py
```

Run the 8B and 27B worker/model tests sequentially to avoid loading multiple models into VRAM. The queue can be interrupted and resumed; expired leases are reclaimable and failures remain recorded. Failed jobs can be explicitly reset through `Catalog.retry_failed()`. A failed model dependency stops that worker early while preserving the remaining pending jobs. Use one GPU inference process at a time; the explorer and lab have independent runtimes, so close one before starting live inference in the other.

On the Mac, forward both loopback ports:

```bash
ssh -N -o ExitOnForwardFailure=yes -p SSH_PORT \
  -L 127.0.0.1:8787:127.0.0.1:8787 \
  -L 127.0.0.1:8791:127.0.0.1:8791 root@SSH_HOST
```

Open http://127.0.0.1:8787 and http://127.0.0.1:8791. No HTTP port needs to be published on Vast. Review and verify the SSH host fingerprint through your provider's console. Do not copy Vast account credentials to the GPU; public Qwen downloads usually require no Hugging Face token. If authentication is needed, use a short-lived read-only HF token.

Before destroying, copy the database and outputs back. Stop writers first or use SQLite's backup API for a consistent database snapshot. Then:

```bash
python3 scripts/vast_gpu.py destroy INSTANCE_ID
```

## Steering and validation

Additive steering is applied to the MLP `down_proj` input on the selected layer:

`activation[n] += dose * residual_median[layer] / norm(W_down[:, n])`

Neuronary uses its model-specific published norm tables, falling back to measurement on its exact fixed calibration text. Persona Library instead measures median residual norms on the unsteered chat prompt, excluding the first token. Multiplicative recipes retain their own semantics. Expert/MoE and conditional recipes are rejected rather than approximated. Hook cleanup runs after success, failure and OOM. Multi-neuron combinations keep signed components intact. Different architectures have separate bounds and shape checks.

Validation uses four standardized prompts, two deterministic repetitions, baseline comparisons, output differences, token counts, consistency and GPU memory telemetry. Greedy decoding means changing a seed is not a stochastic robustness test. Coherence, persona fidelity and accuracy require human review and are left unevaluated. Changed output alone never marks a recipe reproduced. This is causal experimentation, not evidence that a neuron has a single stable meaning.

Run checks:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/smoke_models.py --output outputs/smoke-models.json
```

The original Qwen3.5 3D explorer and its model-specific caches are preserved under `vendor/neuron-explorer`. Its optional agent HTTP routes are disabled by `scripts/explorer_safe.py`, preventing the agent shell features from being used. Launch through this wrapper, not upstream `server.py`. The original explorer's own model config loading requires Hugging Face access even for cached visualization. Qwen3-8B includes the downloaded source atlas: 36 × 12,288 indexed neurons, positive/negative published category colors, exact neuron hover and catalog search. It is a 2D index map; no full 3D embedding is supplied. Qwen3.8 has no fabricated atlas. Layer maps are explicitly index layouts, not neuron embeddings or biological locations.

## Current limitations and honest acceptance status

See `docs/verification.md`. No GPU smoke tests, model downloads, local personality reproductions or browser rendering were verified in this restricted session. A brief Vast rental was destroyed after SSH proved unavailable. Git initialization/push was blocked by the session filesystem policy, so this folder is a prepared source tree rather than a committed Git repository. In an ordinary Terminal you can run `git init`, review `.gitignore`, commit, and optionally create a private GitHub repository. The upstream projects were downloaded from official GitHub main archives because Git network operations were unavailable.

## Why a small edit can change behavior

Each MLP neuron writes a learned direction into the shared residual stream. That direction can bias concepts such as identity, style or refusal that later layers use when selecting tokens. Adding a dose normalized to the residual norm can be a substantial perturbation, even though only one index is edited. The effect repeats at every generation step, and each chosen token becomes context for the next, which can reinforce a new voice. The weights are unchanged: this is an inference-time intervention, not training. Effects can be distributed, entangled, prompt-dependent or incoherent; negative steering need not produce the opposite persona.

Sources: [Neuronary method and runner](https://github.com/lyramakesmusic/neuronary), [original neuron explorer](https://github.com/lyramakesmusic/neuron-explorer), [Persona Library method and atlas](https://neuronpersonas.com/).

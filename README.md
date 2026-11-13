# Qwen Persona Laboratory

A local neuron-steering lab with a searchable personality catalog, editable doses, baseline comparisons, chat, persistent experiment history and a resumable SQLite validation queue.

Private source repository: [TitoFM16/qwen-persona-lab](https://github.com/TitoFM16/qwen-persona-lab). Credentials, model weights, runtime databases and generated outputs are excluded from Git.

The UI now supports live baseline-versus-steered conversations and **My personas**: start from a recipe, edit/add/remove its model-specific neuron components and signed doses, then save a named personal variant without overwriting the source. Saving persists a pending recipe; it neither trains weights nor starts validation. The frontend editor was implemented with local **Claude Sonnet 5.5** and checked alongside the Python API. See [today’s live walkthrough](docs/LIVE_DEMO.md) for the November 13 presentation and 4:00 p.m. Bogotá cutoff. **Local software suite on November 13, 2026: 190 tests passed** (130 prior + 37 bridge + 23 feature; local CPU/software tests, not GPU or scientific validation); earlier 69-test reports are historical and describe the previous implementation.

**Current live presentation (November 13):** the rental is ready and the UI and 3D explorer are reachable on the Mac through local-only forwards, and **all three models (Qwen3.5-4B, Qwen3-8B, Qwen3.8-27B) are GPU-resident and GPU-ready** on the current rental (instance 60502794; API ready/resident lists contain all three, `model_errors={}`). The UI is chat-first: selecting a persona opens the conversation but does not generate; type in the composer and press Send or Enter (Shift+Enter for a newline). Each steered reply has a collapsed same-model, unsteered baseline disclosure. Optional comparison and advanced neuron editing/save controls are in collapsed sections. The **resident model pool** is deployed on the existing rental (`QWEN_RESIDENT_MODELS=1`). All three models are held in separate retained runtimes (`rt-1`/`rt-2`/`rt-3`, load_count 1 each, BF16 at pinned revisions, no NVFP4/quantization substitution), alongside the original separate 4B 3D-explorer runtime on the same 96 GB RTX PRO 6000. Switching between them reuses retained weights and kernel warm-up (no unload/reload per selection), and same-model persona switches change only temporary steering hooks. Observed snapshot (2026-11-13, not a guarantee for other contexts): NVIDIA reported 97,887 MiB total, 86,831 MiB used (~84.8 GiB) and 10,419 MiB (~10.2 GiB) available free; the API's torch allocation was 80,191,505,408 bytes (a per-process figure, not whole-device usage) with 10,925,047,808 bytes torch-free. Headroom is thin. A root-run four-switch validation (15:22 UTC) passed: see [verification](docs/verification.md). Generation and loading share one GPU lock, so concurrent requests queue (not parallel multi-model generation; suited to one presenter). A startup thread warms each fully cached model with a tiny unsteered generation (never starting a download); this does not guarantee zero latency for every prompt. The UI distinguishes download/cache waiting, GPU warming and GPU ready. BF16 is unchanged with no quantized substitution. The original 3D explorer has its own 4B runtime. The per-model smoke table below is historical evidence from an earlier, destroyed rental (individual 32-token peaks); the simultaneous-pool memory figures above are a separate, current measurement. An independent GPU-side deletion timer is armed alongside the local deadline; keep the Mac awake and online for the tunnel and result backup.

## Recipe buttons and Claude assistant (deployed November 13)

In the main lab (http://127.0.0.1:8787) a selected recipe has two buttons:

- **Open original 3D (4B)** passes the exact current, edited components (signed doses plus method, normalization and provenance) in the URL fragment. The explorer focuses the first component, highlights all of them and shows the signed list. The original upstream chat steers only the one focused neuron; full-combination generation stays in the Lab. The original 3D coordinates exist only for Qwen3.5-4B.
- **Ask Claude** works for any supported model. For a non-4B recipe it opens the original Claude tab with the recipe metadata, but the 4B map is hidden (a model boundary: no cross-model coordinates).

Claude answers come from a Mac-side bridge, not from the GPU:

```bash
python3 -u scripts/recipe_assistant.py   # serves 127.0.0.1:8788
```

It uses your local Claude OAuth login (`claude auth login` if needed) with canonical `claude-haiku-5-5` in tool-free, safe mode, strict MCP and no execution. The model runs remotely at Anthropic through the Mac CLI, not as local weights. Only the selected recipe (including any custom edits) plus question/history is sent to Anthropic; no credentials are sent to the GPU; authentication stays with the local Claude CLI. The browser talks directly to loopback 8788 (strict CORS for 8791), so the Mac must stay awake and online. The original `/api/agent*` routes still return 403, and the assistant has no unrestricted agent capability. Its answers do not steer or generate with Qwen, and it performs no actions.

Verified on the deployment (root, real question about the Cookie Monster recipe): it returned HTTP 200 with canonical Haiku 5.5 and `tools_enabled: false`; CORS for 8791 OPTIONS returned 204 and a foreign origin 403; legacy agent overview/sessions returned 403; the wrapper's injected tail and assets match their hashes. All three GPU models stayed resident with no cold reload.

Generation controls: greedy decoding normally ignores the seed. Max new tokens (default 160, maximum 1024) applies separately to the baseline and steered runs, so each send can generate up to 2× the capped new tokens. The 4096 input context is a separate limit; the output control does not change it.

**Five custom Qwen3.8-27B personas: DONE (exploratory).** A delegated coding agent (OpenRouter, model `deepseek/deepseek-v4-pro-0813`) authored exactly five custom 27B presets, persisted them through the lab API and ran 15 fresh 64-token comparisons (three prompts each). No repetition loops were observed under that 64-token cap only; the cap is short, so this is not a robustness result. Several outputs do not clearly express their named target.

| Name (exact) | Custom ID | Basis |
|---|---|---|
| Impatient Evil Qwen (mild single-neuron) | `custom:759fad57730f499094ba5b8ce8f2ef58` | one neuron, lower dose |
| Prankster With One Big Toe Removed | `custom:bff6084544084524b2e94335ec1d96a7` | 27 neurons, one set to 0.0 |
| Poet (softer dose) | `custom:b68d438ca9114b29b8452cbfe4805d66` | one neuron, softer dose |
| Interior Stylist (mild dose) | `custom:9937540baebb4960ac38b79d6001edb0` | one neuron, milder dose |
| Rigid English Teacher (mild multi-neuron) | `custom:241b12c58826468c94df944c913b976e` | 3 neurons, doses ~20% lower |

Poet and Rigid English Teacher showed the strongest identity in the samples; the other three are exploratory and their targets are weak in places. **Find them in the UI:** filter source **My personas**, model **Qwen3.8-27B**, and **all** validation states. Their status is `pending`: the full standardized validation batch has not accepted them. No weights were trained, and the name does not discover neurons; each preset is a saved steering configuration derived from a published parent recipe.

Exact records are in [`data/custom/qwen3.8-27b-personas.json`](data/custom/qwen3.8-27b-personas.json). Evidence (`outputs/today-five-q27-personas.md` and `.json`) is a local, untracked file and is not linked. To import on another replica, see "Import the five custom 27B personas" in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Verified GPU deployment

All three official models downloaded and passed real baseline, zero-dose, single-neuron, multi-neuron and hook-cleanup smoke checks on a 96 GB RTX PRO 6000. Both reference and optional FLA kernel paths passed.

| Official model | Observed peak allocation, 32-token smoke |
|---|---:|
| `Qwen/Qwen3.5-4B` | 8.53 GiB |
| `Qwen/Qwen3-8B` | 15.28 GiB |
| `Qwen/Qwen3.8-27B` | 51.14 GiB |

Verified environment: NVIDIA driver **615.71.09**, driver CUDA capability **13.4**, PyTorch runtime CUDA **13.0**, **PyTorch 2.9.0+cu130**, **Transformers 5.22.2**, BF16 inference. These are measured smoke-run allocations, not guarantees for longer contexts or simultaneous models. Official weights loaded without remote model code or model-service credentials.

Experiment rental **60435160** ran at approximately **US$1.433/hour** and has been **destroyed**; the provider subsequently listed no account instances. Results are saved locally. For tomorrow's live demo, use [the one-command runbook](docs/RUNBOOK.md) to rent under US$2/hour, restore the environment, and destroy it on exit or at its two-hour deadline. The local watchdog requires an awake, connected Mac.

All 36 starter records, 258 supported Neuronary records, and ten Persona Library records completed generation: **2,432 successful comparison jobs**. Four unsupported Neuronary records failed explicitly. Final record counts are **301 completed, 3 reproduced under an exploratory Codex rubric, 4 failed, and 78,111 pending**. The large pending catalog is paused and consumes no GPU time. See [verification](docs/verification.md) and local `outputs/gpu-final/outputs/` for complete evidence. The entire catalog has not been validated.

## Catalog and source fidelity

| Imported source records | Count |
|---|---:|
| Neuronary public paginated API, full recipe details | 262 |
| Persona Library: personas | 34,220 |
| Persona Library: off-topic replies | 35,074 |
| Persona Library: changed-assistant replies | 8,827 |
| Supplied starter CSV | 36 |
| **Total** | **78,419** |

Starter rows can overlap public recipes and remain distinct source records. The original 36-row CSV is tracked at [`data/starter/qwen_personality_recipes.csv`](data/starter/qwen_personality_recipes.csv). Exact models, signed doses, combinations, method sites and raw metadata are retained. The Persona Library's exported response text is explicitly marked as source excerpts, not locally generated full responses. Source snapshots and SHA-256 provenance are retained locally; see [catalog details](docs/CATALOG.md).

The referenced ZIP, detailed specification and supplied synchronization script were never provided. The starter CSV is present and imported.

## Browse on the Mac

Catalog browsing requires Python 3.11+ and no GPU:

```bash
git clone git@github.com:TitoFM16/qwen-persona-lab.git
cd qwen-persona-lab
python3 scripts/sync_catalogs.py neuronary
python3 scripts/sync_catalogs.py personas
python3 scripts/sync_catalogs.py csv data/starter/qwen_personality_recipes.csv --source starter
python3 scripts/sync_catalogs.py queue
python3 -m lab.server --catalog-only --port 8787
```

Open **http://127.0.0.1:8787**. The existing workspace already has its database; do not resynchronize while workers are running. Metadata-only refreshes preserve results; changed interventions archive prior evidence and require fresh validation. Running jobs block intervention changes.

To restore saved public snapshots offline, replace the two download commands with:

```bash
python3 scripts/sync_catalogs.py neuronary-files data/sources/neuronary
python3 scripts/sync_catalogs.py personas-file data/sources/neuronpersonas/library.json
```

The real Qwen3-8B atlas is a separate public asset:

```bash
curl -fsS https://neuronpersonas.com/data/qwen/atlas.json \
  -o data/sources/neuronpersonas/atlas.json
```

## Rent, install and launch

Keep the Vast key in an existing private env file. The helper reads only `VAST_AI_API_KEY`; it never executes the file or prints credentials.

```bash
python3 scripts/vast_gpu.py offers --env-file /private/path/credentials.env
python3 scripts/vast_gpu.py rent OFFER_ID --key ~/.ssh/id_ed25519.pub \
  --env-file /private/path/credentials.env
python3 scripts/vast_gpu.py status --env-file /private/path/credentials.env
```

Renting rechecks availability and rejects a conservative compute-plus-storage estimate above US$2/hour. This is an hourly cap, not a total budget. The helper requests a CUDA 13 PyTorch image and 180 GB disk. Keep provider credentials on the Mac. Transfer the source and catalog through SSH:

```bash
tar --exclude='.git' --exclude='.venv' --exclude='.state' \
  --exclude='.env*' --exclude='outputs' \
  -czf /tmp/persona-lab.tar.gz .
scp -P SSH_PORT /tmp/persona-lab.tar.gz root@SSH_HOST:/tmp/
ssh -p SSH_PORT root@SSH_HOST \
  'mkdir -p /workspace/persona-lab && tar -xzf /tmp/persona-lab.tar.gz -C /workspace/persona-lab'
```

On the GPU:

```bash
cd /workspace/persona-lab
bash scripts/fetch_upstream.sh
bash scripts/bootstrap_gpu.sh
```

Bootstrap reuses a working CUDA-image PyTorch, verifies a real BF16 GPU operation, installs dependencies, records versions and disk/GPU data, runs tests and writes resumable smoke reports. Use `SKIP_SMOKE=1` for installation only. Optional optimized kernel/compiler commands are documented in [kernel setup](docs/kernel_setup.md); the reference path remains available.

Start the local-only servers on the GPU:

```bash
.venv/bin/python -m lab.server --port 8787
# In another terminal:
.venv/bin/python scripts/explorer_safe.py
```

On the Mac, use the existing verified SSH host key:

```bash
bash scripts/tunnel.sh SSH_HOST SSH_PORT .state/known_hosts
```

Open **http://127.0.0.1:8787** for the lab and **http://127.0.0.1:8791** for the original explorer. The helper reconnects, keeps both forwards on loopback and stops its own child on Ctrl+C. See [tunnel instructions](docs/tunnel.md) for the current direct-host key alias and occupied-port handling. No public HTTP port is needed.

The original Qwen3.5 3D source and model-specific caches are preserved under `vendor/neuron-explorer`. Always launch through the safe wrapper: agent execution routes are disabled with HTTP 403. The Qwen3-8B atlas displays **36 × 12,288** actual indexed neurons with signed published categories in 2D; it is not a 3D embedding. Qwen3.8 has no fabricated coordinates.

## Validation, reviews and resumption

```bash
bash scripts/run_validation.sh
```

The batch script processes starter and Neuronary recipes before a default **80-job Persona Library sample**, using four standardized prompts, two greedy repeats and a 96-token generation cap. `MAX_NEW_TOKENS` and `PERSONA_JOBS` adjust these explicit limits. Rerunning resumes pending work; completed jobs are not repeated. Lease renewal and recorded failures support interruption/OOM recovery. Failed jobs require deliberate retry through `Catalog.retry_failed()`.

Use one GPU inference process at a time. The worker, UI and original explorer have separate runtimes; stop worker inference before a live-chat demonstration. Ordinary comparisons generate fresh baselines; validation may reuse a matching real baseline with recorded cache provenance.

Neuronary uses model-bound published residual norms or exact fixed calibration. Persona Library uses unsteered prompt residual norms. **Twelve starter Qwen3-8B rows explicitly request fixed normalization despite originating in the prompt-normalized library.** Those declarations are preserved, so their results are not exact replications of the website's procedure. Unsupported models, experts, methods and normalizations fail explicitly.

Changed text alone never counts as reproduction. Codex rubric reviews and the optional unsteered Qwen judge are labeled exploratory agent assessments, not independent human or scientific validation. Greedy-repeat consistency is not sampling robustness. See [validation methods](docs/validation_methods.md) and the evidence-backed `docs/demo_findings_*.md` reviews.

Export a consistent snapshot while workers run:

```bash
.venv/bin/python scripts/export_results.py --db data/lab.sqlite3 \
  --output outputs/results.jsonl --report outputs/result-report.json \
  --backup outputs/lab-backup.sqlite3
```

Copy these artifacts back to the Mac before stopping or destroying the rental. The report distinguishes recipe/job statuses, exact source/model counts, review verdicts and narrow arithmetic/coherence proxies. Do not copy a live SQLite main file alone; its WAL may contain committed results.

```bash
python3 scripts/vast_gpu.py stop INSTANCE_ID --env-file /private/path/credentials.env
python3 scripts/vast_gpu.py start INSTANCE_ID --env-file /private/path/credentials.env
python3 scripts/vast_gpu.py destroy INSTANCE_ID --env-file /private/path/credentials.env
```

Stopped instances retain storage charges. Destroying removes the rented disk; preserve evidence first.

## Why one edited neuron can change the voice

Each MLP neuron writes a learned direction into the shared residual stream. Pushing that direction can bias identity, topic or style, repeatedly at every generation step; the resulting words then reinforce the new context. A normalized dose of one can move the residual by an entire typical residual norm, so a small number of edited indexes need not mean a small intervention. Weights remain unchanged: this is temporary inference steering, not training. Features are entangled and prompt-dependent; strong doses can produce neighboring identities, factual errors or repetition.

Sources: [Neuronary](https://github.com/lyramakesmusic/neuronary), [neuron explorer](https://github.com/lyramakesmusic/neuron-explorer), [Persona Library](https://neuronpersonas.com/).

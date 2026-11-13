# Qwen Persona Laboratory

A local neuron-steering lab with a searchable personality catalog, editable doses, baseline comparisons, chat, persistent experiment history and a resumable SQLite validation queue.

Private source repository: [TitoFM16/qwen-persona-lab](https://github.com/TitoFM16/qwen-persona-lab). Credentials, model weights, runtime databases and generated outputs are excluded from Git.

The UI now supports live baseline-versus-steered conversations and **My personas**: start from a recipe, edit/add/remove its model-specific neuron components and signed doses, then save a named personal variant without overwriting the source. Saving persists a pending recipe; it neither trains weights nor starts validation. The frontend editor was implemented with local **Claude Sonnet 5.5** and checked alongside the Python API. See [today’s live walkthrough](docs/LIVE_DEMO.md) for the November 13 presentation and 4:00 p.m. Bogotá cutoff. **Local software suite on November 13, 2026: 114 tests passed** (not GPU or scientific validation); earlier 69-test reports are historical and describe the previous implementation.

**Current live presentation (November 13):** the rental is ready and the UI and 3D explorer are reachable on the Mac through local-only forwards, and **Qwen3.5-4B** and **Qwen3-8B** are available on the current rental. **Qwen3.8-27B is still downloading** and enables once its full cache is ready; it is not ready and its verification is pending. The UI is chat-first: selecting a persona opens the conversation but does not generate; type in the composer and press Send or Enter (Shift+Enter for a newline). Each steered reply has a collapsed same-model, unsteered baseline disclosure. Optional comparison and advanced neuron editing/save controls are in collapsed sections. The lab keeps one active model: same-model persona switches reuse weights, switching models reloads, and the original 3D explorer has its own 4B runtime. The all-three-model smoke results below come from an earlier, destroyed rental and remain valid for that run, but do not mean all three models are live now. An independent GPU-side deletion timer is armed alongside the local deadline; keep the Mac awake and online for the tunnel and result backup.

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

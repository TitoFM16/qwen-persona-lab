# Optional optimized kernels on the rented Blackwell GPU

The baseline runtime and `scripts/bootstrap_gpu.sh` keep a reference implementation path available. A missing optional kernel package must not be interpreted as successful optimized execution; result provenance records installed optional packages and actual inference evidence.

The deployment run on the rented RTX PRO 6000 successfully completed the three official-model smoke checks with the reference path and with FLA enabled after installing the host C/C++ compilers. The runtime container did not initially have the compiler tools needed by the JIT kernel build.

Commands used for the optimized setup, from the GPU project directory:

```bash
apt-get update
apt-get install -y --no-install-recommends gcc g++
.venv/bin/python -m pip install flash-linear-attention==0.6.0 fla-core==0.6.0
.venv/bin/python -m pip freeze > outputs/dependencies-optimized.txt
.venv/bin/python scripts/smoke_models.py --output outputs/smoke-models-fla.json
```

Run smoke checks in a fresh process after installing FLA because the Transformers runtime may determine optional-kernel availability during import. Inspect the smoke JSON's model revisions, PyTorch/CUDA versions, optional-kernel provenance and real measured memory, rather than assuming an installed package was used. Keep the reference smoke report as a separate artifact; do not overwrite it with the optimized report.

These commands are Linux GPU-container commands, not Mac installation instructions. Compiler installation is only needed inside the rented image. FLA is optional; do not make the safe bootstrap depend on an optional JIT build succeeding. If the optimized path fails, retain the exact error and use the reference implementation while investigating compatibility.

## Export and retrieve validation evidence

Use the SQLite online backup API while the worker is running; copying the main `.sqlite3` file alone can omit committed WAL records:

```bash
.venv/bin/python scripts/export_results.py \
  --db data/lab.sqlite3 \
  --output outputs/results.jsonl \
  --report outputs/result-report.json \
  --backup outputs/lab-backup.sqlite3
```

`--limit 20` limits JSONL records for a quick preview; it never limits the report's snapshot counts or aggregate proxy statistics. `--source starter` or `--model Qwen/Qwen3-8B` selects a source/model subset. Job IDs determine record order, and SHA-256 hashes identify exported artifacts. Transfer the backup, JSONL and report to the Mac through the existing SSH connection.

The report distinguishes recipe statuses from job statuses, and groups each by source and model. It preserves source descriptions and each job's outputs, errors, prompts, doses, normalization and provenance. Numerical summaries are intentionally narrow: nonempty text, repeated consecutive word runs, changed text, recorded repeat consistency, actual GPU memory, and presence of `323` on the fixed `17 × 19` prompt. These are proxies, not personality, general factual accuracy or coherence ratings. A changed answer never marks a recipe reproduced.

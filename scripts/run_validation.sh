#!/usr/bin/env bash
# Supported source batches first; the large library stays resumable and bounded.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
token_limit="${MAX_NEW_TOKENS:-96}"
for source in starter neuronary; do
  for model in Qwen/Qwen3.5-4B Qwen/Qwen3-8B Qwen/Qwen3.8-27B; do
    label="${model#Qwen/}"
    .venv/bin/python -m lab.worker --model "$model" --source "$source" \
      --max-new-tokens "$token_limit" 2>&1 | tee -a "outputs/validation-${label}-${source}.log"
  done
done
# Remaining public entries include unsupported models/methods: record explicit
# failures instead of silently claiming they ran on a different Qwen model.
.venv/bin/python -m lab.worker --source neuronary --max-new-tokens "$token_limit" \
  2>&1 | tee -a outputs/validation-neuronary-unsupported.log
# Ten library recipes × four prompts × two repeats; increase deliberately.
.venv/bin/python -m lab.worker --model Qwen/Qwen3-8B --source neuronpersonas \
  --max-jobs "${PERSONA_JOBS:-80}" --max-new-tokens "$token_limit" \
  2>&1 | tee -a outputs/validation-persona-library.log
.venv/bin/python scripts/sync_catalogs.py counts | tee outputs/final-counts.json

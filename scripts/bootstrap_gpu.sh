#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
nvidia-smi > outputs/nvidia-smi.txt
df -h . > outputs/disk.txt
# CUDA images normally already include a suitable torch. Reuse it rather than
# paying download/storage costs for a second complete CUDA runtime.
if [ ! -x .venv/bin/python ]; then
  if ! python3 -m venv --system-site-packages .venv; then
    python3 -m pip install virtualenv
    python3 -m virtualenv --system-site-packages .venv
  fi
fi
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python -c 'import torch; from packaging.version import Version; assert Version(torch.__version__.split("+")[0]) >= Version("2.9"); assert torch.cuda.is_available(); x=torch.ones((16,16),device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert float(y[0,0])==16'; then
  # Official CUDA 13 wheels: https://pytorch.org/get-started/previous-versions/
  .venv/bin/pip install --upgrade 'torch>=2.9' --index-url https://download.pytorch.org/whl/cu130
fi
echo 'transformers==5.22.2' > outputs/bootstrap-constraints.txt
.venv/bin/pip install -c outputs/bootstrap-constraints.txt -e '.[gpu,explorer]'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_properties(0)); print("compiled_arches", torch.cuda.get_arch_list()); x=torch.ones((16,16),device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert float(y[0,0])==16' > outputs/torch.txt
.venv/bin/pip freeze > outputs/dependencies.txt
.venv/bin/python -m unittest discover -s tests -v 2>&1 | tee outputs/tests.txt
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  .venv/bin/python scripts/smoke_models.py --output outputs/smoke-models.json --resume "$@" 2>&1 | tee outputs/smoke.txt
fi
echo 'Run .venv/bin/python -m lab.worker --model Qwen/Qwen3.5-4B --poll in one terminal.'
echo 'Run .venv/bin/python -m lab.server --port 8787 in another terminal.'

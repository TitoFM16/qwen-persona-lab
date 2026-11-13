#!/usr/bin/env bash
# Start only the local web services; never starts workers or model judges.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
unset EXPLORER_NO_MODEL
mkdir -p outputs .state
python=.venv/bin/python
if [ ! -x "$python" ]; then
  echo "Missing .venv/bin/python. Run scripts/bootstrap_gpu.sh first." >&2
  exit 1
fi
export QWEN_REVISION_MANIFEST="${QWEN_REVISION_MANIFEST:-$PWD/data/model-cache-manifest.json}"
if [ -f "$QWEN_REVISION_MANIFEST" ]; then
  EXPLORER_MODEL=$("$python" - "$QWEN_REVISION_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
model='Qwen/Qwen3.5-4B'
revision=json.load(open(sys.argv[1]))['models'][model]['revision']
cached=try_to_load_from_cache(model,'config.json',revision=revision)
if not isinstance(cached,str) or not Path(cached).is_file():
    raise SystemExit('Pinned Qwen3.5-4B snapshot is missing; prefetch the manifest revision first.')
print(Path(cached).parent)
PY
  ) || exit 1
  export EXPLORER_MODEL
else
  export EXPLORER_MODEL=Qwen/Qwen3.5-4B
fi

# 0 = expected healthy service; 1 = no listener; 2 = occupied/incompatible.
probe() {
  "$python" - "$1" "$2" <<'PY'
import json, socket, sys, urllib.error, urllib.request
port, kind = int(sys.argv[1]), sys.argv[2]
base = f'http://127.0.0.1:{port}'
try:
    with urllib.request.urlopen(base+'/api/status', timeout=.75) as response:
        status = json.load(response)
    if kind == 'lab':
        if not isinstance(status,dict) or not isinstance(status.get('counts'),dict) or not isinstance(status.get('runtime'),dict):
            raise ValueError('Different service on lab port')
    else:
        if not isinstance(status,dict) or not isinstance(status.get('status'),str) or status['status']=='no model':
            raise ValueError('Different or cache-only service on explorer port')
        try:
            urllib.request.urlopen(base+'/api/agent/overview', timeout=.75).close()
        except urllib.error.HTTPError as error:
            if error.code != 403:
                raise
        else:
            raise ValueError('Explorer agent routes must be disabled')
    sys.exit(0)
except (OSError, ValueError, urllib.error.URLError):
    try:
        socket.create_connection(('127.0.0.1',port),timeout=.25).close()
    except OSError:
        sys.exit(1)
    sys.exit(2)
PY
}

probe 8787 lab
lab_status=$?
probe 8791 explorer
explorer_status=$?
if [ "$lab_status" -eq 2 ] || [ "$explorer_status" -eq 2 ]; then
  echo "Port 8787 or 8791 has an incompatible/unhealthy listener. No existing process was stopped." >&2
  echo "Inspect the occupied port and its logs, then stop it yourself if appropriate." >&2
  exit 1
fi

if [ "$lab_status" -ne 0 ]; then
  export QWEN_REQUIRE_CACHED_MODELS=1
  nohup "$python" -m lab.server --device cuda --port 8787 > outputs/lab-server.log 2>&1 < /dev/null &
  echo "$!" > .state/lab-server.pid
  echo "Started lab server, PID $(cat .state/lab-server.pid)."
else
  echo "Lab server is already healthy; leaving it unchanged."
fi
if [ "$explorer_status" -ne 0 ]; then
  PORT=8791 nohup "$python" scripts/explorer_safe.py > outputs/explorer-server.log 2>&1 < /dev/null &
  echo "$!" > .state/explorer-server.pid
  echo "Started safe live explorer, PID $(cat .state/explorer-server.pid)."
else
  echo "Safe live explorer is already healthy; leaving it unchanged."
fi

echo "Logs: outputs/lab-server.log and outputs/explorer-server.log"
deadline=$((SECONDS + 30))
while [ "$SECONDS" -lt "$deadline" ]; do
  if probe 8787 lab && probe 8791 explorer; then
    echo "Healthy: http://127.0.0.1:8787 and http://127.0.0.1:8791"
    echo "Use scripts/tunnel.sh from the Mac. Model loading may continue after HTTP startup."
    exit 0
  fi
  sleep 1
done
echo "Web-service health checks did not pass within 30 seconds; inspect the printed logs." >&2
echo "PID files identify only processes started by this script; no other process was stopped." >&2
exit 1

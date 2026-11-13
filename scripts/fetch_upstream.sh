#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
for repo in neuron-explorer neuronary; do
  if [ ! -d "vendor/$repo" ]; then
    git clone "https://github.com/lyramakesmusic/$repo.git" "vendor/$repo"
  fi
done

# Catalog sources and reproducibility

The original ZIP, synchronization script, and detailed prompt were not accessible. The user subsequently supplied `/Users/juanfigueroa/Downloads/qwen_personality_recipes.csv`: all 36 rows were imported as source `starter`, preserving the personality, structured components, method site, token scope, normalization and raw row. The CSV is snapshotted under `data/sources/starter/`; SHA-256 and import results appear in `data/starter-import-report.json`. Starter source records remain separate even when they overlap public recipes.

Fetched public sources on November 12, 2026:

- Neuronary: 262 public recipes through three offset-paginated API pages, then every full recipe endpoint including examples. Exact model, component order, layer, neuron, expert, signed dose, method, sampling, combinations and source object are preserved in raw JSON. Source: https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes and https://github.com/lyramakesmusic/neuronary/blob/main/supabase/functions/api/index.ts
- Neuron Persona Library: https://neuronpersonas.com/data/qwen/library.json contains 78,121 Qwen3-8B entries: 34,220 persona responses, 35,074 off-topic responses, and 8,827 altered-assistant responses. Every entry is imported, including potentially sensitive entries. Full JSON metadata is retained; response text is an excerpt in this published artifact, marked `reply_is_excerpt=true`. The original website's CSV export uses these same excerpts. This lab generated `data/qwen3-8b-neuron-personas.csv` from the unfiltered source with the website's CSV column mapping, plus explicit model column. This is not a direct downloaded CSV, because the site's CSV is constructed client-side from the JSON.

The authentic Qwen3-8B atlas is saved at `data/sources/neuronpersonas/atlas.json`. Its Base64 payload decodes to 442,368 bytes, one per actual neuron (`layer * 12,288 + neuron`). The low nibble is the positive-direction category; the high nibble is the negative-direction category. It supplies categories at indexed neuron locations, not learned 3D coordinates.

Source files, retrieval URLs, timestamps, sizes and SHA-256 hashes are in `data/sources/manifest.json`. The SQLite database is runtime state and should be backed up separately. Rebuild it offline from these source snapshots:

```sh
python3 scripts/sync_catalogs.py personas-file data/sources/neuronpersonas/library.json
python3 scripts/sync_catalogs.py neuronary-files data/sources/neuronary
python3 scripts/sync_catalogs.py queue
python3 scripts/sync_catalogs.py counts
```

Refresh public data on a networked host:

```sh
python3 scripts/sync_catalogs.py neuronary
python3 scripts/sync_catalogs.py personas
```

Import the missing starter CSV, or manually exported source CSV:

```sh
python3 scripts/sync_catalogs.py csv /path/to/starter.csv --source starter --model Qwen/Qwen3.5-4B
python3 scripts/sync_catalogs.py csv /path/to/qwen3-8b-neuron-personas.csv --source neuronpersonas-manual --model Qwen/Qwen3-8B
```

CSV errors are reported with original row data and recorded in the `imports` table; they are never silently skipped. CSV model defaults are explicit, and recipes lacking a signed dose are rejected. Manual/starter imports use `normalization=unspecified` unless the CSV explicitly declares one, so the runtime must resolve normalization before steering. Neuronary recipes use the fixed calibration normalization (`neuronary`); Persona Library recipes use per-prompt residual normalization (`persona_library`). Different normalization procedures are not interchangeable.

The queue contains 627,352 jobs: four standard prompts × two repeats × 78,419 source records (including the 36 starter rows). Jobs use an atomic SQLite claim and expiring lease, preserving attempts, error and result details. Expired running jobs may be reclaimed. OOM retries must be bounded by the worker. Successful generation means `completed`; it does not mean personality reproduction. `reproduced` requires explicit evaluated reproduction evidence on every required result. Imported source claims remain metadata, never local experimental results.

Eight catalog tests cover full-precision signed combinations, provenance, idempotent imports and queueing, model assignment, multiline CSV, malformed recipe rejection, paginated full-detail sync, expired lease recovery, same-component layer/neuron filtering, and absence of automatic reproduction claims.

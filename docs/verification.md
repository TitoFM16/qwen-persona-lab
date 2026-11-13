# Verification and limits

Final main-validation checkpoint, November 12–13, 2026. Counts below describe actual recorded inference, not software-test fixtures or imported source examples.

## Catalog and experiment counts

| Recipe status | Count |
|---|---:|
| Imported source records | 78,419 |
| Completed, without strict reproduction acceptance | 301 |
| Reproduced under the exploratory Codex rubric | 3 |
| Failed: unsupported model/method | 4 |
| Pending | 78,111 |
| Running | 0 |

There are **304 successfully generated recipe records**: 36 starter records, 258 supported Neuronary records and a ten-record alphabetical Persona Library sample from layer 0. Each has four standardized prompts and two deterministic repetitions: **2,432 successful generation jobs**. Another **32 jobs failed explicitly** on unsupported records; **624,888 jobs remain pending**. This sample is neither representative nor full validation of the 78,121-entry Persona Library. Starter/public records can overlap and remain separate sources.

The three strict accepted records are **Cookie Monster**, **Contempt**, and the **Evil alternate** recipe. Acceptance means the stated target clearly appeared with sufficient coherence on all four reviewed prompts and both runs under an explicitly labeled **exploratory Codex rubric**. It is not independent scientific validation or human review, and does not imply factual accuracy or helpfulness. Greedy repeat agreement is not sampling robustness. See the evidence-backed `demo_findings_*.md` reports.

## Real GPU verification

All three official models passed real baseline, zero-dose equivalence, single-neuron, multi-neuron and hook-cleanup smoke checks on an **NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB class** GPU. Both reference and optional FLA paths passed.

| Model | Measured peak allocation, 32-token smoke |
|---|---:|
| Qwen/Qwen3.5-4B | 8.53 GiB |
| Qwen/Qwen3-8B | 15.28 GiB |
| Qwen/Qwen3.8-27B | 51.14 GiB |

Environment: NVIDIA driver **615.71.09**, driver CUDA capability **13.4**, PyTorch runtime **CUDA 13.0**, **PyTorch 2.9.0+cu130**, **Transformers 5.22.2**, BF16. These measured allocations are not guarantees for longer contexts or simultaneous models.

Older inference records retain their original null revision fields. The post-run immutable cache inventory in [`data/model-cache-manifest.json`](../data/model-cache-manifest.json) records exactly one snapshot per model, config hashes and weight metadata. No historical revision hash was invented; future loads are pinned to resolved revisions.

## Software, explorer and security

**69 local software tests pass.** The last confirmed remote suite passed 60 tests. The later OOM regression and presentation-controller tests were verified locally; no additional GPU rental was made for those software tests. Tests cover numerical steering, native architecture/cache integration, source fidelity, SQLite recovery, review evidence, timestamp provenance, resource guards and safe failure handling. Python/JavaScript/shell checks passed.

The original Qwen3.5 explorer was launched and verified through HTTP and live SSE: its zero-dose three-row comparison produced identical responses; a nonzero **L22 N2742 +0.8** comparison produced the **Extrali** identity. Agent execution routes return **403**. Real Qwen3-8B atlas data is available as a model-specific 2D index/category map. No Qwen3.8 coordinates or 3D embedding were fabricated.

The lab UI and original explorer were verified through HTTP/SSE. Browser rendering and screenshot-based visual QA remain unverified because no browser-control surface was available. Servers bind to loopback, with SSH forwarding for remote access. Model outputs are text and receive no shell execution tools.

The initial 8B automated judge trial was rejected after it confused target fidelity with task correctness and misrated coherence; its raw evidence is retained. A later 27B v2 advisory trial covered eight cases but remains uncalibrated. The optional full judge was stopped for cost; completing that optional review is not required to display completed generation. Automated judge suggestions cannot mark recipes reproduced.

## Preservation and remaining limits

Source is in the private repository [`TitoFM16/qwen-persona-lab`](https://github.com/TitoFM16/qwen-persona-lab). Catalog/source data, SQLite snapshots and experiment exports are retained locally outside Git; credentials and model weights are excluded from the repository. Official weights can be downloaded again using the recorded revisions.

Twelve Qwen3-8B starter rows explicitly declare fixed normalization despite originating from a prompt-normalized library. Their CSV declarations were preserved, so these are not exact original-method replications. Unsupported models, normalizations, experts and conditional methods are rejected explicitly. The ZIP specification and supplied synchronization script were never provided; the 36-row starter CSV is present and imported.

Primary rental **60435160** was approximately **US$1.433/hour**, below the US$2/hour cap. It was **destroyed**, and a subsequent provider listing returned **[]**, verified 2026-11-13T04:06:27.566579+00:00. No GPU is rented. The local receipt is `outputs/gpu-final/outputs/rental-destruction-receipt.json`; final evidence and a verified SQLite snapshot are in that same directory. The Mac UI at port 8787 now runs in catalog-only mode.

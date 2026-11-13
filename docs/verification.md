# Verification and outstanding acceptance

Verified in the local Mac session on 2026-11-12 (America/Bogota):

- Dependency install succeeded through `uv sync --extra gpu`. Resolved environment: Python 3.14.6, PyTorch 2.15.0, Transformers 5.22.2. `uv.lock` records resolved packages. CUDA is unavailable on this Mac.
- All 24 tests passed, including numerical normalized single/multiple neuron direction checks using real PyTorch tensors, hook cleanup, scope, invalid input, SQLite lease recovery, deterministic queue idempotence, multiline imports and worker resumption. See `outputs/tests.txt` for the final suite count.
- Python byte-compilation, JavaScript syntax and shell syntax checks passed.
- Direct server-handler checks cover catalog search, queue, history, model matching, explicit normalization, traversal, same-origin policy and SQLite thread affinity. Network socket/browser rendering verification is unavailable.
- Imported 78,419 source records: 262 full Neuronary recipes, 78,121 Persona Library entries, and 36 subsequently supplied starter CSV rows. All 34,220 personas are present. Four prompts × two deterministic repeats produce 627,352 pending jobs.
- Completed recipes: 0. Reproduced: 0. Failed validation recipes: 0. Pending: 78,419. These counts describe real experimental state, not test fixture results.
- Three public Neuronary entries target models outside the three supported Qwen models. They remain searchable/pending. Unsupported methods/normalizations are rejected explicitly. Not all 262 recipes are therefore executable by the supported adapters.
- A real 96 GB RTX PRO 6000 Max-Q rental, instance 60432868, was created at about US$1.1393/hour including allocated disk, then destroyed when SSH proved inaccessible. The provider's destroy response returned success; the subsequent instance listing confirmed this rental absent. No model inference was performed. Actual final billing was not retrieved.
- Temporary provider credential/config/account-response files were removed. No secret was copied to the source tree or GPU.

Not yet verified or completed:

1. Supplied ZIP/specification/sync script remain unavailable; starter CSV is now imported.
2. Official model weight downloads and actual inference on the three models.
3. GPU CUDA/PyTorch/VRAM/disk verification or measured inference memory usage.
4. Any personality reproduction, factual accuracy or coherence rating.
5. A launched UI or rendered browser QA: loopback socket binding failed with `Operation not permitted`.
6. Full original 3D explorer launch against model configuration and cached assets.
7. Qwen3.8 model-specific 3D visualization; no coordinates fabricated.
8. Git initialization/commit/push: creating `.git` failed with `Operation not permitted`.

The session's approval policy explicitly rejected the outbound-SSH escalation because `sandbox_approval` and `rules` are disabled. Changing rental offers cannot resolve that host restriction. Run the documented commands in a normal Terminal or a session that permits SSH and local listeners. The implementation is not represented as having met all original GPU acceptance criteria.

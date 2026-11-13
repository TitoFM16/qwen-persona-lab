# Validation method and reproduction limits

The laboratory runs deterministic greedy, batch-one baseline-versus-steered comparisons on four standardized prompts, each repeated twice. Source `sampling` settings are retained in the result provenance, but standardized validation deliberately does not switch decoding settings from recipe to recipe. A baseline or source-example text mismatch is therefore not automatically a failed persona claim. A changed output is not automatically a reproduced persona either. Personality, coherence, accuracy and reproduction verdicts remain pending human review.

The supplied starter CSV contains 36 source records. Twelve Qwen3-8B persona-source rows name the fixed normalization `layer_residual_median_over_down_column_l2`, while the original Persona Library's method uses residual norms measured on each unsteered chat prompt. The CSV's explicit normalization has been preserved exactly; it is not silently replaced based on a source URL. Those starter comparisons may therefore differ from the published Persona Library examples. The separately imported full Persona Library records use the original prompt-based normalization. The UI permits an explicit alternative normalization for an exploratory comparison, which is recorded in experiment provenance.

Twenty-four starter rows target `mlp.down_proj.input`; twelve request addition along a down-projection column at `residual_stream_after_layer`. For the registered Qwen3/Qwen3.5 decoder architectures, the MLP's linear down projection followed by residual addition makes these two additive interventions algebraically equivalent. The laboratory applies its pre-down-projection hook and records both the declared and implemented sites. Residual-stream multiplication is rejected, because it is not equivalent to multiplying a neuron activation. Unsupported sites, expert/MoE components and conditional rules are not guessed.

The downloaded Neuronary catalog includes three recipes for models outside the supported three Qwen models, and one recipe declares the unsupported normalization `gne_relative` with scope `generation`. Such records remain searchable. Their execution is rejected explicitly rather than approximated. A worker filtered to supported Qwen model IDs does not claim the three unrelated-model records.

Use source-filtered workers to finish the small published/starter batches before consuming the large Persona Library queue:

```sh
python -m lab.worker --model Qwen/Qwen3.5-4B --source starter
python -m lab.worker --model Qwen/Qwen3.5-4B --source neuronary
python -m lab.worker --model Qwen/Qwen3-8B --source starter
python -m lab.worker --model Qwen/Qwen3-8B --source neuronary
python -m lab.worker --model Qwen/Qwen3.8-27B --source starter
python -m lab.worker --model Qwen/Qwen3.8-27B --source neuronary
```

Run these sequentially on a single GPU. Model runtimes in the UI server, worker and original explorer are independent; simultaneous inference can load multiple copies into VRAM. Prefer stopping the worker before live-chat demonstrations.

The worker publishes `.state/worker-status.json` atomically every 15 seconds and after lifecycle events. It includes process ID, model, job, state, GPU memory and timestamps. Background lease renewal prevents a long model download/inference from expiring its job claim. Failures retain their job and source; interrupted jobs become reclaimable after lease expiry. A missing runtime stops the worker early to preserve the rest of the queue.

CPU tests instantiate random tiny native Qwen3 and hybrid Qwen3.5 architectures to check residual measurement, cached autoregressive decoding, intervention hooks and cleanup against the installed Transformers API. These tests do not use official pretrained weights and are not GPU smoke results or experimental persona evidence. Real model smoke reports separately record official model revision, software versions and measured GPU memory.

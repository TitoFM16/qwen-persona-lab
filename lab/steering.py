"""Inference-only MLP interventions; model output is never executed.

Semantics: https://github.com/lyramakesmusic/neuronary/blob/main/scripts/run.py
Persona normalization: https://neuronpersonas.com/#method
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from collections import OrderedDict
from difflib import SequenceMatcher

MODELS = {
    'Qwen/Qwen3.5-4B': {'layers': 32, 'neurons': 9216},
    'Qwen/Qwen3-8B': {'layers': 36, 'neurons': 12288},
    'Qwen/Qwen3.8-27B': {'layers': 64, 'neurons': 17408},
}
# Upstream MIT calibration fixture, kept identical to make dose units comparable.
CALIBRATION = (
    'The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, '
    'and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. '
    'def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n'
    '    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. '
)
NORMALIZATIONS = {'neuronary', 'persona_library', 'prompt_residual', 'raw',
                  'layer_residual_median_over_down_column_l2'}


def optional_package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_model_revision(model_id, config):
    """Resolve an immutable cached config snapshot without a second network lookup."""
    reason = 'config_json_not_found_in_huggingface_cache'
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_id, 'config.json')
        if isinstance(cached, (str, os.PathLike)):
            parts = Path(cached).parts
            for index, part in enumerate(parts[:-1]):
                if part == 'snapshots' and re.fullmatch(r'[0-9a-fA-F]{40}', parts[index + 1]):
                    return parts[index + 1], 'huggingface_cached_config_snapshot'
            reason = 'cached_config_path_does_not_contain_an_immutable_snapshot_revision'
    except Exception as exc:
        reason = 'huggingface_cache_revision_lookup_failed:' + type(exc).__name__
    fallback = getattr(config, '_commit_hash', None)
    if isinstance(fallback, str) and re.fullmatch(r'[0-9a-fA-F]{40}', fallback):
        return fallback, 'model_config_commit_hash_fallback'
    return None, reason


def requested_model_revision(model_id):
    """An explicit manifest selects the revision before even fetching config."""
    manifest_path = os.environ.get('QWEN_REVISION_MANIFEST')
    if not manifest_path:
        return None, None
    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text())
        models = manifest.get('models') if isinstance(manifest, dict) else None
        entry = models.get(model_id) if isinstance(models, dict) else None
        revision = entry.get('revision') if isinstance(entry, dict) else None
    except (OSError, ValueError) as exc:
        raise ValueError('Cannot read explicitly requested Qwen revision manifest: ' + str(path)) from exc
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Explicit revision manifest requires an immutable 40-hex revision for ' + model_id)
    return revision.lower(), str(path)


def validate_components(model_id, components):
    if model_id not in MODELS:
        raise ValueError(f'Unsupported model: {model_id}')
    result = []
    for item in components:
        if 'expert' in item:
            raise ValueError('MoE expert steering is unsupported')
        layer, neuron = item['layer'], item['neuron']
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError('Layer must be a zero-based integer')
        if isinstance(neuron, bool) or not isinstance(neuron, int):
            raise ValueError('Neuron must be a zero-based integer')
        if not 0 <= layer < MODELS[model_id]['layers'] or not 0 <= neuron < MODELS[model_id]['neurons']:
            raise ValueError(f'Invalid neuron L{layer}N{neuron} for {model_id}')
        dose = float(item['dose'])
        if not math.isfinite(dose):
            raise ValueError('Dose must be finite')
        result.append({'layer': layer, 'neuron': neuron, 'dose': dose})
    return result


def decoder_layers(model):
    for path in ('model.language_model.layers', 'model.layers', 'language_model.layers'):
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    raise ValueError('Cannot locate decoder layers; refusing guessed hooks')


class Intervention:
    """Context manager installs pre-down-projection hooks and always removes them."""
    def __init__(self, layers, components, units, op='add', apply_to='all_tokens'):
        if op not in {'add', 'mul'} or apply_to not in {'all_tokens', 'prompt_tokens', 'generated_tokens'}:
            raise ValueError('Unsupported steering operation or token scope')
        self.phase = 'prefill'
        self.handles = []
        self.layers, self.components, self.units = layers, components, units
        self.op, self.apply_to = op, apply_to

    def __enter__(self):
        grouped = {}
        for c in self.components:
            grouped.setdefault(c['layer'], []).append(c)
        try:
            for layer, items in grouped.items():
                self.handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(self._hook(items)))
        except BaseException:
            self.remove()
            raise
        return self

    def _hook(self, components):
        def apply(module, inputs):
            if ((self.apply_to == 'prompt_tokens' and self.phase == 'decode') or
                    (self.apply_to == 'generated_tokens' and self.phase == 'prefill')):
                return None
            h = inputs[0].clone()
            # Sequential updates deliberately sum duplicate additive components.
            for c in components:
                n = c['neuron']
                if self.op == 'add':
                    h[..., n] += c['dose'] * self.units[(c['layer'], n)]
                else:
                    h[..., n] *= c['dose']
            return (h,) + tuple(inputs[1:])
        return apply

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *_):
        self.remove()


class ModelRuntime:
    def __init__(self, device='cuda', norms_dir='vendor/neuronary/norms'):
        self.device = device
        self.norms_dir = Path(norms_dir)
        self._lock = threading.RLock()
        self.model = self.tokenizer = self.torch = None
        self.model_id = None
        self.model_revision = None
        self.model_revision_reason = 'model_not_loaded'
        self.model_revision_manifest = None
        self.error = None
        self._medians = {}
        self._norm_sources = {}
        self._baseline_cache = OrderedDict()

    @property
    def status(self):
        result = {'device': self.device, 'model': self.model_id, 'loaded': self.model is not None,
                  'error': self.error, 'gpu': None}
        if self.torch is not None and self.torch.cuda.is_available():
            free, total = self.torch.cuda.mem_get_info()
            result['gpu'] = {'name': self.torch.cuda.get_device_name(), 'free_bytes': free,
                             'total_bytes': total, 'allocated_bytes': self.torch.cuda.memory_allocated(),
                             'peak_allocated_bytes': self.torch.cuda.max_memory_allocated()}
        return result

    def unload(self):
        with self._lock:
            self.model = self.tokenizer = None
            self.model_id = None
            self.model_revision = None
            self.model_revision_reason = 'model_not_loaded'
            self.model_revision_manifest = None
            self._medians.clear()
            self._norm_sources.clear()
            self._baseline_cache.clear()
            gc.collect()
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def load(self, model_id):
        if model_id not in MODELS:
            raise ValueError(f'Unsupported model: {model_id}')
        requested_revision, manifest_path = requested_model_revision(model_id)
        if (self.model is not None and self.model_id == model_id
                and (requested_revision is None or self.model_revision == requested_revision)):
            return
        self.unload()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; attach a GPU or explicitly choose --device cpu/mps')
        dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
        config_kwargs = {'revision': requested_revision} if requested_revision is not None else {}
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False, **config_kwargs)
        if requested_revision is not None:
            revision, revision_reason = requested_revision, 'explicit_model_revision_manifest'
        else:
            revision, revision_reason = resolve_model_revision(model_id, config)
        revision_kwargs = {'revision': revision} if revision is not None else {}
        loader = AutoModelForImageTextToText if getattr(config, 'vision_config', None) else AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **revision_kwargs)
        # Direct placement avoids retaining an entire 27B CPU copy while transferring
        # weights to the GPU. Accelerate is a declared runtime dependency.
        model = loader.from_pretrained(model_id, dtype=dtype, trust_remote_code=False,
                                       attn_implementation='eager', device_map={'': self.device}, **revision_kwargs).eval()
        layers = decoder_layers(model)
        if len(layers) != MODELS[model_id]['layers']:
            raise ValueError('Downloaded model layer count differs from registered architecture')
        for layer in layers:
            if layer.mlp.down_proj.weight.shape[1] != MODELS[model_id]['neurons']:
                raise ValueError('Downloaded MLP shape differs from registered architecture')
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id
        self.model_revision, self.model_revision_reason = revision, revision_reason
        self.model_revision_manifest = manifest_path
        self.error = None
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    def _encode(self, prompt, messages):
        messages = messages or [{'role': 'user', 'content': prompt}]
        if not messages or any(m.get('role') not in {'user', 'assistant', 'system'} or
                               not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Chat messages must contain text and supported roles')
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        return self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)

    def _measure(self, ids, wanted):
        seen, handles = {}, []
        layers = decoder_layers(self.model)
        try:
            for layer in wanted:
                def capture(module, args, out, layer=layer):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] < 2:
                        raise ValueError('Calibration requires at least two tokens')
                    seen[layer] = float(h[0, 1:].float().norm(dim=-1).median())
                handles.append(layers[layer].register_forward_hook(capture))
            with self.torch.inference_mode():
                self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        if set(seen) != set(wanted) or any(not math.isfinite(v) or v <= 0 for v in seen.values()):
            raise ValueError('Invalid residual calibration')
        return seen

    def _units(self, ids, components, normalization, op):
        wanted = {c['layer'] for c in components}
        if not wanted or normalization == 'raw' or op == 'mul':
            return {(c['layer'], c['neuron']): 1.0 for c in components}, {}, 'raw' if op == 'add' else 'multiplicative'
        if normalization in {'persona_library', 'prompt_residual'}:
            medians = self._measure(ids, wanted)
            source = 'unsteered_chat_prompt_residual_median_excluding_first_token'
        else:
            table_path = self.norms_dir / (self.model_id.replace('/', '__') + '.json')
            if table_path.exists():
                table = json.loads(table_path.read_text())
                if table.get('model', self.model_id) != self.model_id:
                    raise ValueError('Norm table model mismatch')
                table = table['medians']
                medians = {l: float(table[str(l)] if isinstance(table, dict) else table[l]) for l in wanted}
                if any(not math.isfinite(v) or v <= 0 for v in medians.values()):
                    raise ValueError('Invalid published norm table')
                source = 'published_table:' + str(table_path)
            else:
                missing = wanted - self._medians.keys()
                if missing:
                    calibration_ids = self.tokenizer(CALIBRATION, return_tensors='pt').input_ids.to(self.device)
                    self._medians.update(self._measure(calibration_ids, missing))
                medians = {l: self._medians[l] for l in wanted}
                source = 'measured_fixed_calibration_sha256:' + hashlib.sha256(CALIBRATION.encode()).hexdigest()
        layers, units = decoder_layers(self.model), {}
        for c in components:
            l, n = c['layer'], c['neuron']
            norm = float(layers[l].mlp.down_proj.weight[:, n].detach().float().norm())
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Zero or invalid down projection norm')
            units[(l, n)] = medians[l] / norm
        return units, medians, source

    def _generate(self, ids, max_new, intervention=None):
        torch = self.torch
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.add(self.tokenizer.eos_token_id)
        tokens, finish = [], 'length'
        if intervention:
            intervention.phase = 'prefill'
        out = token = None
        try:
            with torch.inference_mode():
                out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
                for step in range(max_new):
                    token = out.logits[:, -1].float().argmax(-1)
                    value = int(token[0])
                    if value in eos:
                        finish = 'eos'
                        break
                    tokens.append(value)
                    if step + 1 < max_new:
                        if intervention:
                            intervention.phase = 'decode'
                        out = self.model(input_ids=token.reshape(1, 1), past_key_values=out.past_key_values,
                                         use_cache=True, logits_to_keep=1)
        finally:
            # Release KV/logits so an OOM traceback cannot pin GPU cache tensors.
            out = token = None
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), finish, tokens

    def _baseline(self, ids, max_new_tokens, seed, cache_baseline):
        """Reuse only an actually generated, identical deterministic baseline."""
        key = None
        if cache_baseline:
            identity = [self.model_id, self.model_revision or getattr(self.model.config, '_commit_hash', None),
                        str(self.model.dtype), self.device, seed, max_new_tokens,
                        self.model.generation_config.eos_token_id, self.tokenizer.eos_token_id,
                        ids.detach().cpu().tolist(), 'greedy_batch_one_thinking_off_v1']
            key = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        hit = cache_baseline and key in self._baseline_cache
        if hit:
            output, created_at = self._baseline_cache[key]
            self._baseline_cache.move_to_end(key)
        else:
            output, created_at = self._generate(ids, max_new_tokens), time.time()
            if cache_baseline:
                # Strings and CPU token IDs only; never retain model tensors or KV caches.
                self._baseline_cache[key] = ((output[0], output[1], tuple(output[2])), created_at)
                while len(self._baseline_cache) > 32:
                    self._baseline_cache.popitem(last=False)
        metadata = {'enabled': bool(cache_baseline), 'hit': bool(hit), 'key_sha256': key,
                    'generated_at': created_at,
                    'policy': 'Actual unsteered output generated once for identical model revision, input IDs, '
                              'dtype, device, seed, EOS settings and greedy decoding limit; '
                              'cleared on unload/model switch. Steered output is always freshly generated.'}
        return (output[0], output[1], list(output[2])), metadata

    def compare(self, model_id, prompt, components, normalization='neuronary', method=None,
                max_new_tokens=160, seed=0, messages=None, baseline_messages=None, steered_messages=None,
                cache_baseline=False):
        components = validate_components(model_id, components)
        method = method or {}
        normalization = method.get('normalization', normalization)
        if normalization not in NORMALIZATIONS:
            raise ValueError('Unsupported normalization; refusing silent fallback')
        op, scope = method.get('op', 'add'), method.get('apply_to', 'all_tokens')
        site = method.get('site', 'mlp.down_proj.input')
        if op not in {'add', 'mul'} or scope not in {'all_tokens', 'prompt_tokens', 'generated_tokens'}:
            raise ValueError('Unsupported method')
        if site not in {'mlp.down_proj.input', 'residual_stream_after_layer'}:
            raise ValueError('Unsupported intervention site')
        if site == 'residual_stream_after_layer' and op != 'add':
            raise ValueError('Residual-stream multiplication is not equivalent to MLP neuron multiplication')
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 1024:
            raise ValueError('max_new_tokens must be between 1 and 1024')
        with self._lock:
            start = time.monotonic()
            comparison_started_at = time.time()
            try:
                self.load(model_id)
                torch = self.torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                ids = self._encode(prompt, steered_messages if steered_messages is not None else messages)
                baseline_ids = self._encode(prompt, baseline_messages) if baseline_messages is not None else ids
                if max(ids.shape[-1], baseline_ids.shape[-1]) > 4096:
                    raise ValueError('Chat context exceeds laboratory 4096-token limit')
                units, medians, norm_source = self._units(ids, components, normalization, op)
                (baseline, base_finish, base_tokens), baseline_cache = self._baseline(
                    baseline_ids, max_new_tokens, seed, cache_baseline)
                torch.manual_seed(seed)
                with Intervention(decoder_layers(self.model), components, units, op, scope) as intervention:
                    steered, steer_finish, steer_tokens = self._generate(ids, max_new_tokens, intervention)
                generation_completed_at = time.time()
                metrics = {
                    'changed': baseline != steered,
                    'text_similarity': SequenceMatcher(None, baseline, steered).ratio(),
                    'baseline_tokens': len(base_tokens), 'steered_tokens': len(steer_tokens),
                    'baseline_finish': base_finish, 'steered_finish': steer_finish,
                    'steered_unique_token_fraction': len(set(steer_tokens)) / max(1, len(steer_tokens)),
                    'elapsed_seconds': time.monotonic() - start,
                    'personality': None, 'coherence': None, 'accuracy': None,
                    'rating_status': 'requires_human_review',
                }
                provenance = {
                    'comparison_started_at_unix': comparison_started_at,
                    'generation_completed_at_unix': generation_completed_at,
                    'model': model_id, 'revision': self.model_revision or getattr(self.model.config, '_commit_hash', None),
                    'revision_manifest': self.model_revision_manifest,
                    'revision_resolution': (self.model_revision_reason if self.model_revision is not None or
                                            getattr(self.model.config, '_commit_hash', None) is None
                                            else 'model_config_commit_hash_fallback'),
                    'revision_pinned_for_tokenizer_and_weights': self.model_revision is not None,
                    'components': components, 'normalization': normalization, 'norm_source': norm_source,
                    'residual_medians': medians, 'activation_units': {f'L{l}N{n}': u for (l,n),u in units.items()},
                    'method': {'op': op, 'apply_to': scope, 'site': site}, 'seed': seed, 'decoding': 'greedy',
                    'implemented_site': 'mlp.down_proj.input',
                    'site_equivalence': ('Linear down projection followed by residual addition makes the additive '
                                         'down-input hook equivalent to adding its normalized output column after '
                                         'these registered Qwen decoder layers.' if site == 'residual_stream_after_layer' else None),
                    'batch_size': 1, 'thinking': False, 'max_new_tokens': max_new_tokens,
                    'baseline_prompt_tokens': baseline_ids.shape[-1], 'steered_prompt_tokens': ids.shape[-1],
                    'independent_chat_histories': baseline_messages is not None or steered_messages is not None,
                    'baseline_cache_hit': baseline_cache['hit'], 'baseline_cache': baseline_cache,
                    'dtype': str(self.model.dtype), 'device': self.device,
                    'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                    'optional_kernels': {package: optional_package_version(package) for package in
                                         ('fla-core', 'flash-linear-attention', 'causal-conv1d')},
                    'cuda': torch.version.cuda, 'telemetry': self.status,
                    'reproduction_limits': ['No exact-repeat early stop; EOS or token limit only.',
                                            'Greedy batch-one inference; hardware and batch layout may affect strong steering.'],
                }
                self.error = None
                return {'baseline': baseline, 'steered': steered, 'metrics': metrics, 'provenance': provenance}
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                    self.unload()
                raise


def _is_oom(exc):
    return type(exc).__name__ == 'OutOfMemoryError'


class ModelRuntimePool:
    """Keeps one persistent ModelRuntime per official model resident; never evicts implicitly.

    Generation/loading is serialized by one RLock; status uses a separate brief lock and never blocks on it.
    """

    def __init__(self, device='cuda', norms_dir='vendor/neuronary/norms', runtime_factory=None):
        self.device = device
        self.norms_dir = norms_dir
        self._factory = runtime_factory or (lambda d, n: ModelRuntime(d, n))
        self._gen_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._runtimes = {}
        self._info = {}
        self._warming = set()
        self._errors = {}
        self._last_model = None
        self._runtime_seq = 0

    def _check(self, model_id):
        if model_id not in MODELS:
            raise ValueError('Unsupported model: ' + str(model_id))

    def _runtime(self, model_id):
        with self._state_lock:
            rt = self._runtimes.get(model_id)
            if rt is None:
                rt = self._factory(self.device, self.norms_dir)
                self._runtime_seq += 1
                self._runtimes[model_id] = rt
                self._info[model_id] = {'runtime_id': 'rt-%d' % self._runtime_seq, 'load_count': 0,
                                        'weights_loaded_at': None, 'warm_complete_at': None,
                                        'warmup_seconds': None}
            return rt

    def _gpu(self):
        with self._state_lock:
            runtimes = list(self._runtimes.values())
        for rt in runtimes:
            if getattr(rt, 'torch', None) is not None:
                try:
                    gpu = rt.status.get('gpu')
                except Exception:
                    gpu = None
                if gpu:
                    return gpu
        return None

    def _record_failure(self, model_id, exc, rt):
        entry = {'type': type(exc).__name__, 'message': str(exc)[:500], 'at_unix': time.time(),
                 'oom': _is_oom(exc), 'memory': self._gpu()}
        try:
            rt.unload()  # only the failed candidate; other runtimes stay resident
        except Exception:
            pass
        with self._state_lock:
            self._errors[model_id] = entry
            info = self._info[model_id]
            info['weights_loaded_at'] = info['warm_complete_at'] = info['warmup_seconds'] = None

    def _ensure_loaded(self, model_id, rt):
        if rt.model is not None:
            return
        rt.load(model_id)
        with self._state_lock:
            info = self._info[model_id]
            info['load_count'] += 1
            info['weights_loaded_at'] = time.time()
            info['warm_complete_at'] = None
            self._errors.pop(model_id, None)

    def warm(self, model_ids, retry_failed=False, warm_inference=True):
        model_ids = list(model_ids)
        for model_id in model_ids:
            self._check(model_id)
        for model_id in model_ids:
            with self._gen_lock:
                rt = self._runtime(model_id)
                with self._state_lock:
                    failed = model_id in self._errors
                    done = self._info[model_id]['warm_complete_at'] is not None and rt.model is not None
                if done or (failed and not retry_failed):
                    continue
                with self._state_lock:
                    self._errors.pop(model_id, None)
                    self._warming.add(model_id)
                try:
                    self._ensure_loaded(model_id, rt)
                    if warm_inference:
                        started = time.monotonic()
                        ids = rt._encode('Hello', None)
                        rt._generate(ids, 2)
                        elapsed = time.monotonic() - started
                        with self._state_lock:
                            self._info[model_id]['warmup_seconds'] = elapsed
                            self._info[model_id]['warm_complete_at'] = time.time()
                except Exception as exc:
                    self._record_failure(model_id, exc, rt)
                finally:
                    with self._state_lock:
                        self._warming.discard(model_id)
        return self.status

    def compare(self, model_id, *args, **kwargs):
        self._check(model_id)
        with self._gen_lock:
            rt = self._runtime(model_id)
            with self._state_lock:
                failed = self._errors.get(model_id)
                self._last_model = model_id
            if failed:
                raise RuntimeError('%s is in failed state (%s); explicit warm(retry_failed=True) required'
                                   % (model_id, failed['type']))
            try:
                with self._state_lock:
                    self._warming.add(model_id)
                self._ensure_loaded(model_id, rt)
                with self._state_lock:
                    self._warming.discard(model_id)
                result = rt.compare(model_id, *args, **kwargs)
            except Exception as exc:
                if _is_oom(exc) or rt.model is None:
                    self._record_failure(model_id, exc, rt)
                raise
            finally:
                with self._state_lock:
                    self._warming.discard(model_id)
            with self._state_lock:
                info = dict(self._info[model_id])
                resident = sorted(m for m, r in self._runtimes.items() if r.model is not None)
            if isinstance(result, dict) and isinstance(result.get('provenance'), dict):
                result['provenance']['pool'] = {
                    'residency_enabled': True, 'resident_models': resident,
                    'runtime_id': info['runtime_id'], 'load_count': info['load_count'],
                    'weights_loaded_at': info['weights_loaded_at'], 'warm_complete_at': info['warm_complete_at']}
            return result

    def unload(self, model_id=None):
        if model_id is not None:
            self._check(model_id)
        with self._gen_lock:
            with self._state_lock:
                targets = [model_id] if model_id else list(self._runtimes)
            for mid in targets:
                with self._state_lock:
                    rt = self._runtimes.get(mid)
                if rt is None:
                    continue
                rt.unload()
                with self._state_lock:
                    info = self._info[mid]
                    info['weights_loaded_at'] = info['warm_complete_at'] = info['warmup_seconds'] = None
                    self._errors.pop(mid, None)
        return self.status

    @property
    def status(self):
        gpu = self._gpu()
        with self._state_lock:
            resident = [m for m, r in self._runtimes.items() if r.model is not None]
            ready = [m for m in resident if self._info[m]['warm_complete_at'] is not None]
            models = {}
            for m in MODELS:
                rt = self._runtimes.get(m)
                info = self._info.get(m, {})
                models[m] = {'loaded': rt is not None and rt.model is not None,
                             'revision': getattr(rt, 'model_revision', None) if rt else None,
                             'error': self._errors.get(m),
                             'weights_loaded_at': info.get('weights_loaded_at'),
                             'warm_complete_at': info.get('warm_complete_at'),
                             'warmup_seconds': info.get('warmup_seconds'),
                             'load_count': info.get('load_count', 0), 'runtime_id': info.get('runtime_id'),
                             'warming': m in self._warming}
            errors = dict(self._errors)
            last_error = None
            if errors:
                k = max(errors, key=lambda e: errors[e]['at_unix'])
                last_error = errors[k]['type'] + ': ' + errors[k]['message']
            return {'device': self.device, 'model': self._last_model, 'loaded': bool(resident),
                    'error': last_error, 'gpu': gpu, 'residency_enabled': True,
                    'resident_models': resident, 'ready_models': ready,
                    'warming_models': sorted(self._warming), 'model_errors': errors,
                    'models': models, 'generation_serialized': True}

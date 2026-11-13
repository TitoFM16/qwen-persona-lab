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
import threading
import time
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
        self.error = None
        self._medians = {}
        self._norm_sources = {}

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
            self._medians.clear()
            self._norm_sources.clear()
            gc.collect()
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def load(self, model_id):
        if model_id not in MODELS:
            raise ValueError(f'Unsupported model: {model_id}')
        if self.model is not None and self.model_id == model_id:
            return
        self.unload()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; attach a GPU or explicitly choose --device cpu/mps')
        dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        loader = AutoModelForImageTextToText if getattr(config, 'vision_config', None) else AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        # Direct placement avoids retaining an entire 27B CPU copy while transferring
        # weights to the GPU. Accelerate is a declared runtime dependency.
        model = loader.from_pretrained(model_id, dtype=dtype, trust_remote_code=False,
                                       attn_implementation='eager', device_map={'': self.device}).eval()
        layers = decoder_layers(model)
        if len(layers) != MODELS[model_id]['layers']:
            raise ValueError('Downloaded model layer count differs from registered architecture')
        for layer in layers:
            if layer.mlp.down_proj.weight.shape[1] != MODELS[model_id]['neurons']:
                raise ValueError('Downloaded MLP shape differs from registered architecture')
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id
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
            norm = float(layers[l].mlp.down_proj.weight[:, n].float().norm())
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
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), finish, tokens

    def compare(self, model_id, prompt, components, normalization='neuronary', method=None,
                max_new_tokens=160, seed=0, messages=None, baseline_messages=None, steered_messages=None):
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
                baseline, base_finish, base_tokens = self._generate(baseline_ids, max_new_tokens)
                torch.manual_seed(seed)
                with Intervention(decoder_layers(self.model), components, units, op, scope) as intervention:
                    steered, steer_finish, steer_tokens = self._generate(ids, max_new_tokens, intervention)
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
                    'model': model_id, 'revision': getattr(self.model.config, '_commit_hash', None),
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
                    'dtype': str(self.model.dtype), 'device': self.device,
                    'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
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

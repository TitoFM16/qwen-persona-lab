import math
import types
import unittest

from lab.steering import (Intervention, ModelRuntime, decoder_layers, validate_components,
                          resolve_model_revision, requested_model_revision)


class RevisionTests(unittest.TestCase):
    def test_cached_snapshot_and_unresolved_paths(self):
        from unittest.mock import patch
        revision = 'a' * 40
        config = types.SimpleNamespace(_commit_hash=None)
        with patch('huggingface_hub.try_to_load_from_cache', return_value=f'/cache/models--Qwen--Qwen3-8B/snapshots/{revision}/config.json'):
            self.assertEqual(resolve_model_revision('Qwen/Qwen3-8B',config),
                             (revision,'huggingface_cached_config_snapshot'))
        for cached in [None, object(), '/cache/snapshots/main/config.json']:
            with patch('huggingface_hub.try_to_load_from_cache',return_value=cached):
                resolved,reason=resolve_model_revision('Qwen/Qwen3-8B',config)
                self.assertIsNone(resolved);self.assertTrue(reason)
        config._commit_hash=revision
        with patch('huggingface_hub.try_to_load_from_cache',return_value=None):
            self.assertEqual(resolve_model_revision('Qwen/Qwen3-8B',config),
                             (revision,'model_config_commit_hash_fallback'))

    def test_load_pins_tokenizer_and_model_to_resolved_snapshot(self):
        from unittest.mock import Mock, patch
        import sys
        revision='b'*40
        for vision in [False,True]:
            config=types.SimpleNamespace(_commit_hash=None,vision_config={} if not vision else {'test':True})
            layer=types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(4096,12288)))))
            model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]*36),config=config)
            model.eval=lambda:model
            tokenizer=object()
            config_loader=Mock();config_loader.from_pretrained.return_value=config
            tokenizer_loader=Mock();tokenizer_loader.from_pretrained.return_value=tokenizer
            causal=Mock();causal.from_pretrained.return_value=model
            multimodal=Mock();multimodal.from_pretrained.return_value=model
            transformers=types.SimpleNamespace(AutoConfig=config_loader,AutoTokenizer=tokenizer_loader,
                AutoModelForCausalLM=causal,AutoModelForImageTextToText=multimodal)
            fake_torch=types.SimpleNamespace(float32='float32',bfloat16='bfloat16',cuda=types.SimpleNamespace(is_available=lambda:False))
            with patch.dict(sys.modules,{'torch':fake_torch,'transformers':transformers}), \
                 patch.dict('os.environ', {'QWEN_REVISION_MANIFEST': ''}), \
                 patch('huggingface_hub.try_to_load_from_cache',return_value=f'/cache/snapshots/{revision}/config.json'):
                runtime=ModelRuntime(device='cpu');runtime.load('Qwen/Qwen3-8B')
                self.assertEqual(runtime.model_revision,revision)
                self.assertEqual(tokenizer_loader.from_pretrained.call_args.kwargs['revision'],revision)
                loader=multimodal if vision else causal
                self.assertEqual(loader.from_pretrained.call_args.kwargs['revision'],revision)
                runtime.unload();self.assertIsNone(runtime.model_revision)
                self.assertEqual(runtime.model_revision_reason,'model_not_loaded')

    def test_manifest_pins_first_config_tokenizer_and_weights(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        revision = 'c' * 40
        config = types.SimpleNamespace(vision_config=None, _commit_hash=None)
        projection = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(4096, 12288)))
        decoder = types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=projection))
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[decoder] * 36), config=config)
        model.eval = lambda: model
        config_loader, tokenizer_loader, causal, multimodal = Mock(), Mock(), Mock(), Mock()
        config_loader.from_pretrained.return_value = config
        causal.from_pretrained.return_value = model
        fake_transformers = types.SimpleNamespace(AutoConfig=config_loader, AutoTokenizer=tokenizer_loader,
            AutoModelForCausalLM=causal, AutoModelForImageTextToText=multimodal)
        fake_torch = types.SimpleNamespace(float32='float32', bfloat16='bfloat16',
            cuda=types.SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'models': {'Qwen/Qwen3-8B': {'revision': revision}}}))
            with patch.dict('os.environ', {'QWEN_REVISION_MANIFEST': str(manifest)}), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'transformers': fake_transformers}):
                runtime = ModelRuntime(device='cpu')
                runtime.load('Qwen/Qwen3-8B')
            for loader in (config_loader, tokenizer_loader, causal):
                self.assertEqual(loader.from_pretrained.call_args.kwargs['revision'], revision)
                self.assertFalse(loader.from_pretrained.call_args.kwargs['trust_remote_code'])
            self.assertEqual(runtime.model_revision, revision)
            self.assertEqual(runtime.model_revision_reason, 'explicit_model_revision_manifest')
            self.assertEqual(runtime.model_revision_manifest, str(manifest.resolve()))

    def test_requested_manifest_invalid_or_missing_model_fails_before_loading(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with patch.dict('os.environ', {'QWEN_REVISION_MANIFEST': ''}):
            self.assertEqual(requested_model_revision('Qwen/Qwen3-8B'), (None, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for content in [{'models': {}}, {'models': {'Qwen/Qwen3-8B': {'revision': 'main'}}},
                            {'models': {'Qwen/Qwen3-8B': {'revision': 'a' * 39}}}, []]:
                path.write_text(json.dumps(content))
                with self.subTest(content=content), patch.dict('os.environ', {'QWEN_REVISION_MANIFEST': str(path)}):
                    runtime = ModelRuntime(device='cpu')
                    with self.assertRaises(ValueError): runtime.load('Qwen/Qwen3-8B')
                    self.assertIsNone(runtime.model)
            with patch.dict('os.environ', {'QWEN_REVISION_MANIFEST': str(Path(directory) / 'absent')}):
                with self.assertRaisesRegex(ValueError, 'Cannot read'): requested_model_revision('Qwen/Qwen3-8B')


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class Module:
    def __init__(self, fail=False):
        self.fail = fail
        self.handles = []

    def register_forward_pre_hook(self, callback):
        if self.fail:
            raise RuntimeError('registration failed')
        handle = Handle()
        self.handles.append(handle)
        return handle


def layer(module):
    return types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=module))


class ValidationTests(unittest.TestCase):
    def test_coordinates_are_model_specific(self):
        component = {'layer': 50, 'neuron': 16000, 'dose': -0.75}
        self.assertEqual(validate_components('Qwen/Qwen3.8-27B', [component]), [component])
        for model in ['Qwen/Qwen3-8B', 'Qwen/Qwen3.5-4B']:
            with self.assertRaises(ValueError):
                validate_components(model, [component])

    def test_invalid_doses_and_indices(self):
        for dose in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                validate_components('Qwen/Qwen3-8B', [{'layer': 0, 'neuron': 1, 'dose': dose}])
        for index in [-1, 36, 2.5, True]:
            with self.assertRaises(ValueError):
                validate_components('Qwen/Qwen3-8B', [{'layer': index, 'neuron': 1, 'dose': 1}])

    def test_signed_duplicate_doses_preserved(self):
        cs = [{'layer': 8, 'neuron': 5315, 'dose': -1}, {'layer': 8, 'neuron': 5315, 'dose': 0.25}]
        self.assertEqual([x['dose'] for x in validate_components('Qwen/Qwen3-8B', cs)], [-1, .25])

    def test_real_decoder_layouts(self):
        expected = [object()]
        for model in [types.SimpleNamespace(model=types.SimpleNamespace(layers=expected)),
                      types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=expected)))]:
            self.assertIs(decoder_layers(model), expected)
        with self.assertRaises(ValueError):
            decoder_layers(object())

    def test_hook_cleanup_on_generation_error(self):
        module = Module()
        with self.assertRaises(RuntimeError):
            with Intervention([layer(module)], [{'layer': 0, 'neuron': 1, 'dose': 1}], {(0, 1): 2}):
                raise RuntimeError('generation failed')
        self.assertTrue(module.handles[0].removed)

    def test_partial_hook_registration_cleanup(self):
        first, second = Module(), Module(fail=True)
        with self.assertRaises(RuntimeError):
            with Intervention([layer(first), layer(second)],
                              [{'layer': 0, 'neuron': 1, 'dose': 1}, {'layer': 1, 'neuron': 2, 'dose': 2}],
                              {(0, 1): 1, (1, 2): 1}):
                pass
        self.assertTrue(first.handles[0].removed)

    def test_unsupported_norm_fails_before_loading(self):
        runtime = ModelRuntime()
        with self.assertRaises(ValueError):
            runtime.compare('Qwen/Qwen3-8B', 'Who are you?', [], normalization='guessed')
        self.assertFalse(runtime.status['loaded'])

    def test_unsupported_site_and_residual_multiplication_fail_before_loading(self):
        runtime = ModelRuntime()
        for method in [{'site': 'attention.output'},
                       {'site': 'residual_stream_after_layer', 'op': 'mul'}]:
            with self.assertRaises(ValueError):
                runtime.compare('Qwen/Qwen3-8B', 'Who are you?', [], method=method)
        self.assertFalse(runtime.status['loaded'])


class WorkerTests(unittest.TestCase):
    def test_source_filter_and_persisted_worker_telemetry(self):
        import tempfile
        import json
        from unittest.mock import patch
        from pathlib import Path
        from lab.catalog import Catalog
        from lab.worker import run
        class Runtime:
            def __init__(self, **kwargs): pass
            @property
            def status(self): return {'gpu': {'name': 'test-only GPU'}, 'loaded': False}
            def compare(self, *args, **kwargs):
                return {'baseline': 'base', 'steered': 'out', 'metrics': {}, 'provenance': {}}
            def unload(self): pass
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / 'test.sqlite3')
            status = Path(temp) / 'worker.json'
            catalog = Catalog(db)
            recipe = {'model': 'Qwen/Qwen3-8B', 'name': 'test', 'neurons': [{'layer': 1, 'neuron': 2, 'dose': 1}]}
            persona = catalog.upsert(recipe, 'neuronpersonas')
            starter = catalog.upsert(recipe, 'starter')
            catalog.enqueue([persona, starter], prompts=['Who are you?'], repeats=1)
            with patch('lab.worker.ModelRuntime', Runtime):
                result = run(db, source='starter', status_file=str(status))
            self.assertEqual(result['completed_jobs'], 1)
            self.assertEqual(catalog.counts()['jobs']['pending'], 1)
            self.assertEqual(catalog.get_recipe(persona)['status'], 'pending')
            self.assertEqual(catalog.get_recipe(starter)['status'], 'completed')
            snapshot = json.loads(status.read_text())
            self.assertEqual(snapshot['state'], 'stopped')
            self.assertEqual(snapshot['completed_jobs'], 1)
            self.assertEqual(snapshot['gpu']['name'], 'test-only GPU')
            catalog.close()
    def test_resume_only_pending_and_record_consistency(self):
        import tempfile
        from unittest.mock import patch
        from pathlib import Path
        from lab.catalog import Catalog
        from lab.worker import run
        class Runtime:
            def __init__(self, **kwargs): pass
            @property
            def status(self): return {'gpu': None}
            def compare(self, *args, **kwargs):
                return {'baseline': 'baseline', 'steered': 'response', 'metrics': {}, 'provenance': {}}
            def unload(self): pass
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / 'test.sqlite3')
            catalog = Catalog(db)
            rid = catalog.upsert({'model': 'Qwen/Qwen3-8B', 'name': 'test',
                                  'neurons': [{'layer': 1, 'neuron': 2, 'dose': 1}]}, 'neuronary-test')
            catalog.enqueue([rid], prompts=['Who are you?'], repeats=2)
            with patch('lab.worker.ModelRuntime', Runtime):
                first = run(db, max_jobs=1)
                second = run(db)
            self.assertEqual(first['completed_jobs'], 1)
            self.assertEqual(second['completed_jobs'], 1)
            jobs = catalog.list_jobs()
            self.assertEqual(sum(j['attempts'] for j in jobs), 2)
            result = next(j['result'] for j in jobs if j['result']['metrics']['consistency_prior_runs'])
            self.assertEqual(result['metrics']['consistency_exact_match_fraction'], 1)
            self.assertIsNone(result['review']['reproduced'])
            self.assertEqual(catalog.counts()['reproduced'], 0)
            catalog.close()

    def test_missing_runtime_leaves_remaining_queue_pending(self):
        import tempfile
        from unittest.mock import patch
        from pathlib import Path
        from lab.catalog import Catalog
        from lab.worker import run
        class Runtime:
            def __init__(self, **kwargs): pass
            def compare(self, *args, **kwargs): raise ModuleNotFoundError('torch')
            def unload(self): pass
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / 'test.sqlite3')
            catalog = Catalog(db)
            rid = catalog.upsert({'model': 'Qwen/Qwen3-8B', 'name': 'test',
                                  'neurons': [{'layer': 1, 'neuron': 2, 'dose': 1}]}, 'neuronary-test')
            catalog.enqueue([rid], prompts=['Who are you?'], repeats=2)
            with patch('lab.worker.ModelRuntime', Runtime):
                result = run(db)
            self.assertEqual(result['failed_jobs'], 1)
            self.assertEqual(catalog.counts()['jobs']['pending'], 1)
            catalog.close()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required for numeric hook integration tests; run on GPU installation')
class NumericTests(unittest.TestCase):
    def test_synthetic_cuda_oom_removes_real_hook_and_unloads_runtime(self):
        # Raise the real exception type on CPU. This checks the software recovery
        # path without allocating CUDA memory or provoking a hardware OOM.
        from unittest.mock import patch
        projection=torch.nn.Linear(2,2,bias=False)
        runtime=ModelRuntime(device='cpu')
        runtime.torch=torch
        runtime.model_id='Qwen/Qwen3-8B'
        runtime.model_revision='a'*40
        runtime.model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer(projection)]),
            config=types.SimpleNamespace(_commit_hash=None),dtype=torch.float32,
            generation_config=types.SimpleNamespace(eos_token_id=63))
        runtime.tokenizer=types.SimpleNamespace(eos_token_id=63)
        runtime._medians[0]=1.0;runtime._norm_sources[0]='fixture'
        runtime._baseline_cache['previous_fixture']=(('older','length',(1,)),0)
        phases=[]
        def generate(ids,max_new,intervention=None):
            phases.append('baseline' if intervention is None else 'steered')
            if intervention is None:
                self.assertFalse(projection._forward_pre_hooks)
                return ('successful baseline','length',[1,2])
            self.assertTrue(projection._forward_pre_hooks)
            projection(torch.zeros(1,2,2))  # The installed hook really executes.
            self.assertEqual(len(runtime._baseline_cache),2)
            raise torch.cuda.OutOfMemoryError('synthetic CPU-only recovery test')
        with patch.object(runtime,'load'), \
             patch.object(runtime,'_encode',return_value=torch.tensor([[1,2,3]])), \
             patch.object(runtime,'_generate',side_effect=generate), \
             patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(torch.cuda.OutOfMemoryError,'synthetic CPU-only'):
                runtime.compare('Qwen/Qwen3-8B','test',[{'layer':0,'neuron':0,'dose':.5}],
                                normalization='raw',max_new_tokens=3,cache_baseline=True)
        self.assertEqual(phases,['baseline','steered'])
        self.assertFalse(projection._forward_pre_hooks)
        self.assertIsNone(runtime.model);self.assertIsNone(runtime.tokenizer)
        self.assertIsNone(runtime.model_id);self.assertIsNone(runtime.model_revision)
        self.assertFalse(runtime._baseline_cache);self.assertFalse(runtime._medians)
        self.assertFalse(runtime._norm_sources)
        self.assertIn('OutOfMemoryError',runtime.error)

    def test_worker_synthetic_oom_keeps_failed_job_and_preserves_pending_bound(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from lab.catalog import Catalog
        from lab.worker import run
        instances=[]
        class Runtime:
            def __init__(self,**kwargs):
                self.calls=0;self.unloaded=False;instances.append(self)
            @property
            def status(self):return {'gpu':None}
            def compare(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    raise torch.cuda.OutOfMemoryError('synthetic worker recovery test')
                return {'baseline':'base','steered':'response','metrics':{},'provenance':{}}
            def unload(self):self.unloaded=True
        with tempfile.TemporaryDirectory() as temp:
            db=str(Path(temp)/'test.sqlite3');catalog=Catalog(db)
            try:
                rid=catalog.upsert({'id':'oom','name':'OOM test','model':'Qwen/Qwen3-8B',
                    'neurons':[{'layer':0,'neuron':0,'dose':.5}]},'fixture')
                catalog.enqueue([rid],['Who are you?'],repeats=3)
                with patch('lab.worker.ModelRuntime',Runtime):
                    report=run(db,max_jobs=2)
                self.assertEqual(report,{'completed_jobs':1,'failed_jobs':1})
                self.assertTrue(instances[0].unloaded)
                rows=[dict(row) for row in catalog.db.execute('SELECT * FROM jobs ORDER BY id')]
                self.assertEqual([row['status'] for row in rows],['failed','completed','pending'])
                self.assertIn('OutOfMemoryError',rows[0]['error'])
                self.assertIn('synthetic worker',rows[0]['error'])
                self.assertIsNone(rows[0]['result_json'])
                self.assertEqual(rows[2]['attempts'],0)
                self.assertEqual(catalog.retry_failed(),1)
                reclaimed=catalog.claim_job()
                self.assertEqual(reclaimed['id'],rows[0]['id'])
                self.assertEqual(reclaimed['attempts'],2)
                self.assertEqual(catalog.counts()['jobs']['pending'],1)
                self.assertEqual(catalog.db.execute('SELECT status FROM jobs WHERE id=?',
                    (rows[1]['id'],)).fetchone()[0],'completed')
            finally:catalog.close()

    def test_baseline_cache_uses_resolved_revision_when_config_hash_missing(self):
        from unittest.mock import Mock
        runtime=ModelRuntime(device='cpu')
        runtime.model_id='Qwen/Qwen3-8B';runtime.model_revision='a'*40
        runtime.model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=None),
            dtype=torch.float32,generation_config=types.SimpleNamespace(eos_token_id=63))
        runtime.tokenizer=types.SimpleNamespace(eos_token_id=63)
        runtime._generate=Mock(return_value=('baseline','length',[1]))
        ids=torch.tensor([[1,2,3]])
        _,first=runtime._baseline(ids,3,0,True)
        self.assertTrue(runtime._baseline(ids,3,0,True)[1]['hit'])
        runtime.model_revision='b'*40
        _,second=runtime._baseline(ids,3,0,True)
        self.assertFalse(second['hit'])
        self.assertNotEqual(first['key_sha256'],second['key_sha256'])

    def test_baseline_cache_identity_capacity_and_invalidation(self):
        from unittest.mock import Mock
        runtime = ModelRuntime(device='cpu')
        runtime.model_id = 'Qwen/Qwen3-8B'
        runtime.model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='revision1'),
            dtype=torch.float32, generation_config=types.SimpleNamespace(eos_token_id=63))
        runtime.tokenizer = types.SimpleNamespace(eos_token_id=63)
        runtime._generate = Mock(return_value=('baseline', 'length', [1, 2]))
        ids = torch.tensor([[1, 2, 3]])
        first, meta1 = runtime._baseline(ids, 3, 0, True)
        second, meta2 = runtime._baseline(ids, 3, 0, True)
        self.assertFalse(meta1['hit']); self.assertTrue(meta2['hit'])
        self.assertEqual(meta1['generated_at'], meta2['generated_at'])
        self.assertEqual(runtime._generate.call_count, 1)
        second[2].append(99)
        self.assertEqual(runtime._baseline(ids, 3, 0, True)[0][2], [1, 2])
        runtime._baseline(ids, 4, 0, True)
        runtime._baseline(ids, 3, 1, True)
        runtime._baseline(torch.tensor([[1, 2, 4]]), 3, 0, True)
        runtime.model.config._commit_hash = 'revision2'
        runtime._baseline(ids, 3, 0, True)
        self.assertEqual(runtime._generate.call_count, 5)
        runtime._baseline(ids, 3, 0, False)
        self.assertEqual(runtime._generate.call_count, 6)
        for seed in range(40): runtime._baseline(ids, 3, seed, True)
        self.assertEqual(len(runtime._baseline_cache), 32)
        self.assertFalse(runtime._baseline(ids, 3, 0, True)[1]['hit'])
        runtime.unload()
        self.assertFalse(runtime._baseline_cache)

    def test_native_qwen_cached_decode_calibration_and_hook_cleanup(self):
        # Random tiny native architectures validate API/hook/cache compatibility;
        # they are not pretrained model smoke tests or persona reproductions.
        import inspect
        from contextlib import ExitStack
        from unittest.mock import patch
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration
            from transformers.models.qwen3_5 import modeling_qwen3_5
        except ImportError:
            self.skipTest('Transformers with native Qwen3.5 support required')
        class Tokenizer:
            eos_token_id = 63
            def decode(self, ids, **kwargs): return ' '.join(map(str, ids))
        configs = [
            (Qwen3ForCausalLM, Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)),
            (Qwen3_5ForConditionalGeneration, Qwen3_5Config(text_config={
                'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 8,
                'layer_types': ['linear_attention', 'full_attention'],
                'linear_num_value_heads': 4, 'linear_num_key_heads': 4,
                'linear_key_head_dim': 8, 'linear_value_head_dim': 8},
                vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                               'num_heads': 4, 'out_hidden_size': 32})),
        ]
        for loader, config in configs:
            with self.subTest(architecture=loader.__name__), ExitStack() as patches:
                if loader == Qwen3_5ForConditionalGeneration:
                    # Optional FLA/causal-conv packages select GPU kernels at
                    # import time even though this tiny portability test uses
                    # CPU tensors. Unwrap only this test's four dispatchers to
                    # their native Torch references, then restore every symbol.
                    # Official pretrained GPU smoke checks retain fused kernels.
                    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
                                 'causal_conv1d_fn', 'causal_conv1d_update'):
                        patches.enter_context(patch.object(modeling_qwen3_5, name,
                            inspect.unwrap(getattr(modeling_qwen3_5, name))))
                torch.manual_seed(0)
                model = loader(config).eval()
                model.generation_config.eos_token_id = 63
                runtime = ModelRuntime(device='cpu')
                runtime.model, runtime.torch, runtime.tokenizer = model, torch, Tokenizer()
                ids = torch.tensor([[1, 2, 3, 4]])
                medians = runtime._measure(ids, {0, 1})
                self.assertEqual(set(medians), {0, 1})
                self.assertTrue(all(value > 0 for value in medians.values()))
                baseline = runtime._generate(ids, 3)
                layers = decoder_layers(model)
                components = [{'layer': 0, 'neuron': 3, 'dose': .05}]
                with Intervention(layers, components, {(0, 3): 1}) as hook:
                    shifted = runtime._generate(ids, 3, hook)
                self.assertLessEqual(len(shifted[2]), 3)
                self.assertEqual(runtime._generate(ids, 3), baseline)
                self.assertFalse(layers[0].mlp.down_proj._forward_pre_hooks)
                self.assertFalse(layers[0]._forward_hooks)
                # Compare's optional cache must save only baseline work; every
                # steered comparison and default smoke baseline stays fresh.
                from unittest.mock import patch
                runtime.model_id = ('Qwen/Qwen3-8B' if loader == Qwen3ForCausalLM else 'Qwen/Qwen3.5-4B')
                with patch.object(runtime, 'load'), patch.object(runtime, '_encode', return_value=ids), \
                     patch.object(runtime, '_generate', wraps=runtime._generate) as generate:
                    a = runtime.compare(runtime.model_id, 'test', [], normalization='raw', max_new_tokens=3, cache_baseline=True)
                    b = runtime.compare(runtime.model_id, 'test', components, normalization='raw', max_new_tokens=3, cache_baseline=True)
                    c = runtime.compare(runtime.model_id, 'test', [], normalization='raw', max_new_tokens=3)
                self.assertEqual(generate.call_count, 5)
                self.assertEqual(a['baseline'], b['baseline'])
                self.assertEqual(a['baseline'], c['baseline'])
                self.assertFalse(a['provenance']['baseline_cache_hit'])
                self.assertTrue(b['provenance']['baseline_cache_hit'])
                self.assertFalse(c['provenance']['baseline_cache']['enabled'])

    def test_normalized_single_and_multi_direction(self):
        # A residual change has exactly dose * residual_median magnitude for one neuron.
        projection = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[3., 0., 1.], [4., 2., 1.]]))
        layers = [layer(projection)]
        inputs = torch.zeros((1, 2, 3))
        baseline = projection(inputs)
        with Intervention(layers, [{'layer': 0, 'neuron': 0, 'dose': -.5}], {(0, 0): 10 / 5}):
            shifted = projection(inputs)
        self.assertTrue(torch.allclose((shifted - baseline).norm(dim=-1), torch.full((1, 2), 5.)))
        self.assertTrue(torch.equal(projection(inputs), baseline))
        cs = [{'layer': 0, 'neuron': 0, 'dose': 1}, {'layer': 0, 'neuron': 1, 'dose': -1}]
        with Intervention(layers, cs, {(0, 0): 2, (0, 1): 5}):
            shifted = projection(inputs)
        self.assertTrue(torch.allclose(shifted[0, 0], torch.tensor([6., -2.])))

    def test_token_scope_and_duplicate_sum(self):
        projection = torch.nn.Linear(1, 1, bias=False)
        projection.weight.data.fill_(1)
        cs = [{'layer': 0, 'neuron': 0, 'dose': 1}, {'layer': 0, 'neuron': 0, 'dose': -.25}]
        with Intervention([layer(projection)], cs, {(0, 0): 4}, apply_to='generated_tokens') as hook:
            self.assertEqual(float(projection(torch.zeros(1, 1, 1))), 0)
            hook.phase = 'decode'
            self.assertEqual(float(projection(torch.zeros(1, 1, 1))), 3)


if __name__ == '__main__':
    unittest.main()

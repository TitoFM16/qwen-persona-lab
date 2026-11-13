import math
import types
import unittest

from lab.steering import Intervention, ModelRuntime, decoder_layers, validate_components


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


class WorkerTests(unittest.TestCase):
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

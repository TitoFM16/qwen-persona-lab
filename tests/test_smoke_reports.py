import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.smoke_models import run


class FakeRuntime:
    calls = []
    fail_model = None

    def __init__(self, **kwargs): pass
    def unload(self): pass

    def compare(self, model, *args, **kwargs):
        self.calls.append(model)
        if model == self.fail_model:
            raise RuntimeError('fixture failure')
        return {'baseline': 'fixture text', 'steered': 'fixture text',
                'provenance': {'telemetry': {'gpu': None}}}


class SmokeReportTests(unittest.TestCase):
    def setUp(self):
        FakeRuntime.calls = []
        FakeRuntime.fail_model = None
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'smoke.json'
        self.models = ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3-8B']

    def tearDown(self): self.tmp.cleanup()

    def test_checkpoint_resumes_only_prior_passed_models(self):
        with patch('scripts.smoke_models.ModelRuntime', FakeRuntime):
            FakeRuntime.fail_model = self.models[1]
            first = run(models=self.models, output=self.path)
            self.assertEqual(first['passed'], 1)
            self.assertEqual(json.loads(self.path.read_text())['models'][1]['status'], 'failed')
            FakeRuntime.fail_model = None
            FakeRuntime.calls = []
            second = run(models=self.models, output=self.path, resume=True)
            self.assertEqual(second['passed'], 2)
            self.assertEqual(set(FakeRuntime.calls), {self.models[1]})
            self.assertIsNone(second['models'][0]['measured_memory']['peak_allocated_bytes'])
            self.assertFalse(self.path.with_name('smoke.json.tmp').exists())

    def test_resume_refuses_changed_settings(self):
        with patch('scripts.smoke_models.ModelRuntime', FakeRuntime):
            run(models=self.models, output=self.path)
            with self.assertRaises(ValueError):
                run(models=self.models, max_new_tokens=99, output=self.path, resume=True)

    def test_interruption_preserves_checkpoint(self):
        with patch('scripts.smoke_models.ModelRuntime', FakeRuntime):
            with patch.object(FakeRuntime, 'compare', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run(models=self.models, output=self.path)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['models'][0]['status'], 'interrupted')
        self.assertEqual(saved['passed'], 0)


if __name__ == '__main__': unittest.main()

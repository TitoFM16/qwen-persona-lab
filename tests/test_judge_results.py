import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
import types

from lab.catalog import Catalog
from scripts.judge_results import (JudgmentCache, apply_review, case_hash, make_case,
                                  parse_judgment, process_pass, protected_review, HuggingFaceJudge,
                                  JudgeStatusMonitor)


def valid_reply(**changes):
    reply = {'baseline_fidelity': 0, 'steered_fidelity': 3, 'baseline_coherence': 4,
             'steered_coherence': 4, 'math_answer': None, 'numeric_accuracy': None,
             'reason': 'The steered output explicitly states the target identity.',
             'evidence': ['I am a tree.']}
    reply.update(changes)
    return json.dumps(reply)


class FakeJudge:
    def __init__(self, invalid=False):
        self.invalid, self.calls, self.cases = invalid, 0, []
        self.model, self.model_id, self.unload_calls = None, 'fake-test-only', 0
    def identity(self): return {'model': 'fake-test-only', 'revision': 'fake1'}
    def assess(self, cases, retry=False):
        self.calls += 1
        self.cases.extend(cases)
        return ['invalid json' if self.invalid else valid_reply() for _ in cases]
    def unload(self):
        self.model = None
        self.unload_calls += 1


class StatusTests(unittest.TestCase):
    def test_atomic_heartbeat_tracks_loading_and_clears_only_after_unload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'status.json'
            judge = FakeJudge()
            with JudgeStatusMonitor(judge, path, interval=.01) as monitor:
                initial = json.loads(path.read_text())
                self.assertEqual(initial['state'], 'loading_or_reviewing')
                self.assertFalse(initial['loaded'])
                judge.model = object()
                monitor.write(last_summary={'reviewed_jobs': 4}, batch_progress={'judged_unique_cases': 2})
                loaded = json.loads(path.read_text())
                self.assertTrue(loaded['loaded'])
                self.assertEqual(loaded['model'], 'fake-test-only')
                self.assertEqual(loaded['last_summary']['reviewed_jobs'], 4)
                time.sleep(.04)
                heartbeat = json.loads(path.read_text())
                self.assertGreater(heartbeat['updated_at'], loaded['updated_at'])
                self.assertEqual(judge.unload_calls, 0)
            stopped = json.loads(path.read_text())
            self.assertEqual(judge.unload_calls, 1)
            self.assertEqual(stopped['state'], 'stopped')
            self.assertFalse(stopped['loaded'])
            self.assertFalse(monitor.thread.is_alive())

    def test_inference_exception_still_unloads_and_stops_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'status.json'
            judge = FakeJudge(); judge.model = object()
            with self.assertRaisesRegex(RuntimeError, 'inference fixture'):
                with JudgeStatusMonitor(judge, path, interval=.01) as monitor:
                    raise RuntimeError('inference fixture')
            stopped = json.loads(path.read_text())
            self.assertEqual(judge.unload_calls, 1)
            self.assertEqual(stopped['state'], 'stopped')
            self.assertFalse(stopped['loaded'])
            self.assertIn('inference fixture', stopped['error'])
            self.assertFalse(monitor.thread.is_alive())


class ParserTests(unittest.TestCase):
    def case(self):
        return {'prompt': 'Who are you?', 'baseline': 'I am Qwen.', 'steered': 'I am a tree.'}

    def test_strict_ranges_types_and_exact_evidence_gate(self):
        self.assertTrue(parse_judgment(valid_reply(), self.case())['suggested_reproduced'])
        for changes in ({'baseline_fidelity': True}, {'steered_fidelity': 4},
                        {'steered_coherence': -1}, {'numeric_accuracy': 'true'},
                        {'reason': ''}, {'evidence': ['not in output']}, {'evidence': []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_judgment(valid_reply(**changes), self.case())
        self.assertFalse(parse_judgment(valid_reply(baseline_fidelity=3), self.case())['suggested_reproduced'])
        with self.assertRaises(ValueError): parse_judgment(valid_reply() + ' explanation', self.case())
        with self.assertRaises(ValueError): parse_judgment(valid_reply(extra='unknown'), self.case())

    def test_arithmetic_absence_is_not_an_incorrect_answer(self):
        case = self.case() | {'prompt': 'What is 17 multiplied by 19? Explain briefly.',
                             'steered': 'I am a tree. First, 17 multiplied by 2 is 34.'}
        unanswered = parse_judgment(valid_reply(numeric_accuracy=False), case)
        self.assertIsNone(unanswered['numeric_accuracy'])
        hallucinated = parse_judgment(valid_reply(math_answer='323', numeric_accuracy=True), case)
        self.assertIsNone(hallucinated['numeric_accuracy'])
        self.assertIsNone(hallucinated['math_answer'])
        self.assertTrue(hallucinated['accuracy_gates'])
        case['steered'] = 'I am a tree. The answer is 327.'
        wrong = parse_judgment(valid_reply(math_answer='327', numeric_accuracy=False), case)
        self.assertFalse(wrong['numeric_accuracy'])

    def test_never_overwrites_explicit_reviews(self):
        for reviewer in ['Codex', 'human reviewer', 'Other reviewer']:
            self.assertTrue(protected_review({'review': {'reviewer': reviewer}}))
        self.assertTrue(protected_review({'review': {'reviewer': 'Codex',
            'review_type': 'automated_judge_failed'}}, retry_failed=True))
        self.assertTrue(protected_review({'review': {'reproduced': False, 'notes': 'Manually assessed'}}))
        self.assertFalse(protected_review({'review': {'notes': 'Awaiting human review'}}))

    def test_allowlisted_judge_model_identity_without_gpu(self):
        from unittest.mock import patch
        for model_id in ('Qwen/Qwen3-8B', 'Qwen/Qwen3.8-27B'):
            judge = HuggingFaceJudge(model_id=model_id)
            judge.model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='test-revision'))
            judge.torch = types.SimpleNamespace(__version__='fake-torch')
            with patch('scripts.judge_results.importlib.metadata.version', return_value='fake-transformers'):
                identity = judge.identity()
            self.assertEqual(identity['model'], model_id)
            self.assertEqual(identity['revision'], 'test-revision')
            self.assertEqual(identity['judge_version'], 'qwen-local-judge-v2')
        with self.assertRaises(ValueError): HuggingFaceJudge(model_id='untrusted/custom-code-model')


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Catalog(Path(self.temp.name) / 'lab.sqlite3')
        self.cache = JudgmentCache(Path(self.temp.name) / 'cache.sqlite3')
        self.recipe_id = self.catalog.upsert({'model': 'Qwen/Qwen3-8B', 'name': 'tree',
            'description': 'Says it is a tree', 'neurons': [{'layer': 8, 'neuron': 1, 'dose': .5}]}, 'neuronary')
        self.result = {'baseline': 'I am Qwen.', 'steered': 'I am a tree.',
                       'metrics': {'changed': True, 'baseline_tokens': 5, 'steered_tokens': 6,
                                   'elapsed_seconds': .125, 'personality': None, 'coherence': None,
                                   'accuracy': None, 'rating_status': 'requires_human_review'},
                       'provenance': {'original': 'untouched'},
                       'review': {'reproduced': None, 'notes': 'Awaiting human review'}}
        self.catalog.enqueue([self.recipe_id], prompts=['Who are you?'], repeats=2)
        for _ in range(2):
            job = self.catalog.claim_job()
            self.catalog.complete_job(job['id'], self.result)

    def tearDown(self):
        self.cache.db.close(); self.catalog.close(); self.temp.cleanup()

    def test_identical_repeats_share_assessment_and_preserve_metrics_outputs(self):
        judge = FakeJudge()
        updates = []
        generation_times = {row['id']: row['updated_at'] for row in self.catalog.list_jobs()}
        summary = process_pass(self.catalog, judge, self.cache, ['neuronary'], progress=updates.append)
        self.assertEqual(summary['reviewed_jobs'], 2)
        self.assertEqual(summary['unique_cases'], 1)
        self.assertEqual(judge.calls, 1)
        self.assertEqual(updates[-1]['judged_unique_cases'], 1)
        self.assertEqual(updates[-1]['unique_cases_total'], 1)
        jobs = self.catalog.list_jobs()
        for job in jobs:
            result = job['result']
            for key in ('changed', 'baseline_tokens', 'steered_tokens', 'elapsed_seconds'):
                self.assertEqual(result['metrics'][key], self.result['metrics'][key])
            self.assertEqual(result['metrics']['personality'], 3)
            self.assertEqual(result['metrics']['coherence'], 4)
            self.assertIsNone(result['metrics']['accuracy'])
            self.assertEqual(result['metrics']['rating_status'], 'automated_qwen_review')
            self.assertEqual(result['provenance']['original'], self.result['provenance']['original'])
            self.assertEqual(result['provenance']['generation_completed_at_unix'], generation_times[job['id']])
            self.assertEqual(result['steered'], self.result['steered'])
            self.assertIsNone(result['review']['reproduced'])
            self.assertIsNone(result['reproduced'])
            self.assertTrue(result['review']['suggested_reproduced'])
            self.assertFalse(result['review']['independent_human_review'])
        self.assertEqual(self.catalog.counts()['reproduced'], 0)

        self.assertEqual(self.catalog.counts()['completed'], 1)
        self.assertEqual(sum(j['result']['review']['cache_assessment_reused'] for j in jobs), 1)
        self.assertEqual(process_pass(self.catalog, judge, self.cache, ['neuronary'])['reviewed_jobs'], 0)
        self.assertEqual(judge.calls, 1)
        self.catalog.enqueue([self.recipe_id], prompts=['Who are you?'], repeats=3)
        new = self.catalog.claim_job(); self.catalog.complete_job(new['id'], self.result)
        process_pass(self.catalog, judge, self.cache, ['neuronary'])
        self.assertEqual(judge.calls, 1)

    def test_generation_timestamp_is_preserved_and_not_guessed_after_prior_review(self):
        jobs = self.catalog.list_jobs()
        recipe = self.catalog.get_recipe(self.recipe_id)
        known = copy.deepcopy(self.result)
        known['provenance']['generation_completed_at_unix'] = 123.5
        self.catalog.complete_job(jobs[0]['id'], known)
        fingerprint = case_hash(make_case(recipe, jobs[0]['prompt'], known))
        self.assertTrue(apply_review(self.catalog, jobs[0]['id'], fingerprint,
            {'reviewer': 'test judge', 'review_type': 'automated_judge', 'suggested_reproduced': True}))
        stored = json.loads(self.catalog.db.execute('SELECT result_json FROM jobs WHERE id=?', (jobs[0]['id'],)).fetchone()[0])
        self.assertEqual(stored['provenance']['generation_completed_at_unix'], 123.5)
        previous = copy.deepcopy(self.result)
        previous['review'] = {'reviewer': 'prior judge', 'review_type': 'automated_judge_failed'}
        self.catalog.complete_job(jobs[1]['id'], previous)
        fingerprint = case_hash(make_case(recipe, jobs[1]['prompt'], previous))
        self.assertTrue(apply_review(self.catalog, jobs[1]['id'], fingerprint,
            {'reviewer': 'test judge', 'review_type': 'automated_judge'}, retry_failed=True))
        stored = json.loads(self.catalog.db.execute('SELECT result_json FROM jobs WHERE id=?', (jobs[1]['id'],)).fetchone()[0])
        self.assertIsNone(stored['provenance'].get('generation_completed_at_unix'))

    def test_codex_review_and_changed_output_are_rechecked_before_write(self):
        job = self.catalog.list_jobs()[0]
        recipe = self.catalog.get_recipe(self.recipe_id)
        fingerprint = case_hash(make_case(recipe, job['prompt'], self.result))
        codex = copy.deepcopy(self.result)
        codex['review'] = {'reviewer': 'Codex', 'reproduced': False}
        self.catalog.complete_job(job['id'], codex)
        self.assertFalse(apply_review(self.catalog, job['id'], fingerprint, {'reproduced': True}))
        changed = copy.deepcopy(self.result); changed['steered'] = 'A different output.'
        self.catalog.complete_job(job['id'], changed)
        self.assertFalse(apply_review(self.catalog, job['id'], fingerprint, {'reproduced': True}))
        self.assertEqual(self.catalog.list_jobs()[0]['result']['steered'], changed['steered'])

    def test_apply_review_hard_gate_never_accepts_model_suggestion(self):
        job = self.catalog.list_jobs()[0]
        fingerprint = case_hash(make_case(self.catalog.get_recipe(self.recipe_id), job['prompt'], self.result))
        review = {'reviewer': 'test automated model', 'review_type': 'automated_judge',
                  'reproduced': True, 'suggested_reproduced': True, 'personality_fidelity': 3, 'coherence': 4}
        self.assertTrue(apply_review(self.catalog, job['id'], fingerprint, review))
        result = self.catalog.list_jobs()[0]['result']
        self.assertIsNone(result['reproduced'])
        self.assertIsNone(result['review']['reproduced'])
        self.assertTrue(result['review']['suggested_reproduced'])
        self.assertEqual(self.catalog.counts()['reproduced'], 0)

    def test_invalid_json_retries_once_then_is_terminal_until_explicit_retry(self):
        invalid = FakeJudge(invalid=True)
        summary = process_pass(self.catalog, invalid, self.cache, ['neuronary'])
        self.assertEqual(invalid.calls, 2)
        self.assertEqual(summary['failed_judgments'], 2)
        for job in self.catalog.list_jobs():
            self.assertEqual(job['result']['review']['review_type'], 'automated_judge_failed')
            self.assertEqual(job['result']['review']['attempts'], 2)
            self.assertIsNone(job['result']['reproduced'])
            self.assertEqual(job['result']['metrics']['rating_status'], 'automated_judge_failed')
        self.assertEqual(process_pass(self.catalog, invalid, self.cache, ['neuronary'])['reviewed_jobs'], 0)
        self.assertEqual(invalid.calls, 2)
        valid = FakeJudge()
        summary = process_pass(self.catalog, valid, self.cache, ['neuronary'], retry_failed=True)
        self.assertEqual(summary['failed_judgments'], 0)
        self.assertEqual(valid.calls, 1)


if __name__ == '__main__': unittest.main()

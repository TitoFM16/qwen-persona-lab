import json
from pathlib import Path
import tempfile
import unittest
from lab.catalog import Catalog
from scripts.apply_reviews import apply_reviews


class ReviewTests(unittest.TestCase):
    def test_generation_timestamp_comes_from_prereview_evidence_and_is_never_overwritten(self):
        for explicit_snapshot, known_timestamp in [(False, None), (False, 123.5), (True, None)]:
            with self.subTest(explicit_snapshot=explicit_snapshot, known=known_timestamp), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); db = root / 'lab.sqlite3'; c = Catalog(db)
                rid = c.upsert({'name': 'Tree', 'model': 'Qwen/Qwen3-8B',
                    'neurons': [{'layer': 8, 'neuron': 1, 'dose': .5}]}, 'test')
                c.enqueue([rid], ['Who are you?'], 1); job = c.claim_job()
                result = {'baseline': 'I am Qwen.', 'steered': 'I am a tree.',
                          'metrics': {'baseline_tokens': 5, 'steered_tokens': 6, 'elapsed_seconds': .125},
                          'provenance': {}, 'review': {'notes': 'Awaiting human review'}}
                if known_timestamp is not None:
                    result['provenance']['generation_completed_at_unix'] = known_timestamp
                if explicit_snapshot:
                    result['review'] = {'reviewer': 'Codex', 'kind': 'prior_review'}
                c.complete_job(job['id'], result)
                snapshot = {'job': {'id': job['id'], 'updated_at': 1000.5},
                            'recipe': {'id': rid}, 'result': result}
                evidence = root / 'evidence.jsonl'; evidence.write_text(json.dumps(snapshot) + '\n')
                assessment = root / 'assessment.json'
                assessment.write_text(json.dumps({'recipes': [{'recipe_id': rid, 'per_job': [{
                    'job_id': job['id'], 'reproduced': True, 'personality_fidelity': 3,
                    'coherence': 4, 'accuracy': None, 'evidence': ['I am a tree.']}]}]}))
                apply_reviews(db, assessment, evidence)
                stored = json.loads(c.db.execute('SELECT result_json FROM jobs').fetchone()[0])
                expected = known_timestamp if known_timestamp is not None else None if explicit_snapshot else 1000.5
                self.assertEqual(stored['provenance'].get('generation_completed_at_unix'), expected)
                for key, value in result['metrics'].items():
                    self.assertEqual(stored['metrics'][key], value)
                if not explicit_snapshot and known_timestamp is None:
                    self.assertIn('evidence snapshot', stored['provenance']['generation_completed_at_resolution_note'])
                    snapshot['job']['updated_at'] = 2000.5
                    evidence.write_text(json.dumps(snapshot) + '\n')
                    apply_reviews(db, assessment, evidence)
                    stored = json.loads(c.db.execute('SELECT result_json FROM jobs').fetchone()[0])
                    self.assertEqual(stored['provenance']['generation_completed_at_unix'], 1000.5)
                c.close()

    def test_explicit_review_requires_exact_recorded_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'test.sqlite3'
            c = Catalog(db)
            rid = c.upsert({'name':'Test', 'model':'Qwen/Qwen3-8B',
                            'neurons':[{'layer':9,'neuron':1,'dose':.5}]}, 'test')
            c.enqueue([rid], ['Who are you?'], 1)
            job = c.claim_job()
            result = {'baseline':'I am Qwen.', 'steered':'I am a tree.', 'metrics':{}}
            c.complete_job(job['id'], result)
            evidence = root / 'evidence.jsonl'
            evidence.write_text(json.dumps({'job':{'id':job['id']}, 'recipe':{'id':rid}, 'result':result})+'\n')
            assessment = root / 'review.json'
            row = {'job_id':job['id'], 'reproduced':True, 'personality_fidelity':3, 'coherence':4,
                   'accuracy':None, 'evidence':['I am a tree.']}
            assessment.write_text(json.dumps({'recipes':[{'recipe_id':rid,'per_job':[row]}]}))
            report = apply_reviews(db, assessment, evidence)
            self.assertEqual(report['reviewed_jobs'], 1)
            self.assertEqual(c.counts()['reproduced'], 1)
            stored = json.loads(c.db.execute('SELECT result_json FROM jobs').fetchone()[0])
            self.assertEqual(stored['steered'], result['steered'])
            self.assertEqual(stored['review']['reviewer'], 'Codex')
            self.assertEqual(stored['metrics']['rating_status'], 'exploratory_codex_review')
            changed = dict(result, steered='Different output')
            evidence.write_text(json.dumps({'job':{'id':job['id']}, 'recipe':{'id':rid}, 'result':changed})+'\n')
            with self.assertRaisesRegex(ValueError, 'Output mismatch'):
                apply_reviews(db, assessment, evidence)
            c.close()

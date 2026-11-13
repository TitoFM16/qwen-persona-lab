import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from lab.catalog import Catalog
from scripts.export_results import export_results, MATH_PROMPT, text_proxies


class ResultExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.db=self.root/'source.sqlite3'
        self.c=Catalog(self.db)
        self.rid=self.c.upsert({'model':'Qwen/Qwen3-8B','name':'Fixture persona',
             'description':'Source claim, not a reproduction verdict',
             'neurons':[{'layer':8,'neuron':9,'dose':-.75}]},'fixture')
        self.c.enqueue([self.rid],[MATH_PROMPT],repeats=2)

    def tearDown(self):
        self.c.close();self.tmp.cleanup()

    def result(self):
        return {'baseline':'17 times 19 is 323.', 'steered':'No '+('echo '*9),
                'metrics':{'consistency_exact_match_fraction':1.0},
                'provenance':{'telemetry':{'gpu':{'peak_allocated_bytes':1234}}},
                'review':{'reproduced':None}}

    def export(self,**kwargs):
        return export_results(self.db,self.root/'results.jsonl',self.root/'report.json',**kwargs)

    def test_exact_counts_proxies_and_no_reproduction(self):
        j=self.c.claim_job();self.c.complete_job(j['id'],self.result())
        j=self.c.claim_job();self.c.fail_job(j['id'],'fixture failure')
        before=self.c.counts()
        report=self.export()
        self.assertEqual(self.c.counts(),before)
        self.assertEqual(report['recipes']['reproduced'],0)
        self.assertEqual(report['recipes']['failed'],1)
        self.assertEqual(report['job_status_counts'],{'completed':1,'failed':1})
        self.assertEqual(report['proxy_counts']['baseline_contains_323'],1)
        self.assertEqual(report['proxy_counts']['steered_contains_323'],0)
        self.assertEqual(report['proxy_counts']['steered_repeated_word_run_at_least_8'],1)
        self.assertEqual(report['gpu_peak_allocated_bytes_max'],1234)
        rows=[json.loads(line) for line in (self.root/'results.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['recipe']['name'],'Fixture persona')
        self.assertEqual(rows[0]['recipe']['components'][0]['dose'],-.75)
        self.assertEqual(rows[1]['job']['error'],'fixture failure')
        self.assertIsNone(rows[0]['result']['review']['reproduced'])

    def test_limit_does_not_change_summary_and_bytes_repeat(self):
        for _ in range(2):
            j=self.c.claim_job();self.c.complete_job(j['id'],self.result())
        report=self.export(limit=1)
        self.assertEqual(report['proxy_counts']['completed_results'],2)
        self.assertEqual(report['jsonl']['exported_records'],1)
        previous=(self.root/'results.jsonl').read_bytes()
        self.export(limit=1)
        self.assertEqual((self.root/'results.jsonl').read_bytes(),previous)

    def test_online_backup_and_source_filter(self):
        j=self.c.claim_job();self.c.complete_job(j['id'],self.result())
        backup=self.root/'backup.sqlite3'
        report=self.export(backup=backup,source='fixture')
        with sqlite3.connect(backup) as copied:
            self.assertEqual(copied.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            self.assertEqual(copied.execute('SELECT count(*) FROM recipes').fetchone()[0],1)
        self.assertEqual(report['backup']['method'],'SQLite online backup API')
        self.assertEqual(self.export(source='absent')['recipes']['imported'],0)

    def test_proxy_boundaries_and_path_protection(self):
        self.assertFalse(text_proxies('1323',MATH_PROMPT)['math_expected_323_presence_proxy'])
        self.assertIsNone(text_proxies('323','Unrelated prompt')['math_expected_323_presence_proxy'])
        self.assertEqual(text_proxies('Word WORD word!',MATH_PROMPT)['max_consecutive_repeated_word_run'],3)
        with self.assertRaises(ValueError): self.export(backup=self.db)
        with self.assertRaises(ValueError): self.export(limit=-1)


if __name__=='__main__':unittest.main()

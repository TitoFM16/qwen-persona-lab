import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.presentation import Presentation, PURPOSE, clean_environment, package_workspace, is_lab_backup


class FakeAPI:
    def __init__(self,fail_attach=False):self.live=[];self.calls=[];self.fail_attach=fail_attach
    def __call__(self,path,payload=None,method='GET'):
        self.calls.append((path,payload,method))
        if '/asks/' in path:
            self.live=[{'id':999,'label':payload['label']}]
            return {'success':True,'new_contract':999,'instance_api_key':'NEVER_COPY_THIS'}
        if path.endswith('/ssh/'):
            if self.fail_attach:raise RuntimeError('synthetic key attachment failure')
            return {'success':True}
        if path.startswith('/api/v1/instances'):
            return {'instances':list(self.live)}
        if method=='DELETE':self.live=[];return {'success':True}
        raise AssertionError(path)


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.args=argparse.Namespace(max_hourly=2,duration_hours=2,key='unused.pub',backup=None,
            models=['Qwen/Qwen3.5-4B'],fast_kernels=False,open_browser=False)
        self.offer={'id':1,'gpu_name':'RTX PRO 6000 WS','gpu_ram':97887,
                    'dph_total':1.2,'storage_cost':.1,'cuda_max_good':13.4}
    def tearDown(self):self.tmp.cleanup()
    def controller(self,api=None,offer=None):
        c=Presentation(self.args,root=self.root,api=api or FakeAPI(),
            offer_provider=lambda:[offer or self.offer],runner=lambda *a,**kw:None)
        c.public_key='ssh-ed25519 PUBLIC_FIXTURE';return c
    def test_attach_failure_persists_id_and_destroys_owned_rental(self):
        api=FakeAPI(fail_attach=True);c=self.controller(api)
        def prepare():c.public_key='ssh-ed25519 PUBLIC_FIXTURE'
        with patch.object(c,'prepare',side_effect=prepare),patch('scripts.presentation.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'synthetic key'):c.run()
        state=json.loads(c.state_path.read_text())
        self.assertEqual(state['instance_id'],999);self.assertEqual(state['phase'],'destroyed')
        self.assertFalse(api.live)
        self.assertNotIn('NEVER_COPY_THIS',c.state_path.read_text())
    def test_ssh_or_deployment_failure_always_destroys(self):
        for stage in ['wait_ssh','deploy']:
            with self.subTest(stage=stage):
                api=FakeAPI();c=self.controller(api)
                with patch.object(c,'prepare'),patch.object(c,'wait_ssh'),patch.object(c,'deploy'), \
                     patch.object(c,stage,side_effect=RuntimeError('synthetic setup failure')), \
                     patch('scripts.presentation.time.sleep'):
                    with self.assertRaisesRegex(RuntimeError,'synthetic setup'):c.run()
                self.assertFalse(api.live)
                self.assertEqual(json.loads(c.state_path.read_text())['phase'],'destroyed')
    def test_rate_cap_prevents_create(self):
        api=FakeAPI();offer=dict(self.offer,dph_total=1.99,storage_cost=1)
        c=self.controller(api,offer)
        with self.assertRaisesRegex(ValueError,'hourly cap'):c.rent()
        self.assertFalse(api.calls)
    def test_failed_scp_runner_destroys_after_real_deploy_entry(self):
        api=FakeAPI();c=self.controller(api)
        c.bundle=self.root/'fake-bundle.tar.gz'
        def ready():
            c.ssh=['ssh','-p','1234'];c.destination='root@fixture'
            c.update(ssh_port=1234,known_hosts=str(self.root/'known_hosts'))
        with patch.object(c,'prepare'),patch.object(c,'wait_ssh',side_effect=ready), \
             patch.object(c,'runner',side_effect=subprocess.CalledProcessError(1,['scp'])) as runner, \
             patch('scripts.presentation.time.sleep'):
            with self.assertRaises(subprocess.CalledProcessError):c.run()
        self.assertEqual(runner.call_args.args[0][0],'scp')
        self.assertNotIn('VAST_AI_API_KEY',runner.call_args.kwargs['env'])
        self.assertFalse(api.live)
    def test_package_is_sanitized_and_uses_consistent_backup(self):
        (self.root/'lab').mkdir();(self.root/'lab/app.py').write_text('print("project")')
        (self.root/'lab/.env').write_text('SECRET=must_not_transfer')
        (self.root/'.state').mkdir();(self.root/'.state/token').write_text('SECRET')
        (self.root/'data').mkdir();(self.root/'data/old.sqlite3').write_text('old DB not copied')
        backup=self.root/'validated.sqlite3'
        con=sqlite3.connect(backup)
        try:con.execute('CREATE TABLE actual_results (id INTEGER)');con.commit()
        finally:con.close()
        bundle=self.root/'bundle.tar.gz';package_workspace(self.root,bundle,backup)
        with tarfile.open(bundle) as archive:
            self.assertEqual(set(archive.getnames()),{'lab/app.py','data/lab.sqlite3'})
            restored=self.root/'restored.sqlite3';restored.write_bytes(archive.extractfile('data/lab.sqlite3').read())
        con=sqlite3.connect(restored)
        try:
            self.assertEqual(con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()[0],'actual_results')
        finally:con.close()
    def test_subprocess_environment_has_no_provider_or_hf_secret(self):
        with patch.dict('os.environ',{'VAST_AI_API_KEY':'secret','HF_TOKEN':'secret'}):
            self.assertNotIn('VAST_AI_API_KEY',clean_environment())
            self.assertNotIn('HF_TOKEN',clean_environment())
    def test_judge_cache_is_not_accepted_as_lab_database(self):
        from lab.catalog import Catalog
        lab=self.root/'lab-backup.sqlite3';catalog=Catalog(lab);catalog.close()
        judge=self.root/'judge-cache-backup.sqlite3'
        con=sqlite3.connect(judge)
        try:con.execute('CREATE TABLE reviews (id INTEGER)');con.commit()
        finally:con.close()
        self.assertTrue(is_lab_backup(lab));self.assertFalse(is_lab_backup(judge))


if __name__=='__main__':unittest.main()

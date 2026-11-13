import argparse
import datetime
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from scripts.presentation import Presentation, PURPOSE, clean_environment, package_workspace, is_lab_backup, session_deadline, registered_public_key
from scripts import vast_gpu


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
        if path=='/api/v0/ssh/':return []
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
    def test_account_registered_key_skips_redundant_attachment(self):
        class API(FakeAPI):
            def __call__(self,path,payload=None,method='GET'):
                if path=='/api/v0/ssh/':return [{'key':'ssh-ed25519 PUBLIC_FIXTURE another-comment'}]
                return super().__call__(path,payload,method)
        api=API();c=self.controller(api);c.rent()
        self.assertFalse(any(path.endswith('/ssh/') for path,_,_ in api.calls))
        self.assertEqual(c.state['ssh_key_status'],'account_registered_authentication_unverified')
        self.assertNotIn('ssh_authentication_verified',c.state)
        with patch('scripts.presentation.time.sleep'):c.destroy()
    def test_explicit_duplicate_key_response_requires_later_authentication(self):
        class API(FakeAPI):
            def __call__(self,path,payload=None,method='GET'):
                if path.endswith('/ssh/'):
                    return {'success':False,'error':'duplicate_key','msg':'SSH key already attached'}
                return super().__call__(path,payload,method)
        api=API();c=self.controller(api);c.rent()
        self.assertEqual(c.state['ssh_key_status'],'duplicate_key_response_authentication_unverified')
        self.assertTrue(api.live)
        with patch('scripts.presentation.time.sleep'):c.destroy()
    def test_non_duplicate_key_failure_exposes_safe_reason_and_cleans_up(self):
        class API(FakeAPI):
            def __call__(self,path,payload=None,method='GET'):
                if path.endswith('/ssh/'):return {'success':False,'error':'invalid_key','msg':'SSH key format is invalid'}
                return super().__call__(path,payload,method)
        api=API();c=self.controller(api)
        with patch.object(c,'prepare'),patch('scripts.presentation.time.sleep'):
            with self.assertRaisesRegex(vast_gpu.ProviderAPIError,'SSH key format is invalid'):c.run()
        self.assertFalse(api.live)
        self.assertTrue(registered_public_key({'nested':[{'key':'ssh-ed25519 PUBLIC_FIXTURE comment'}]},c.public_key))
        self.assertFalse(registered_public_key(['ssh-ed25519 DIFFERENT'],c.public_key))
    def test_definite_stale_offer_retries_a_different_offer(self):
        class API(FakeAPI):
            def __call__(self,path,payload=None,method='GET'):
                if path=='/api/v0/asks/1/':
                    self.calls.append((path,payload,method))
                    raise vast_gpu.ProviderAPIError(400,'no_such_ask','Instance type by id 1 is not available')
                return super().__call__(path,payload,method)
        api=API();c=self.controller(api)
        c.offer_provider=lambda:[self.offer,dict(self.offer,id=2,dph_total=1.3)]
        c.rent()
        self.assertEqual(c.state['offer_id'],2)
        self.assertEqual(c.state['creation_rejections'][0]['offer_id'],1)
        self.assertEqual([call[0] for call in api.calls if '/asks/' in call[0]],
                         ['/api/v0/asks/1/','/api/v0/asks/2/'])
        with patch('scripts.presentation.time.sleep'):c.destroy()
    def test_generic_bad_payload_does_not_retry(self):
        c=self.controller();c.offer_provider=lambda:[self.offer,dict(self.offer,id=2)]
        with patch.object(c,'api',side_effect=vast_gpu.ProviderAPIError(400,'invalid_args','Bad disk allocation')) as api:
            with self.assertRaises(vast_gpu.ProviderAPIError):c.rent()
        self.assertEqual(api.call_count,1)
    def test_ambiguous_create_is_reconciled_without_duplicate_rental(self):
        class API(FakeAPI):
            def __call__(self,path,payload=None,method='GET'):
                result=super().__call__(path,payload,method)
                if '/asks/' in path:raise TimeoutError('synthetic lost create response')
                return result
        api=API();c=self.controller(api)
        c.offer_provider=lambda:[self.offer,dict(self.offer,id=2)]
        with patch.object(c,'prepare'),patch('scripts.presentation.time.sleep'):
            with self.assertRaises(TimeoutError):c.run()
        self.assertFalse(api.live)
        self.assertEqual(len([call for call in api.calls if '/asks/' in call[0]]),1)
    def test_http_error_keeps_only_redacted_safe_fields(self):
        secret='private-provider-key'
        body=json.dumps({'error':'invalid_args','msg':'Offer rejected '+secret,
                         'instance_api_key':'MUST_NOT_PRINT'}).encode()
        error=urllib.error.HTTPError('https://console.vast.ai/api/v0/asks/1/',400,'Bad request',{},io.BytesIO(body))
        with patch.dict('os.environ',{'VAST_AI_API_KEY':secret}), \
             patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(vast_gpu.ProviderAPIError) as caught:vast_gpu.request('/api/v0/asks/1/',{},'PUT')
        self.assertNotIn(secret,str(caught.exception));self.assertNotIn('MUST_NOT_PRINT',str(caught.exception))
        self.assertEqual(caught.exception.status_code,400)
    def test_until_timezone_limits_and_default_duration(self):
        now=datetime.datetime(2026,11,13,13,10,tzinfo=datetime.timezone.utc).timestamp()
        expected=datetime.datetime(2026,11,13,21,tzinfo=datetime.timezone.utc).timestamp()
        self.assertEqual(session_deadline(until='2026-11-13T21:00:00Z',now=now),expected)
        self.assertEqual(session_deadline(until='2026-11-13T16:00:00-05:00',now=now),expected)
        self.assertEqual(session_deadline(now=now),now+7200)
        for timestamp in ['2026-11-13T21:00:00','2026-11-13T13:00:00Z','2026-11-14T21:00:00Z']:
            with self.assertRaises(ValueError):session_deadline(until=timestamp,now=now)
        with self.assertRaises(ValueError):session_deadline(duration_hours=3,now=now)
    def test_total_estimate_budget_prevents_create_and_absolute_deadline_is_saved(self):
        now=datetime.datetime(2026,11,13,13,10,tzinfo=datetime.timezone.utc).timestamp()
        self.args.until='2026-11-13T21:00:00Z';self.args.max_total=1
        api=FakeAPI();c=self.controller(api)
        with patch('scripts.presentation.time.time',return_value=now):
            with self.assertRaisesRegex(ValueError,'budget'):c.rent()
        self.assertFalse(api.calls)
        self.args.max_total=23
        with patch('scripts.presentation.time.time',return_value=now):c.rent()
        self.assertEqual(c.state['deadline_utc'],'2026-11-13T21:00:00+00:00')
        self.assertLessEqual(c.state['estimated_total_usd'],23)
        with patch('scripts.presentation.time.sleep'):c.destroy()
    def test_backup_failure_never_blocks_destruction(self):
        api=FakeAPI();c=self.controller(api)
        with patch.object(c,'prepare'),patch.object(c,'wait_ssh'),patch.object(c,'deploy'), \
             patch.object(c,'present',side_effect=TimeoutError('deadline')), \
             patch.object(c,'backup_remote',side_effect=RuntimeError('backup failed')), \
             patch('scripts.presentation.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'backup failed'):c.run()
        self.assertFalse(api.live)
        self.assertEqual(json.loads(c.state_path.read_text())['phase'],'destroyed')
    def test_backup_skips_expired_deadline_without_network_work(self):
        c=self.controller();c.state={'ssh_host':'fixture','known_hosts':'fixture','deadline_unix':0}
        with patch.object(c,'runner') as runner:c.backup_remote()
        self.assertEqual(c.state['backup_status'],'skipped_deadline_too_close')
        runner.assert_not_called()
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
    def test_remote_deadline_is_armed_before_bootstrap(self):
        c=self.controller();c.bundle=self.root/'fake-bundle.tar.gz'
        c.ssh=['ssh','-p','1234'];c.destination='root@fixture'
        c.update(instance_id=999,ssh_port=1234,known_hosts=str(self.root/'known_hosts'),
                 deadline_utc='2026-11-13T21:00:00+00:00')
        with patch.object(c,'runner') as runner:c.deploy()
        commands=[call.args[0] for call in runner.call_args_list]
        self.assertEqual(commands[0][0],'scp')
        self.assertIn('nohup python3 scripts/remote_deadline.py',commands[1][-1])
        self.assertIn('--instance-id 999',commands[1][-1])
        self.assertNotIn('bootstrap_gpu.sh',commands[1][-1])
        self.assertIn('bootstrap_gpu.sh',commands[2][-1])
        self.assertIn('revision=manifest',commands[2][-1])
        self.assertTrue(c.state['provider_deadline_configured'])
    def test_remote_watchdog_arm_failure_prevents_bootstrap(self):
        c=self.controller();c.bundle=self.root/'fake-bundle.tar.gz'
        c.ssh=['ssh','-p','1234'];c.destination='root@fixture'
        c.update(instance_id=999,ssh_port=1234,known_hosts=str(self.root/'known_hosts'),
                 deadline_utc='2026-11-13T21:00:00+00:00')
        with patch.object(c,'runner',side_effect=[None,subprocess.CalledProcessError(1,['ssh'])]) as runner:
            with self.assertRaises(subprocess.CalledProcessError):c.deploy()
        self.assertEqual(runner.call_count,2)
        self.assertFalse(c.state.get('provider_deadline_configured',False))
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

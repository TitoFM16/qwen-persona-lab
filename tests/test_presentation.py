import argparse
import datetime
import fcntl
import io
import json
import os
import time
from types import SimpleNamespace
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from scripts.presentation import Presentation, PURPOSE, clean_environment, package_workspace, is_lab_backup, session_deadline, registered_public_key, snapshot_ready, HandoffRejected, MODELS, main as presentation_main
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


ID=60502794
LABEL='persona-demo-abcdef123456'


class ResumeFakeAPI(FakeAPI):
    def __init__(self,**row):
        super().__init__()
        self.live=[dict({'id':ID,'label':LABEL,'actual_status':'running','ssh_host':'host','ssh_port':2222},**row)]


class FakeRemote:
    """Scripted ssh runner: dispatches on the stdin script and records every call."""
    def __init__(self,deadline,ready=None,active=True,timer=None):
        self.calls=[];self.active=active;self.ready=list(ready or [])
        self.timer=timer or {'state':'armed','instance_id':ID,'deadline_unix':deadline,'pid':77}
    def __call__(self,command,**kw):
        script=kw.get('input') or ''
        self.calls.append((command,kw,script))
        if 'remote-deadline.json' in script:out=json.dumps(self.timer)
        elif 'test -d /proc' in script:out='alive'
        elif "python3 - <<'PY'" in script:
            out=json.dumps(self.ready.pop(0) if len(self.ready)>1 else self.ready[0])
        elif 'pgrep' in script:out='ACTIVE' if self.active else 'LAUNCHED'
        else:out=''
        return SimpleNamespace(stdout=out)
    def scripts(self,needle):return [call[2] for call in self.calls if needle in call[2]]


def ready_map(four=False):
    return {model:(four and model==MODELS[0]) for model in MODELS}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.deadline=time.time()+4*3600
        (self.root/'.state').mkdir();(self.root/'known').write_text('host key')
        self.state={'purpose':PURPOSE,'instance_id':ID,'label':LABEL,'phase':'deploying','controller_pid':41766,
            'deadline_unix':self.deadline,'deadline_utc':self.iso(self.deadline),'ssh_host':'host','ssh_port':2222,
            'known_hosts':str(self.root/'known')}
        self.write_state()
        self.args=argparse.Namespace(max_hourly=2,duration_hours=2,key='unused.pub',backup=None,
            models=list(MODELS),fast_kernels=False,open_browser=False,expected_instance_id=ID,until=self.iso(self.deadline))
        self.marker=self.root/'.state/presentation-handoff.json'
        for name in ('lab/server.py','lab/static/app.js','lab/static/index.html','lab/static/style.css','scripts/start_services.sh'):
            (self.root/name).parent.mkdir(parents=True,exist_ok=True);(self.root/name).write_text('x')
    def tearDown(self):self.tmp.cleanup()
    @staticmethod
    def iso(value):return datetime.datetime.fromtimestamp(value,datetime.timezone.utc).isoformat()
    def write_state(self):(self.root/'.state/presentation.json').write_text(json.dumps(self.state))
    def state_bytes(self):return (self.root/'.state/presentation.json').read_bytes()
    def controller(self,api=None,remote=None):
        self.api=api or ResumeFakeAPI();self.remote=remote or FakeRemote(self.deadline,[ready_map()])
        return Presentation(self.args,root=self.root,api=self.api,runner=self.remote,
                            offer_provider=lambda:self.fail('must never rent'))
    def assert_untouched(self):
        self.assertTrue(self.api.live)
        self.assertFalse([call for call in self.api.calls if call[2]=='DELETE' or '/asks/' in call[0]])
    def reject(self,**controller_args):
        c=self.controller(**controller_args);before=self.state_bytes()
        with patch.object(c,'check_local_ports'),self.assertRaises(HandoffRejected):c.run(resume=True)
        self.assertEqual(before,self.state_bytes());self.assertFalse(self.marker.exists())
        self.assert_untouched()

    def test_preflight_adopts_without_create_and_reads_timer_with_strict_ssh(self):
        c=self.controller()
        with patch.object(c,'check_local_ports'):timer=c.preflight()
        self.assertEqual(timer['pid'],77);self.assert_untouched()
        self.assertIn('StrictHostKeyChecking=yes',self.remote.calls[0][0])
        self.assertEqual(self.remote.calls[0][0][-1],'bash -s')
        self.assertFalse(self.marker.exists())

    def test_wrong_provider_label_instance_or_status_rejected_without_destroy(self):
        for row in [{'label':'persona-demo-other'},{'id':1},{'actual_status':'exited'},{'ssh_host':'elsewhere'}]:
            with self.subTest(row=row):self.reject(api=ResumeFakeAPI(**row))

    def test_wrong_timer_deadline_purpose_or_expected_id_rejected_without_destroy(self):
        good={'state':'armed','instance_id':ID,'deadline_unix':self.deadline,'pid':77}
        for timer in [dict(good,instance_id=5),dict(good,deadline_unix=self.deadline+3600),dict(good,state='failed')]:
            with self.subTest(timer=timer):self.reject(remote=FakeRemote(self.deadline,[ready_map()],timer=timer))
        self.args.until=self.iso(self.deadline+3600)
        self.reject()
        self.args.until=self.iso(self.deadline);self.args.expected_instance_id=5
        self.reject()
        self.args.expected_instance_id=ID;self.state['purpose']='other';self.write_state()
        self.reject()
        self.state.update(purpose=PURPOSE,deadline_unix=time.time()-1);self.write_state()
        self.args.until=self.iso(self.state['deadline_unix'])
        self.reject()

    def test_marker_then_lock_gate_then_ownership_and_cleanup(self):
        old=(self.root/'.state/presentation.lock').open('a');fcntl.flock(old,fcntl.LOCK_EX)
        c=self.controller();events=[];before=self.state_bytes()
        def sleeper(seconds):
            events.append('sleep')
            if len(events)==1:  # old controller still owns lock and rental
                data=json.loads(self.marker.read_text())
                self.assertEqual((data['phase'],data['old_pid'],data['new_pid'],data['instance_id']),
                                 ('ready_for_takeover',41766,os.getpid(),ID))
                self.assertEqual(data['validated_timer']['state'],'armed')
                self.assertEqual(data['deadline_unix'],self.deadline)
                self.assertEqual(before,self.state_bytes());self.assertFalse(c.owned);self.assert_untouched()
            else:old.close()
        def adopt():events.append('adopt');self.assertTrue(c.owned)
        with patch.object(c,'check_local_ports'),patch.object(c,'adopt',side_effect=adopt), \
             patch.object(c,'present'),patch.object(c,'backup_remote'),patch('scripts.presentation.time.sleep',sleeper):
            c.run(resume=True)
        self.assertEqual(events[:3],['sleep','sleep','adopt'])
        self.assertEqual(json.loads(self.marker.read_text())['phase'],'taken_over')
        self.assertFalse(self.api.live)  # owner's lifecycle cleanup destroys the rental at the end

    def test_cancel_before_ownership_leaves_old_controller_and_rental(self):
        old=(self.root/'.state/presentation.lock').open('a');fcntl.flock(old,fcntl.LOCK_EX)
        c=self.controller();before=self.state_bytes()
        with patch.object(c,'check_local_ports'),patch.object(c,'adopt') as adopt, \
             patch('scripts.presentation.time.sleep',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):c.run(resume=True)
        adopt.assert_not_called();self.assert_untouched()
        self.assertFalse(self.marker.exists());self.assertEqual(before,self.state_bytes())

    def post_lock_failure(self,patcher):
        c=self.controller();before=self.state_bytes()
        with patch.object(c,'check_local_ports'),patch.object(c,'adopt') as adopt,patch.object(c,'present'), \
             patch.object(c,'backup_remote') as backup,patcher(c):
            with self.assertRaises(Exception):c.run(resume=True)
        adopt.assert_not_called();backup.assert_not_called()
        self.assertFalse(c.owned);self.assertEqual(before,self.state_bytes());self.assert_untouched()
        self.assertFalse(self.marker.exists())
        probe=(self.root/'.state/presentation.lock').open('a')
        fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)  # lock was released after the failed adoption
        probe.close()

    def test_post_lock_timer_failure_never_deletes_or_changes_state(self):
        seen=[]
        def patcher(c):
            real=c.preflight
            def flaky(ports=True):
                seen.append(ports)
                if not ports:raise HandoffRejected('timer no longer armed')
                return real(ports)
            return patch.object(c,'preflight',side_effect=flaky)
        self.post_lock_failure(patcher);self.assertEqual(seen,[True,False])

    def test_state_commit_failure_never_deletes_or_changes_state(self):
        def patcher(c):
            def failing(path,state):
                if Path(path)==c.state_path:raise OSError('disk full')
            return patch('scripts.presentation.save_state',side_effect=failing)
        self.post_lock_failure(patcher)

    def test_scp_updates_follow_ownership_and_precede_trusted_start_services(self):
        remote=FakeRemote(self.deadline,ready=[ready_map(four=True)],active=True);c=self.controller(remote=remote)
        events=[];real=c.take_over
        def take(timer):
            real(timer);events.append(('owned',c.owned,json.loads(c.state_path.read_text())['controller_pid']))
        def runner(command,**kw):
            if command[0]=='scp':events.append(('scp',command))
            elif 'start_services.sh' in (kw.get('input') or ''):events.append(('start',))
            return remote(command,**kw)
        c.runner=runner
        with patch.object(c,'take_over',side_effect=take),patch.object(c,'check_local_ports'),patch.object(c,'present'), \
             patch.object(c,'backup_remote'),patch('scripts.presentation.time.sleep'),patch('scripts.presentation.sys.platform','linux'):
            c.run(resume=True)
        self.assertEqual([e[0] for e in events],['owned']+['scp']*5+['start'])
        self.assertEqual(events[0][1:],(True,os.getpid()))
        self.assertEqual([e[1][-1] for e in events if e[0]=='scp'],[f'root@host:/workspace/persona-lab/{t}' for t in
            ['lab/server.py','lab/static/app.js','lab/static/index.html','lab/static/style.css','scripts/start_services.sh']])
        for e in events[1:6]:
            text=' '.join(e[1]);self.assertIn('StrictHostKeyChecking=yes',text);self.assertNotIn('.env',text);self.assertNotIn('.state',text)

    def stop(self,c):
        with patch('scripts.presentation.Presentation',return_value=c),patch('scripts.presentation.read_key',return_value='k'), \
             patch('scripts.presentation.sys.argv',['p','--env-file','e','--stop']),patch.object(c,'backup_remote'), \
             patch('scripts.presentation.time.sleep'):
            return presentation_main()

    def test_unowned_destroy_is_inert_and_stop_rejects_invalid_state_before_ownership(self):
        c=self.controller();c.state=dict(self.state);c.owned=False
        c.destroy();self.assert_untouched()
        good=dict(self.state)
        for bad in [{'label':''},{'instance_id':'60502794'},{'purpose':'other'}]:
            self.state=dict(good,**bad);self.write_state();c=self.controller()
            with self.assertRaises(SystemExit):self.stop(c)
            self.assert_untouched()

    def test_direct_stop_after_ambiguous_create_destroys_labelled_rental_only(self):
        self.state={'purpose':PURPOSE,'label':LABEL,'phase':'creating'};self.write_state()  # no instance ID yet
        c=self.controller(api=ResumeFakeAPI(label='someone-else'))
        self.assertEqual(self.stop(c),0)  # a differently labelled instance is never deleted
        self.assert_untouched()
        self.state={'purpose':PURPOSE,'label':LABEL,'phase':'creating'};self.write_state()
        c=self.controller();self.assertEqual(self.stop(c),0)
        self.assertEqual([call[0] for call in self.api.calls if call[2]=='DELETE'],[f'/api/v0/instances/{ID}/'])

    def test_direct_stop_with_known_absent_instance_confirms_absence(self):
        self.state['phase']='rented';self.write_state();c=self.controller(api=ResumeFakeAPI(id=5))
        c.api.live=[];self.assertEqual(self.stop(c),0)
        self.assertFalse([call for call in c.api.calls if call[2]=='DELETE'])
        self.assertEqual(json.loads(c.state_path.read_text())['phase'],'destroyed')

    def snapshot(self,shards=('a.safetensors','b.safetensors'),skip=(),broken=()):
        base=self.root/'hf/hub/models--Qwen--Qwen3.5-4B';snap=base/'snapshots/r1'
        snap.mkdir(parents=True);(base/'blobs').mkdir()
        for name in ('config.json','tokenizer_config.json','tokenizer.json'):(snap/name).write_text('{}')
        (snap/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':shards[0],'y':shards[1],'z':shards[0]}}))
        for index,name in enumerate(shards):
            if name in skip:continue
            if name in broken:(snap/name).symlink_to('../../blobs/missing');continue
            (base/'blobs'/f'blob{index}').write_bytes(b'weights');(snap/name).symlink_to(f'../../blobs/blob{index}')
        return base,snap
    def ready(self):return snapshot_ready(str(self.root/'hf'),'Qwen/Qwen3.5-4B','r1')

    def test_complete_4b_shards_detection(self):
        self.snapshot();self.assertTrue(self.ready())
    def test_missing_broken_incomplete_empty_shard_or_tokenizer_is_not_ready(self):
        self.snapshot(skip=('b.safetensors',));self.assertFalse(self.ready())
        self.tearDown();self.setUp();self.snapshot(broken=('b.safetensors',));self.assertFalse(self.ready())
        self.tearDown();self.setUp();base,_=self.snapshot();(base/'blobs/c.incomplete').write_text('x');self.assertFalse(self.ready())
        self.tearDown();self.setUp();base,_=self.snapshot();(base/'blobs/blob1').write_bytes(b'');self.assertFalse(self.ready())
        self.tearDown();self.setUp();_,snap=self.snapshot();(snap/'tokenizer_config.json').unlink();self.assertFalse(self.ready())

    def test_active_prefetch_is_never_duplicated_and_guard_precedes_launch(self):
        c=self.controller();c.state=dict(self.state);c.ssh=['ssh'];c.destination='root@host'
        with patch.object(c,'update',side_effect=AssertionError('no launch expected')):
            self.assertFalse(c.ensure_prefetch(MODELS))
        script=self.remote.calls[0][2]
        self.assertLess(script.index('pgrep'),script.index('nohup'))
        self.assertNotIn('kill ',script)
    def test_single_trusted_launch_when_none_active(self):
        remote=FakeRemote(self.deadline,active=False);c=self.controller(remote=remote)
        c.state=dict(self.state);c.ssh=['ssh'];c.destination='root@host'
        with patch.object(c,'update'):self.assertTrue(c.ensure_prefetch(MODELS[1:]))
        script=remote.calls[0][2]
        self.assertEqual(script.count('nohup'),1)
        self.assertIn('Qwen/Qwen3-8B',script);self.assertNotIn('Qwen3.5-4B',script)

    def test_services_start_when_4b_ready_without_waiting_for_larger_models(self):
        remote=FakeRemote(self.deadline,ready=[ready_map(),ready_map(four=True)],active=True)
        c=self.controller(remote=remote)
        with patch.object(c,'check_local_ports'),patch.object(c,'present') as present,patch.object(c,'backup_remote'), \
             patch('scripts.presentation.time.sleep'),patch('scripts.presentation.sys.platform','linux'):
            c.run(resume=True)
        self.assertEqual(len(remote.scripts('start_services.sh')),1);present.assert_called_once()
        self.assertEqual(c.pending,MODELS[1:])  # 8B/27B continue in the existing background downloader
        self.assertNotIn('prefetch_launched_at_unix',c.state)  # downloader reported ACTIVE: no second launch
        self.assertFalse([call for call in self.api.calls if '/asks/' in call[0]])
        saved=json.loads((self.root/'.state/presentation.json').read_text())
        self.assertEqual(saved['controller_pid'],os.getpid());self.assertEqual(saved['deadline_unix'],self.deadline)

    def test_setup_timeout_follows_deadline_not_one_hour(self):
        c=self.controller();c.bundle=self.root/'bundle';c.ssh=['ssh'];c.destination='root@host'
        c.state=dict(self.state);seen=[]
        c.runner=lambda command,**kw:seen.append(kw['timeout'])
        c.deploy()
        self.assertGreater(seen[-1],3600);self.assertLessEqual(seen[-1],4*3600)


if __name__=='__main__':unittest.main()

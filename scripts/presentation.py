#!/usr/bin/env python3
"""Rent, restore, open a bounded presentation, then destroy its own Vast rental.

Run on the Mac: python3 scripts/presentation.py --env-file /private/path/keys.env
Ctrl+C, setup failure and the wall-clock deadline destroy this controller's rental.
The local watchdog cannot run while the Mac sleeps or this process is killed.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import webbrowser

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import vast_gpu

ROOT=Path(__file__).resolve().parents[1]
PURPOSE='qwen-persona-presentation-v1'
MODELS=['Qwen/Qwen3.5-4B','Qwen/Qwen3-8B','Qwen/Qwen3.8-27B']


def session_deadline(duration_hours=2,until=None,now=None):
    now=time.time() if now is None else now
    if until is None:
        if not 0<duration_hours<=2:raise ValueError('Duration must be positive and at most 2 hours without --until')
        return now+duration_hours*3600
    parsed=datetime.datetime.fromisoformat(until.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('--until must include a timezone, such as 2026-11-13T21:00:00Z')
    deadline=parsed.timestamp()
    if not 0<deadline-now<=12*3600:
        raise ValueError('--until must be in the future and no more than 12 hours away')
    return deadline


def read_key(path):
    for line in Path(path).expanduser().read_text().splitlines():
        key,separator,value=line.partition('=')
        if separator and key.strip().removeprefix('export ')=='VAST_AI_API_KEY':
            value=value.strip()
            if len(value)>1 and value[0]==value[-1] and value[0] in '\"\'':value=value[1:-1]
            if value:return value
    raise ValueError('VAST_AI_API_KEY was not found in the private env file')


def registered_public_key(document,public_key):
    identity=tuple(public_key.split()[:2])
    if len(identity)!=2:return False
    if isinstance(document,str):return tuple(document.split()[:2])==identity
    if isinstance(document,dict):return any(registered_public_key(value,public_key) for value in document.values())
    if isinstance(document,list):return any(registered_public_key(value,public_key) for value in document)
    return False


def duplicate_key_response(error):
    text=(error.error+' '+error.message).lower()
    return ('key' in text and any(phrase in text for phrase in
        ['already attached','already exists','already added','already registered','already associated','duplicate']))


def clean_environment():
    # Never forward provider/HF secrets through a subprocess or SSH SendEnv rule.
    return {key:value for key,value in os.environ.items()
            if key in {'PATH','HOME','USER','LOGNAME','SSH_AUTH_SOCK','TMPDIR','LANG','LC_ALL'}}


def save_state(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as handle:json.dump(state,handle,indent=2)
    os.replace(temporary,path)


def package_workspace(root,destination,backup):
    """Whitelist project trees; reject symlinks, credential/runtime dirs and DB files."""
    allowed={'lab','scripts','tests','docs','data','vendor'}
    topfiles={'README.md','pyproject.toml','uv.lock','.gitignore'}
    denied={'.git','.venv','.state','.aws','.ssh','__pycache__','outputs'}
    def forbidden(parts):
        return any(part in denied or part.startswith('.env') or
                   any(fnmatch.fnmatch(part,pattern) for pattern in ('*.sqlite*','*.log','*.pem','*.key','*.token'))
                   for part in parts)
    with tempfile.TemporaryDirectory(prefix='persona-db-') as temporary:
        snapshot=Path(temporary)/'lab.sqlite3'
        origin=sqlite3.connect(Path(backup).resolve().as_uri()+'?mode=ro',uri=True)
        target=sqlite3.connect(snapshot)
        try:origin.backup(target)
        finally:target.close();origin.close()
        with tarfile.open(destination,'w:gz') as archive:
            for directory,folders,files in os.walk(root):
                relative=Path(directory).relative_to(root)
                folders[:]=sorted(name for name in folders if not forbidden(relative.parts+(name,)) and
                    not (Path(directory)/name).is_symlink() and (relative.parts or name in allowed))
                for name in sorted(files):
                    path=Path(directory)/name;rel=path.relative_to(root)
                    if forbidden(rel.parts) or path.is_symlink() or (len(rel.parts)==1 and name not in topfiles):continue
                    archive.add(path,arcname=str(rel),recursive=False)
            archive.add(snapshot,arcname='data/lab.sqlite3')
    os.chmod(destination,0o600)


def is_lab_backup(path):
    try:
        con=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
        try:
            recipes={row[1] for row in con.execute('PRAGMA table_info(recipes)')}
            jobs={row[1] for row in con.execute('PRAGMA table_info(jobs)')}
            return {'id','neurons_json','model','source'}<=recipes and {'id','recipe_id','result_json','status'}<=jobs
        finally:con.close()
    except sqlite3.DatabaseError:return False


class Presentation:
    def __init__(self,args,root=ROOT,api=None,offer_provider=None,runner=None):
        self.args=args;self.root=Path(root);self.api=api or vast_gpu.request
        self.offer_provider=offer_provider or vast_gpu.offers
        self.runner=runner or subprocess.run
        self.state_path=self.root/'.state/presentation.json'
        self.state={};self.tunnel=None;self.bundle=None;self.log=None;self.awake=None;self.backup_done=False

    def update(self,**values):
        self.state.update(values);save_state(self.state_path,self.state)

    def remaining(self,maximum):
        available=self.state.get('deadline_unix',time.time()+maximum)-time.time()
        if available<=0:raise TimeoutError('Presentation deadline reached')
        return min(maximum,available)

    def command(self,command,maximum=1800):
        return self.runner(command,check=True,timeout=self.remaining(maximum),env=clean_environment(),
                           stdout=self.log,stderr=subprocess.STDOUT)

    def prepare(self):
        for tool in ['ssh','scp','bash']:
            if shutil.which(tool) is None:raise ValueError(f'Required local command unavailable: {tool}')
        if sys.platform=='darwin' and shutil.which('caffeinate'):
            self.awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],
                env=clean_environment(),start_new_session=True)
        for port in [8787,8791]:
            with socket.socket() as probe:
                try:probe.bind(('127.0.0.1',port))
                except OSError:raise ValueError(f'Local port {port} is occupied; close your old tunnel before renting')
        self.public_key=Path(self.args.key).expanduser().read_text().strip()
        if not self.public_key.startswith('ssh-') or 'PRIVATE KEY' in self.public_key:
            raise ValueError('--key must name an SSH public key')
        if self.state_path.exists():
            previous=json.loads(self.state_path.read_text())
            if previous.get('phase')!='destroyed':
                raise ValueError('Previous presentation state is unresolved; use --stop before renting again')
        backups=[path for path in (self.root/'outputs').rglob('*backup*.sqlite3') if is_lab_backup(path)]
        backup=Path(self.args.backup).expanduser() if self.args.backup else max(backups,key=lambda p:p.stat().st_mtime,default=None)
        if backup is None or not backup.is_file() or not is_lab_backup(backup):
            raise ValueError('A validated local SQLite backup is required; specify --backup PATH')
        directory=self.root/'.state';directory.mkdir(exist_ok=True)
        self.bundle=directory/'presentation-bundle.tar.gz'
        package_workspace(self.root,self.bundle,backup)
        (self.root/'outputs').mkdir(exist_ok=True)
        self.log=(self.root/'outputs/presentation-setup.log').open('a',buffering=1)
        print(f'Restoring database backup: {backup}',flush=True)

    def rent(self):
        created=time.time()
        deadline=session_deadline(self.args.duration_hours,getattr(self.args,'until',None),created)
        hours=(deadline-created)/3600
        maximum_total=getattr(self.args,'max_total',23)
        candidates=[]
        for offer in self.offer_provider():
            rate=float(offer['dph_total'])+float(offer.get('storage_cost',0))*180/720
            if ('RTX PRO 6000' in offer.get('gpu_name','') and float(offer.get('gpu_ram',0))>=95000
                    and float(offer.get('cuda_max_good',0))>=13.0
                    and rate*hours<=maximum_total
                    and rate<=min(self.args.max_hourly,2) and offer.get('verified') is not False):
                candidates.append((rate,offer))
        if not candidates:raise ValueError('No verified 96GB RTX PRO 6000 offer meets the hourly cap and total estimate budget')
        rejected=[]
        for rate,offer in sorted(candidates,key=lambda pair:pair[0])[:5]:
            label='persona-demo-'+uuid.uuid4().hex[:12]
            self.update(purpose=PURPOSE,label=label,phase='creating',offer_id=offer['id'],
                        estimated_hourly_usd=rate,created_at_unix=created,
                        deadline_unix=deadline,deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
                        controller_pid=os.getpid(),estimated_total_usd=rate*hours,max_total_usd=maximum_total,
                        provider_deadline_configured=False,creation_rejections=list(rejected))
            try:
                result=self.api(f"/api/v0/asks/{offer['id']}/",{'client_id':'me',
                    'image':'pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime','disk':180,
                    'runtype':'ssh_direc ssh_proxy','label':label,'cancel_unavail':True},'PUT')
                if not result.get('success'):
                    raise vast_gpu.ProviderAPIError(400,result.get('error'),result.get('msg'))
            except vast_gpu.ProviderAPIError as error:
                if not vast_gpu.unavailable_offer(error):raise
                rejection={'offer_id':offer['id'],'error':error.error,'msg':error.message}
                rejected.append(rejection);self.update(creation_rejections=list(rejected))
                print(f'Stale offer {offer["id"]} rejected: {error}; trying next eligible offer.',flush=True)
                continue
            # A timeout/connection failure is intentionally NOT retried. The
            # outer finally reconciles this unique label and cleans up any
            # accepted rental before another controller may be launched.
            break
        else:raise RuntimeError('All selected offers were explicitly unavailable; no replacement rental was created')
        # Persist before attaching a key, transferring files or performing any setup.
        instance=int(result['new_contract']);self.update(instance_id=instance,phase='rented')
        try:registered=registered_public_key(self.api('/api/v0/ssh/'),self.public_key)
        except vast_gpu.ProviderAPIError as error:
            registered=False
            print(f'Account SSH-key lookup unavailable: {error}',flush=True)
        if registered:
            self.update(ssh_key_status='account_registered_authentication_unverified')
            print('Exact public key is already registered with Vast; verifying real SSH authentication next.',flush=True)
        else:
            try:
                attached=self.api(f'/api/v0/instances/{instance}/ssh/',{'ssh_key':self.public_key},'POST')
                if attached.get('success') is False:
                    raise vast_gpu.ProviderAPIError(200,attached.get('error'),attached.get('msg'))
            except vast_gpu.ProviderAPIError as error:
                if not duplicate_key_response(error):raise
                self.update(ssh_key_status='duplicate_key_response_authentication_unverified',
                            ssh_key_message=error.message,ssh_key_error=error.error)
                print(f'Existing-key response: {error}. Verifying real SSH authentication next.',flush=True)
        print(f'Rented instance {instance}; conservative estimate ${rate:.3f}/hour.',flush=True)
        print(f"Local destruction deadline: {self.state['deadline_utc']}; estimated session cost ${rate*hours:.2f}. Keep this Mac awake.",flush=True)

    def wait_ssh(self):
        until=time.time()+self.remaining(600)
        known=self.root/f".state/presentation-known-hosts-{self.state['instance_id']}"
        while time.time()<until:
            instances=self.api('/api/v1/instances/?owner=me').get('instances',[])
            instance=next((row for row in instances if row['id']==self.state['instance_id']),None)
            if instance and instance.get('actual_status')=='running' and instance.get('ssh_host') and instance.get('ssh_port'):
                host,port=instance['ssh_host'],int(instance['ssh_port'])
                self.ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
                    '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={known}','-p',str(port)]
                private=Path(self.args.key).expanduser().with_suffix('')
                if private.is_file():self.ssh+=['-i',str(private),'-o','IdentitiesOnly=yes']
                try:
                    self.command(self.ssh+[f'root@{host}','true'],maximum=15)
                except subprocess.SubprocessError:pass
                else:
                    self.destination=f'root@{host}'
                    self.update(phase='ssh_ready',ssh_host=host,ssh_port=port,known_hosts=str(known),ssh_authentication_verified=True)
                    # Only the first key was accepted; every subsequent command is strict.
                    self.ssh[self.ssh.index('StrictHostKeyChecking=accept-new')]='StrictHostKeyChecking=yes'
                    return
            time.sleep(min(5,self.remaining(5)))
        raise TimeoutError('SSH did not become ready within ten minutes')

    def deploy(self):
        self.update(phase='deploying')
        scp=['scp','-P',str(self.state['ssh_port']),'-o','BatchMode=yes',
             '-o','StrictHostKeyChecking=yes','-o',f"UserKnownHostsFile={self.state['known_hosts']}"]
        private=Path(self.args.key).expanduser().with_suffix('')
        if private.is_file():scp+=['-i',str(private),'-o','IdentitiesOnly=yes']
        self.command(scp+[str(self.bundle),self.destination+':/tmp/persona-presentation.tar.gz'])
        timer_check=('import json,time; from pathlib import Path; p=Path(".state/remote-deadline.json"); '
                     'deadline=time.monotonic()+20\n'
                     'while time.monotonic()<deadline:\n'
                     ' try:\n'
                     '  state=json.loads(p.read_text())\n'
                     f'  if state.get("state")=="armed" and state.get("instance_id")=={self.state["instance_id"]}: break\n'
                     ' except (OSError,ValueError): pass\n'
                     ' time.sleep(.5)\n'
                     'else: raise SystemExit("Remote deadline watchdog did not arm; refusing setup")')
        unpack=('set -e; mkdir -p /workspace/persona-lab; '
                'tar -xzf /tmp/persona-presentation.tar.gz -C /workspace/persona-lab; '
                'cd /workspace/persona-lab; mkdir -p outputs .state; '
                'nohup python3 scripts/remote_deadline.py '
                f'--instance-id {self.state["instance_id"]} --until '+shlex.quote(self.state['deadline_utc'])+
                ' --status-file .state/remote-deadline.json > outputs/remote-deadline.log 2>&1 < /dev/null & '
                'python3 -c '+shlex.quote(timer_check))
        self.command(self.ssh+[self.destination,unpack],maximum=45)
        self.update(provider_deadline_configured=True,
                    deadline_mechanism='provider_container_self_delete_watchdog',remote_deadline_status='armed')
        models=list(dict.fromkeys(['Qwen/Qwen3.5-4B']+self.args.models))
        download=('import json; from huggingface_hub import snapshot_download; '
                  'manifest=json.load(open("data/model-cache-manifest.json"))["models"]; '+
                  '; '.join(f'snapshot_download({model!r},revision=manifest[{model!r}]["revision"])' for model in models))
        setup=('set -e; cd /workspace/persona-lab; export HF_HOME=/workspace/hf-cache; '
               'SKIP_SMOKE=1 bash scripts/bootstrap_gpu.sh; ')
        if self.args.fast_kernels:
            setup+='apt-get update; apt-get install -y --no-install-recommends gcc g++; .venv/bin/python -m pip install flash-linear-attention==0.6.0 fla-core==0.6.0; '
        setup+='.venv/bin/python -c '+shlex.quote(download)+'; bash scripts/start_services.sh'
        self.command(self.ssh+[self.destination,setup],maximum=3600)
        self.update(phase='ready')

    def present(self):
        self.tunnel=subprocess.Popen(['bash',str(self.root/'scripts/tunnel.sh'),self.state['ssh_host'],
            str(self.state['ssh_port']),self.state['known_hosts']],env=clean_environment(),start_new_session=True)
        print('UI: http://127.0.0.1:8787   Original 3D: http://127.0.0.1:8791',flush=True)
        print('Ctrl+C destroys this rental. Setup log: outputs/presentation-setup.log',flush=True)
        if self.args.open_browser:webbrowser.open('http://127.0.0.1:8787')
        while self.remaining(5)>0:
            saved=json.loads(self.state_path.read_text())
            if saved.get('phase')=='destroyed':return
            if self.tunnel.poll() is not None:raise RuntimeError('Tunnel controller exited')
            if not self.backup_done and self.state['deadline_unix']-time.time()<=130:
                self.backup_remote()
            time.sleep(self.remaining(5))

    def backup_remote(self):
        """Bounded best effort; never extend the rental deadline to retrieve evidence."""
        if self.backup_done:return
        self.backup_done=True
        if not self.state.get('ssh_host') or not self.state.get('known_hosts'):return
        available=min(110,self.state.get('deadline_unix',time.time()+120)-time.time()-10)
        if available<5:
            self.update(backup_status='skipped_deadline_too_close');return
        end=time.monotonic()+available
        directory=self.root/f"outputs/presentation-{self.state['instance_id']}"
        directory.mkdir(parents=True,exist_ok=True)
        destination=f"root@{self.state['ssh_host']}"
        options=['-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
                 '-o',f"UserKnownHostsFile={self.state['known_hosts']}"]
        private=Path(self.args.key).expanduser().with_suffix('')
        if private.is_file():options+=['-i',str(private),'-o','IdentitiesOnly=yes']
        remote='/tmp/persona-presentation-evidence.tar.gz'
        paths=['outputs/presentation-backup.sqlite3','outputs/presentation-results.jsonl',
               'outputs/presentation-result-report.json']
        command=('set -e; cd /workspace/persona-lab; .venv/bin/python scripts/export_results.py '
                 '--db data/lab.sqlite3 --backup '+paths[0]+' --output '+paths[1]+' --report '+paths[2]+
                 '; tar -czf '+remote+' '+' '.join(paths))
        def bounded(arguments):
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Evidence backup time limit reached')
            return self.runner(arguments,check=True,timeout=remaining,env=clean_environment(),
                               stdout=self.log,stderr=subprocess.STDOUT)
        try:
            print('Saving new remote database and output evidence before teardown...',flush=True)
            bounded(['ssh','-p',str(self.state['ssh_port'])]+options+[destination,command])
            archive_path=directory/'evidence.tar.gz'
            bounded(['scp','-P',str(self.state['ssh_port'])]+options+[destination+':'+remote,str(archive_path)])
            with tarfile.open(archive_path) as archive:
                for name in paths:
                    member=archive.getmember(name)
                    if not member.isfile():raise ValueError('Unexpected evidence archive member')
                    target=directory/Path(name).name
                    with archive.extractfile(member) as source,target.open('wb') as output:
                        shutil.copyfileobj(source,output)
            if not is_lab_backup(directory/'presentation-backup.sqlite3'):
                raise ValueError('Downloaded backup is not a lab database')
            self.update(backup_status='saved',backup_directory=str(directory),backup_completed_at_unix=time.time())
            print(f'New evidence saved to {directory}',flush=True)
        except Exception as error:
            self.update(backup_status='failed',backup_error=type(error).__name__)
            print('Evidence backup failed or timed out; rental destruction will still proceed.',file=sys.stderr,flush=True)

    def destroy(self):
        if self.state.get('purpose')!=PURPOSE:return
        instance=self.state.get('instance_id')
        for attempt in range(3):
            try:
                live=self.api('/api/v1/instances/?owner=me').get('instances',[])
                owned=[row for row in live if (row['id']==instance if instance is not None
                                             else row.get('label')==self.state.get('label'))]
                if any(row.get('label') not in (None,self.state.get('label')) for row in owned):
                    raise RuntimeError('Recorded instance label does not match this presentation')
                if owned:
                    for row in owned:
                        result=self.api(f"/api/v0/instances/{row['id']}/",method='DELETE')
                        if not result.get('success'):raise RuntimeError('Provider did not confirm destruction')
                    # Recheck on the next pass rather than trusting request acceptance.
                    if attempt<2:time.sleep(2);continue
                    raise RuntimeError('Destruction requested; final provider absence was not confirmed')
                self.update(phase='destroyed',destroyed_at_unix=time.time())
                print('Provider confirms this presentation rental is absent; no rented disk remains.',flush=True)
                return
            except Exception as error:
                if attempt==2:
                    self.update(phase='cleanup_failed',cleanup_error=type(error).__name__)
                    raise RuntimeError('Cleanup could not be confirmed. Use --stop or destroy the recorded instance in Vast immediately.') from error
                time.sleep(2)

    def run(self):
        try:
            self.prepare();self.rent();self.wait_ssh();self.deploy();self.present()
        finally:
            # A second Ctrl+C or a vanished SSH child must not skip paid-resource cleanup.
            saved={sig:signal.signal(sig,signal.SIG_IGN) for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]}
            try:
                try:
                    if self.tunnel is not None:
                        try:
                            self.tunnel.terminate();self.tunnel.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(self.tunnel.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                finally:
                    try:self.backup_remote()
                    finally:self.destroy()
            finally:
                if self.log is not None:self.log.close()
                if self.bundle is not None:self.bundle.unlink(missing_ok=True)
                if self.awake is not None:
                    try:self.awake.terminate();self.awake.wait(timeout=3)
                    except ProcessLookupError:pass
                    except subprocess.TimeoutExpired:self.awake.kill()
                for sig,handler in saved.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',required=True)
    parser.add_argument('--key',default='~/.ssh/id_ed25519.pub')
    parser.add_argument('--backup',help='Consistent validated backup; default newest outputs/**/*backup*.sqlite3')
    parser.add_argument('--max-hourly',type=float,default=2)
    timing=parser.add_mutually_exclusive_group()
    timing.add_argument('--duration-hours',type=float,default=2)
    timing.add_argument('--until',help='Explicit timezone-aware destruction timestamp, at most 12 hours away')
    parser.add_argument('--max-total',type=float,default=23,help='Maximum estimated compute/storage session cost in USD')
    parser.add_argument('--models','--prefetch-models',dest='models',nargs='+',choices=MODELS,default=['Qwen/Qwen3.5-4B'])
    parser.add_argument('--fast-kernels',action='store_true')
    parser.add_argument('--open-browser',action='store_true')
    parser.add_argument('--stop',action='store_true',help='Destroy only the rental recorded by this presentation controller')
    args=parser.parse_args()
    if not 0<args.max_hourly<=2 or not 0<args.max_total:
        parser.error('Hourly limit must be positive and at most 2; total estimate budget must be positive')
    try:session_deadline(args.duration_hours,args.until)
    except ValueError as error:parser.error(str(error))
    os.environ['VAST_AI_API_KEY']=read_key(args.env_file)
    controller=Presentation(args)
    if args.stop:
        controller.state=json.loads(controller.state_path.read_text())
        if controller.state.get('purpose')!=PURPOSE:raise SystemExit('Unrecognized state; refusing to destroy any rental')
        try:controller.backup_remote()
        finally:controller.destroy()
        return 0
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    # Respect nohup's inherited SIGHUP ignore policy for a detached controller.
    if signal.getsignal(signal.SIGHUP)!=signal.SIG_IGN:signal.signal(signal.SIGHUP,interrupt)
    (ROOT/'.state').mkdir(exist_ok=True)
    lock=(ROOT/'.state/presentation.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another presentation controller is active; use --stop to destroy its rental')
    try:controller.run()
    except KeyboardInterrupt:print('Presentation stopped; cleanup completed.',flush=True)
    except TimeoutError:print('Presentation deadline/setup timeout reached; cleanup completed.',flush=True)
    except Exception as error:
        print(f'Presentation failed: {error}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

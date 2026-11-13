#!/usr/bin/env python3
"""Rent, restore, open a bounded presentation, then destroy its own Vast rental.

Run on the Mac: python3 scripts/presentation.py --env-file /private/path/keys.env
Ctrl+C, setup failure and the wall-clock deadline destroy this controller's rental.
The local watchdog cannot run while the Mac sleeps or this process is killed.
"""
from __future__ import annotations
import argparse
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


def read_key(path):
    for line in Path(path).expanduser().read_text().splitlines():
        key,separator,value=line.partition('=')
        if separator and key.strip().removeprefix('export ')=='VAST_AI_API_KEY':
            value=value.strip()
            if len(value)>1 and value[0]==value[-1] and value[0] in '\"\'':value=value[1:-1]
            if value:return value
    raise ValueError('VAST_AI_API_KEY was not found in the private env file')


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
        self.state={};self.tunnel=None;self.bundle=None;self.log=None;self.awake=None

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
        candidates=[]
        for offer in self.offer_provider():
            rate=float(offer['dph_total'])+float(offer.get('storage_cost',0))*180/720
            if ('RTX PRO 6000' in offer.get('gpu_name','') and float(offer.get('gpu_ram',0))>=95000
                    and float(offer.get('cuda_max_good',0))>=13.0
                    and rate<=min(self.args.max_hourly,2) and offer.get('verified') is not False):
                candidates.append((rate,offer))
        if not candidates:raise ValueError('No verified 96GB RTX PRO 6000 offer meets the hourly cap')
        rate,offer=min(candidates,key=lambda pair:pair[0])
        label='persona-demo-'+uuid.uuid4().hex[:12]
        self.update(purpose=PURPOSE,label=label,phase='creating',offer_id=offer['id'],
                    estimated_hourly_usd=rate,created_at_unix=time.time(),
                    deadline_unix=time.time()+self.args.duration_hours*3600,
                    provider_deadline_configured=False)
        result=self.api(f"/api/v0/asks/{offer['id']}/",{'client_id':'me',
            'image':'pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime','disk':180,
            'runtype':'ssh_direc ssh_proxy','label':label,'cancel_unavail':True},'PUT')
        if not result.get('success'):raise RuntimeError('Provider declined the selected offer')
        # Persist before attaching a key, transferring files or performing any setup.
        instance=int(result['new_contract']);self.update(instance_id=instance,phase='rented')
        attached=self.api(f'/api/v0/instances/{instance}/ssh/',{'ssh_key':self.public_key},'POST')
        if attached.get('success') is False:raise RuntimeError('Provider declined SSH key attachment')
        print(f'Rented instance {instance}; conservative estimate ${rate:.3f}/hour.',flush=True)
        print('Automatic destruction at the 2-hour-or-shorter local deadline; keep this Mac awake.',flush=True)

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
                    self.update(phase='ssh_ready',ssh_host=host,ssh_port=port,known_hosts=str(known))
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
        models=list(dict.fromkeys(['Qwen/Qwen3.5-4B']+self.args.models))
        download='from huggingface_hub import snapshot_download; '+'; '.join(f'snapshot_download({model!r})' for model in models)
        setup=('set -e; mkdir -p /workspace/persona-lab; '
               'tar -xzf /tmp/persona-presentation.tar.gz -C /workspace/persona-lab; '
               'cd /workspace/persona-lab; export HF_HOME=/workspace/hf-cache; '
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
            time.sleep(self.remaining(5))

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
    parser.add_argument('--duration-hours',type=float,default=2)
    parser.add_argument('--models','--prefetch-models',dest='models',nargs='+',choices=MODELS,default=['Qwen/Qwen3.5-4B'])
    parser.add_argument('--fast-kernels',action='store_true')
    parser.add_argument('--open-browser',action='store_true')
    parser.add_argument('--stop',action='store_true',help='Destroy only the rental recorded by this presentation controller')
    args=parser.parse_args()
    if not 0<args.max_hourly<=2 or not 0<args.duration_hours<=2:
        parser.error('Hourly limit and session hours must be positive and at most 2')
    os.environ['VAST_AI_API_KEY']=read_key(args.env_file)
    controller=Presentation(args)
    if args.stop:
        controller.state=json.loads(controller.state_path.read_text())
        if controller.state.get('purpose')!=PURPOSE:raise SystemExit('Unrecognized state; refusing to destroy any rental')
        controller.destroy();return 0
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGHUP,interrupt)
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

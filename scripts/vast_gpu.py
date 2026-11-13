"""Vast REST helper. Secrets are read from environment; rates capped at $2/hour.

VAST_AI_API_KEY=... python scripts/vast_gpu.py offers
python scripts/vast_gpu.py rent OFFER_ID --key ~/.ssh/id_ed25519.pub
python scripts/vast_gpu.py status
python scripts/vast_gpu.py destroy INSTANCE_ID
"""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://console.vast.ai'

class ProviderAPIError(RuntimeError):
    def __init__(self,status,error=None,message=None):
        self.status_code=status
        def safe(value):
            value=str(value or '')[:500]
            token=os.environ.get('VAST_AI_API_KEY')
            if token:value=value.replace(token,'[redacted]')
            return re.sub(r'\b(?:[a-fA-F0-9]{40,}|(?:hf_|sk_|ghp_)[A-Za-z0-9_]{20,})\b','[redacted]',value)
        self.error=safe(error);self.message=safe(message)
        super().__init__(f'Vast API HTTP {status}: {self.error or "request_rejected"}: {self.message or "No safe error detail supplied"}')

def unavailable_offer(error):
    """Retry only a definite stale offer, never a generic bad payload or ambiguous failure."""
    if not isinstance(error,ProviderAPIError) or error.status_code not in {400,404,410}:return False
    text=(error.error+' '+error.message).lower()
    return (any(value in text for value in ['no_such_ask','no such ask','offer_not_available',
        'already rented','offer is not available','offer unavailable','offer is no longer'])
        or ('instance type by id' in text and 'not available' in text))

def request(path, payload=None, method='GET'):
    token = os.environ.get('VAST_AI_API_KEY')
    if not token: raise SystemExit('Set VAST_AI_API_KEY; never pass credentials as arguments.')
    req = urllib.request.Request(BASE+path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as res:return json.load(res)
    except urllib.error.HTTPError as error:
        try:body=json.loads(error.read(65536).decode('utf-8',errors='replace'))
        except (ValueError,OSError):body={}
        if not isinstance(body,dict):body={}
        raise ProviderAPIError(error.code,body.get('error'),body.get('msg')) from None

def offers():
    q = {'verified':{'eq':True}, 'rentable':{'eq':True}, 'rented':{'eq':False},
         'num_gpus':{'eq':1}, 'gpu_ram':{'gte':95000}, 'disk_space':{'gte':180},
         'order':[['dph_total','asc']], 'limit':100}
    return request('/api/v0/bundles/?'+urllib.parse.urlencode({'q':json.dumps(q)}))['offers']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['offers','rent','status','start','stop','destroy'])
    p.add_argument('id',nargs='?',type=int)
    p.add_argument('--key',default='~/.ssh/id_ed25519.pub')
    p.add_argument('--env-file',help='Read only VAST_AI_API_KEY from an existing env file; never execute its contents')
    a=p.parse_args()
    if a.env_file:
        for line in Path(a.env_file).expanduser().read_text().splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip().removeprefix('export ') == 'VAST_AI_API_KEY':
                value = value.strip()
                if len(value)>1 and value[0] == value[-1] and value[0] in '\"\'':
                    value = value[1:-1]
                os.environ['VAST_AI_API_KEY'] = value
                break
        else: raise SystemExit('VAST_AI_API_KEY not found in env file')
    if a.action=='offers':
        print(json.dumps([{k:o.get(k) for k in ['id','gpu_name','dph_total','storage_cost','reliability','disk_space']} for o in offers() if 'RTX PRO 6000' in o.get('gpu_name','') and o['dph_total']<2],indent=2))
    elif a.action=='rent':
        key=Path(a.key).expanduser().read_text().strip()
        if not key.startswith('ssh-') or 'PRIVATE KEY' in key:
            raise SystemExit('Provide an SSH public key')
        o=next((o for o in offers() if o['id']==a.id),None)
        if not o or 'RTX PRO 6000' not in o.get('gpu_name',''): raise SystemExit('Offer unavailable or wrong GPU.')
        # Vast storage_cost is dollars per GB per month. Conservatively add full disk charge.
        rate=o['dph_total']+float(o.get('storage_cost',0))*180/720
        if rate>2: raise SystemExit(f'Rejected: conservative hourly estimate ${rate:.3f} exceeds $2.')
        result=request(f'/api/v0/asks/{a.id}/',{'client_id':'me','image':'pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime','disk':180,'runtype':'ssh_direc ssh_proxy','label':'qwen-persona-lab','cancel_unavail':True},'PUT')
        if not result.get('success'): raise SystemExit('Rental rejected: '+str(result.get('msg','unknown error')))
        instance=result['new_contract']
        # Record the contract before key attachment, so a provider-side failure
        # cannot leave a paid rental without a locally recorded identifier.
        Path('.state').mkdir(exist_ok=True)
        Path('.state/rental.json').write_text(json.dumps({'instance_id':instance,'estimated_hourly_usd':rate},indent=2))
        request(f'/api/v0/instances/{instance}/ssh/',{'ssh_key':key},'POST')
        print(json.dumps({'instance':instance,'estimated_hourly_usd':rate,'note':'Destroy the instance when finished; stopped disk can still incur charges.'}))
    elif a.action in {'start','stop'}:
        if a.id is None: p.error('Instance id required')
        instances=request('/api/v1/instances/?owner=me').get('instances',[])
        instance=next((o for o in instances if o['id']==a.id),None)
        if not instance: raise SystemExit('Instance not found in your account')
        if a.action=='start' and float(instance.get('dph_total',float('inf')))>2:
            raise SystemExit('Rejected: provider hourly rate exceeds $2')
        result=request(f'/api/v0/instances/{a.id}/',{'state':'running' if a.action=='start' else 'stopped'},'PUT')
        print(json.dumps({'success':result.get('success'),'requested_state':a.action,
                          'note':'Stopped instances retain disk storage charges.'}))
    elif a.action=='destroy':
        if a.id is None: p.error('Instance id required')
        r=request(f'/api/v0/instances/{a.id}/',method='DELETE')
        print(json.dumps({'success':r.get('success')}))
    else:
        r=request('/api/v1/instances/?owner=me')
        print(json.dumps([{k:o.get(k) for k in ['id','actual_status','gpu_name','dph_total','ssh_host','ssh_port']} for o in r.get('instances',[])],indent=2))

if __name__=='__main__': main()

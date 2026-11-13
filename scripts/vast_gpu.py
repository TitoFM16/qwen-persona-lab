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
import urllib.parse
import urllib.request

BASE = 'https://console.vast.ai'

def request(path, payload=None, method='GET'):
    token = os.environ.get('VAST_AI_API_KEY')
    if not token: raise SystemExit('Set VAST_AI_API_KEY; never pass credentials as arguments.')
    req = urllib.request.Request(BASE+path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization':'Bearer '+token, 'Content-Type':'application/json'}, method=method)
    with urllib.request.urlopen(req, timeout=90) as res: return json.load(res)

def offers():
    q = {'verified':{'eq':True}, 'rentable':{'eq':True}, 'rented':{'eq':False},
         'num_gpus':{'eq':1}, 'gpu_ram':{'gte':95000}, 'disk_space':{'gte':180},
         'order':[['dph_total','asc']], 'limit':100}
    return request('/api/v0/bundles/?'+urllib.parse.urlencode({'q':json.dumps(q)}))['offers']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['offers','rent','status','destroy'])
    p.add_argument('id',nargs='?',type=int)
    p.add_argument('--key',default='~/.ssh/id_ed25519.pub')
    a=p.parse_args()
    if a.action=='offers':
        print(json.dumps([{k:o.get(k) for k in ['id','gpu_name','dph_total','storage_cost','reliability','disk_space']} for o in offers() if 'RTX PRO 6000' in o.get('gpu_name','') and o['dph_total']<2],indent=2))
    elif a.action=='rent':
        o=next((o for o in offers() if o['id']==a.id),None)
        if not o or 'RTX PRO 6000' not in o.get('gpu_name',''): raise SystemExit('Offer unavailable or wrong GPU.')
        # Vast storage_cost is dollars per GB per month. Conservatively add full disk charge.
        rate=o['dph_total']+float(o.get('storage_cost',0))*180/720
        if rate>2: raise SystemExit(f'Rejected: conservative hourly estimate ${rate:.3f} exceeds $2.')
        result=request(f'/api/v0/asks/{a.id}/',{'client_id':'me','image':'pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime','disk':180,'runtype':'ssh_direc ssh_proxy','label':'qwen-persona-lab','cancel_unavail':True},'PUT')
        if not result.get('success'): raise SystemExit('Rental rejected: '+str(result.get('msg','unknown error')))
        instance=result['new_contract']
        key=Path(a.key).expanduser().read_text().strip()
        request(f'/api/v0/instances/{instance}/ssh/',{'ssh_key':key},'POST')
        print(json.dumps({'instance':instance,'estimated_hourly_usd':rate,'note':'Destroy the instance when finished; stopped disk can still incur charges.'}))
    elif a.action=='destroy':
        if a.id is None: p.error('Instance id required')
        r=request(f'/api/v0/instances/{a.id}/',method='DELETE')
        print(json.dumps({'success':r.get('success')}))
    else:
        r=request('/api/v1/instances/?owner=me')
        print(json.dumps([{k:o.get(k) for k in ['id','actual_status','gpu_name','dph_total','ssh_host','ssh_port']} for o in r.get('instances',[])],indent=2))

if __name__=='__main__': main()

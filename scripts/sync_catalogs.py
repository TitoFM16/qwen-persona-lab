#!/usr/bin/env python3
"""Download public source artifacts and import exact recipes. No account tokens needed."""
import argparse, csv, hashlib, json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.catalog import Catalog, NEURONARY_API

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Qwen-Persona-Laboratory/1.0'}),timeout=90) as response: data=response.read()
    path.write_bytes(data)
    path.with_suffix(path.suffix+'.provenance.json').write_text(json.dumps({'url':url,'retrieved_at':time.time(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},indent=2))
    return json.loads(data)

def sync_neuronary(catalog, directory='data/sources/neuronary', fetch=download):
    directory=Path(directory);offset=0;seen=set();errors=[];expected=None
    while True:
        page=fetch(f'{NEURONARY_API}/recipes?limit=100&offset={offset}',directory/f'page-{offset:06d}.json')
        if not isinstance(page,dict) or not isinstance(page.get('recipes'),list): raise ValueError('Malformed Neuronary page')
        expected=page['count'];rows=page['recipes']
        if not rows:
            if offset<expected:raise ValueError('Empty page before advertised total')
            break
        for row in rows:
            source_id=row['id']
            if source_id in seen:raise ValueError('Duplicate across pagination; catalog changed during download, retry sync')
            seen.add(source_id)
            try:
                full=fetch(f'{NEURONARY_API}/recipes/{source_id}',directory/'recipes'/f'{source_id}.json')
                catalog.upsert(full,'neuronary',source_id)
            except Exception as e: errors.append({'id':source_id,'error':str(e)})
        offset+=len(rows)
        if offset>=expected:break
    return {'expected':expected,'seen':len(seen),'imported':len(seen)-len(errors),'errors':errors}

def import_personas(catalog, path, output_csv='data/qwen3-8b-neuron-personas.csv'):
    source=json.loads(Path(path).read_text());fields=source['fields'];categories=source['categories']
    heads=['model','layer','neuron','sign','category','persona','weirdness','coherence','strength','incoherent_at','tags','reply_start']
    imported=0;counts={};errors=[]
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True)
    with open(output_csv,'w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=heads);writer.writeheader()
        for index,values in enumerate(source['rows']):
            r=dict(zip(fields,values));category=categories[r['cat']]
            export=dict(model='Qwen/Qwen3-8B',layer=r['L'],neuron=r['n'],sign=r['s'],category=category,persona=r['persona'],weirdness=r['w'],coherence=r['c'],strength=r['a'],incoherent_at=' '.join(map(str,r['before'])),tags=' '.join(r['tags']),reply_start=r['text'])
            writer.writerow(export)
            try:
                raw=dict(r,model='Qwen/Qwen3-8B',category=category,normalization='persona_library',source_url='https://neuronpersonas.com/data/qwen/library.json',reply_is_excerpt=True,excerpt_limit=source.get('excerpt'),name=r['persona'] or category)
                catalog.upsert(raw,'neuronpersonas',f"L{r['L']}N{r['n']}:{r['s']}")
                imported+=1;counts[category]=counts.get(category,0)+1
            except Exception as e:errors.append({'index':index,'error':str(e)})
    return {'imported':imported,'categories':counts,'errors':errors,'csv':output_csv,'reply_is_excerpt':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',default='data/lab.sqlite3');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('neuronary');nf=s.add_parser('neuronary-files');nf.add_argument('directory',nargs='?',default='data/sources/neuronary');s.add_parser('personas');ip=s.add_parser('personas-file');ip.add_argument('path')
    cp=s.add_parser('csv');cp.add_argument('path');cp.add_argument('--model');cp.add_argument('--source',default='manual')
    s.add_parser('counts');s.add_parser('queue')
    a=p.parse_args();c=Catalog(a.db)
    if a.command=='neuronary':result=sync_neuronary(c)
    elif a.command=='neuronary-files':
        directory=Path(a.directory)
        def cached_fetch(url,path): return json.loads((directory/Path(path).relative_to(directory)).read_text())
        result=sync_neuronary(c,directory,cached_fetch)
    elif a.command=='personas':
        path=Path('data/sources/neuronpersonas/library.json');download('https://neuronpersonas.com/data/qwen/library.json',path);result=import_personas(c,path)
    elif a.command=='personas-file':result=import_personas(c,a.path)
    elif a.command=='csv':result=c.import_csv(a.path,source=a.source,default_model=a.model)
    elif a.command=='queue':result={'queued':c.enqueue()}
    else:result=c.counts()
    print(json.dumps(result,indent=2));c.close()
if __name__=='__main__':main()

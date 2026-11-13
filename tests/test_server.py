"""No-socket HTTP and API checks; inference below is an isolated test double."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from lab.server import Application, handler

class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.worker_file=Path(self.temp.name)/'worker-status.json'
        self.judge_file=Path(self.temp.name)/'judge-status.json'
        self.app=Application(str(Path(self.temp.name)/'lab.sqlite3'),worker_status_file=self.worker_file,judge_status_file=self.judge_file)
        self.addCleanup(self.app.close_catalog)
        self.app.post('/api/import',{'csv':'name,model,recipe\nTest,Qwen/Qwen3-8B,L8N100+0.5\n'})
        self.recipe=self.app.get('/api/recipes',{})['recipes'][0]

    def request(self,method,path,headers,body=b''):
        h=handler(self.app).__new__(handler(self.app))
        h.server=SimpleNamespace(server_port=8787)
        h.path=path;h.headers=headers;h.rfile=io.BytesIO(body)
        output=[];h.send_data=lambda *args:output.append(args)
        getattr(h,'do_'+method)()
        return output[0]

    def test_catalog_queue_and_thread_affinity(self):
        self.assertEqual(self.app.get('/api/status',{})['counts']['imported'],1)
        self.assertEqual(self.app.get('/api/recipes',{'layer':['8'],'neuron':['100']})['recipes'][0]['id'],self.recipe['id'])
        self.assertFalse(self.app.get('/api/recipes',{'layer':['9']})['recipes'])
        self.assertEqual(self.app.post('/api/queue',{'recipe_ids':[self.recipe['id']]})['queued'],8)
        self.assertFalse(self.app.get('/api/history',{})['jobs'])
        job=self.app.catalog.claim_job()
        self.app.catalog.complete_job(job['id'],{'baseline':'a','steered':'b'})
        self.assertEqual(self.app.get('/api/history',{})['jobs'][0]['status'],'completed')
        self.assertEqual(len(self.app.get('/api/recipes',{'source':['manual CSV']})['recipes']),1)
        self.assertFalse(self.app.get('/api/recipes',{'source':['starter']})['recipes'])
        errors=[]
        def run():
            try: self.assertEqual(self.app.get('/api/status',{})['counts']['imported'],1)
            except Exception as exc: errors.append(exc)
            finally: self.app.close_catalog()
        threads=[threading.Thread(target=run) for _ in range(3)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertFalse(errors)

    def test_comparison_saved_and_separate_histories_forwarded(self):
        calls=[]
        class Runtime:
            def compare(self,*args,**kwargs):
                calls.append(kwargs)
                return {'baseline':'test-baseline','steered':'test-steered','metrics':{},'provenance':{}}
        self.app._runtime=Runtime()
        baseline=[{'role':'user','content':'test baseline'}]
        steered=[{'role':'user','content':'test steered'}]
        self.app.post('/api/compare',{'recipe_id':self.recipe['id'],'model':self.recipe['model'],'prompt':'test',
             'components':self.recipe['components'],'baseline_messages':baseline,'steered_messages':steered})
        self.assertEqual(calls[0]['baseline_messages'],baseline)
        self.assertEqual(calls[0]['steered_messages'],steered)
        self.assertEqual(len(self.app.get('/api/history',{})['experiments']),1)

    def test_host_origin_and_traversal_guards(self):
        self.assertEqual(self.request('GET','/',{'Host':'127.0.0.1:8787'})[0],200)
        self.assertEqual(self.request('GET','/../../lab/catalog.py',{'Host':'127.0.0.1:8787'})[0],404)
        self.assertEqual(self.request('GET','/api/status',{'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('POST','/api/queue',{'Host':'127.0.0.1:8787'})[0],403)
        self.assertEqual(self.request('POST','/api/queue',{'Host':'127.0.0.1:8787','Origin':'https://evil.example','X-Lab-Request':'1'})[0],403)
        body=json.dumps({'recipe_ids':[self.recipe['id']]}).encode()
        self.assertEqual(self.request('POST','/api/queue',{'Host':'127.0.0.1:8787','Origin':'http://127.0.0.1:8787',
          'X-Lab-Request':'1','Content-Length':str(len(body)),'Content-Type':'application/json'},body)[0],200)

    def test_recipe_detail_decodes_id_once_for_opaque_ids(self):
        from urllib.parse import quote
        host={'Host':'127.0.0.1:8787'}
        rid=self.recipe['id']
        raw=self.request('GET','/api/recipes/'+rid,host)
        encoded=self.request('GET','/api/recipes/'+quote(rid,safe=''),host)
        self.assertEqual(encoded[0],200)
        self.assertEqual(raw[0],200)
        self.assertEqual(encoded[1]['id'],rid)
        self.assertEqual(encoded[1]['name'],raw[1]['name'])
        # newly saved custom persona round-trips through its encoded ID
        saved=self.app.post('/api/personas',{'name':'Encoded round trip','model':self.recipe['model'],
              'components':self.recipe['components'],'normalization':'neuronary'})['recipe']
        got=self.request('GET','/api/recipes/'+quote(saved['id'],safe=''),host)
        self.assertEqual(got[0],200)
        self.assertEqual(got[1]['id'],saved['id'])
        self.assertEqual(got[1]['name'],'Encoded round trip')
        # no double decoding
        double=quote(quote(rid,safe=''),safe='')
        self.assertNotEqual(double,quote(rid,safe=''))
        self.assertEqual(self.request('GET','/api/recipes/'+double,host)[0],404)
        # opaque IDs with slash, percent and Unicode are plain SQL lookups, not filesystem routing
        self.app.post('/api/import',{'csv':'name,model,recipe\nOpaque,Qwen/Qwen3-8B,L9N7+0.5\n','source':'manual/path/%/é'})
        odd=self.app.get('/api/recipes',{'source':['manual/path/%/é']})['recipes'][0]
        for suffix in (quote(odd['id'],safe=''),odd['id']):
            got=self.request('GET','/api/recipes/'+suffix,host)
            self.assertEqual(got[0],200,suffix)
            self.assertEqual(got[1]['id'],odd['id'])
            self.assertEqual(got[1]['name'],'Opaque')
        # static traversal protection is a separate branch and is unchanged
        self.assertEqual(self.request('GET','/../../lab/catalog.py',host)[0],404)

    def test_chat_rejects_active_worker_before_loading_another_model(self):
        self.worker_file.write_text(json.dumps({'updated_at':time.time(),'state':'completed','runtime':{'loaded':True}}))
        body=json.dumps({'model':'Qwen/Qwen3-8B','prompt':'Who are you?','components':[]}).encode()
        headers={'Host':'127.0.0.1:8787','Origin':'http://127.0.0.1:8787',
                 'X-Lab-Request':'1','Content-Length':str(len(body)),'Content-Type':'application/json'}
        response=self.request('POST','/api/compare',headers,body)
        self.assertEqual(response[0],409)
        self.assertIn('after the batch',response[1]['error'])
        self.assertIsNone(self.app._runtime)
        self.assertFalse(self.app.status()['live_chat_available'])
        for state,loaded,age,blocked in [('loading_or_inference',False,0,True),('stopped',True,0,False),
                                         ('completed',True,60,False),('idle',False,0,False)]:
            self.worker_file.write_text(json.dumps({'updated_at':time.time()-age,'state':state,'runtime':{'loaded':loaded}}))
            self.assertEqual(self.app.status()['live_chat_available'],not blocked)

    def test_chat_rejects_active_judge_before_loading_another_model(self):
        self.worker_file.write_text(json.dumps({'updated_at':time.time(),'state':'stopped','runtime':{'loaded':False}}))
        body=json.dumps({'model':'Qwen/Qwen3-8B','prompt':'Who are you?','components':[]}).encode()
        headers={'Host':'127.0.0.1:8787','Origin':'http://127.0.0.1:8787',
                 'X-Lab-Request':'1','Content-Length':str(len(body)),'Content-Type':'application/json'}
        for state,loaded,age,blocked in [('loading_or_reviewing',False,0,True),('loaded',True,0,True),
                                         ('stopped',True,0,False),('loading_or_reviewing',True,60,False)]:
            self.judge_file.write_text(json.dumps({'updated_at':time.time()-age,'state':state,'loaded':loaded,
                                                'model':'Qwen/Qwen3.8-27B','batch_progress':{'judged_unique_cases':3,'unique_cases_total':10}}))
            status=self.app.status()
            self.assertEqual(status['live_chat_available'],not blocked)
            self.assertEqual(status['judge']['batch_progress']['judged_unique_cases'],3)
            if blocked:
                response=self.request('POST','/api/compare',headers,body)
                self.assertEqual(response[0],409)
                self.assertIn('GPU review is active',response[1]['error'])
                self.assertIsNone(self.app._runtime)

    def test_catalog_only_prevents_model_loading_and_preserves_browsing(self):
        self.app.catalog_only=True
        self.app.catalog.record_experiment(self.recipe['id'],self.recipe['model'],'Who are you?',
                                           {'baseline':'Recorded baseline','steered':'Recorded steered'})
        status=self.app.status()
        self.assertFalse(status['live_chat_available'])
        self.assertIn('Catalog-only mode',status['live_chat_block_reason'])
        body=json.dumps({'model':self.recipe['model'],'prompt':'Who are you?','components':[]}).encode()
        response=self.request('POST','/api/compare',{'Host':'127.0.0.1:8787','Origin':'http://127.0.0.1:8787',
                              'X-Lab-Request':'1','Content-Length':str(len(body)),'Content-Type':'application/json'},body)
        self.assertEqual(response[0],409)
        self.assertIsNone(self.app._runtime)
        self.assertEqual(len(self.app.get('/api/recipes',{})['recipes']),1)
        self.assertEqual(self.app.get('/api/history',{})['experiments'][0]['result']['baseline'],'Recorded baseline')
        self.assertEqual(self.app.post('/api/queue',{'recipe_ids':[self.recipe['id']]})['queued'],8)

    def test_save_personal_variant_preserves_source_without_queue_or_runtime(self):
        self.app.catalog_only=True
        body={'name':'My quiet narrator','description':'A personal dose experiment',
              'model':self.recipe['model'],'components':[{'layer':8,'neuron':100,'dose':-.25},
                  {'layer':9,'neuron':42,'dose':.125}], 'method':{'op':'add','apply_to':'all_tokens'},
              'normalization':'neuronary','parent_recipe_id':self.recipe['id']}
        saved=self.app.post('/api/personas',body)['recipe']
        self.assertEqual(saved['name'],'My quiet narrator')
        self.assertEqual(saved['source'],'custom')
        self.assertEqual(saved['status'],'pending')
        self.assertEqual(saved['components'],body['components'])
        self.assertEqual(saved['raw']['parent_recipe_id'],self.recipe['id'])
        self.assertEqual(saved['method'],body['method'])
        self.assertEqual(self.app.catalog.get_recipe(self.recipe['id'])['components'],self.recipe['components'])
        self.assertEqual(self.app.catalog.counts()['jobs'],{})
        self.assertIsNone(self.app._runtime)
        self.assertEqual(len(self.app.get('/api/recipes',{'source':['custom']})['recipes']),1)

    def test_personal_variant_validates_name_model_and_components(self):
        valid={'name':'My persona','model':self.recipe['model'],'components':self.recipe['components'],
               'normalization':'neuronary','method':{'op':'add'},'parent_recipe_id':self.recipe['id']}
        for changes in [{'name':''},{'name':'   '},{'name':'x'*121},{'model':'not-a-supported-model'},
                        {'components':[]},{'components':[{'layer':8,'neuron':12288,'dose':1}]},
                        {'components':[{'layer':8,'neuron':1,'dose':float('nan')}]},
                        {'components':[{'layer':8,'neuron':1}]},{'normalization':'unspecified'},
                        {'normalization':{}},{'method':{'op':'exec'}},
                        {'model':'Qwen/Qwen3.5-4B'},{'parent_recipe_id':'missing'}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.app.post('/api/personas',dict(valid,**changes))
        self.assertEqual(self.app.catalog.counts()['imported'],1)
        self.assertIsNone(self.app._runtime)

    def test_personal_variant_write_uses_same_origin_guard(self):
        body=json.dumps({'name':'Personal','model':self.recipe['model'],'components':self.recipe['components'],
                         'normalization':'neuronary'}).encode()
        headers={'Host':'127.0.0.1:8787','Origin':'https://unrelated.example','X-Lab-Request':'1',
                 'Content-Length':str(len(body)),'Content-Type':'application/json'}
        self.assertEqual(self.request('POST','/api/personas',headers,body)[0],403)
        self.assertEqual(self.app.catalog.counts()['imported'],1)
        headers['Origin']='http://127.0.0.1:8787'
        response=self.request('POST','/api/personas',headers,body)
        self.assertEqual(response[0],200)
        self.assertEqual(response[1]['recipe']['source'],'custom')

    def make_gated_app(self,sizes_on_disk,present=None):
        """Fake manifest + tiny files; lookup is injected and never touches network or the real HF cache."""
        import hashlib
        root=Path(self.temp.name)/'fake-cache'; root.mkdir(exist_ok=True)
        rev='a'*40; models={}
        for model,(w1,w2) in {'Qwen/Qwen3.5-4B':(4,6),'Qwen/Qwen3-8B':(5,7),'Qwen/Qwen3.8-27B':(8,9)}.items():
            tag=model.replace('/','_'); cfg=b'{}'
            models[model]={'revision':rev,'config_sha256':hashlib.sha256(cfg).hexdigest(),
                           'weights':[{'name':'a.safetensors','bytes':w1,'cache_blob_key':'k1'},{'name':'b.safetensors','bytes':w2,'cache_blob_key':'k2'}]}
            if present is None or model in present:
                (root/f'{tag}-config.json').write_bytes(cfg)
                for name,size in (('a.safetensors',w1),('b.safetensors',w2)):
                    on_disk=sizes_on_disk.get((model,name),size)
                    if on_disk is not None: (root/f'{tag}-{name}').write_bytes(b'x'*on_disk)
        manifest=Path(self.temp.name)/'manifest.json'; manifest.write_text(json.dumps({'models':models}))
        lookups=[]
        def lookup(model,filename,revision):
            lookups.append((model,filename,revision))
            path=root/f"{model.replace('/','_')}-{filename}"
            return str(path) if path.exists() else None
        app=Application(str(Path(self.temp.name)/'gated.sqlite3'),worker_status_file=self.worker_file,judge_status_file=self.judge_file,
                        require_cached_models=True,manifest_path=manifest,cache_lookup=lookup)
        self.addCleanup(app.close_catalog)
        app.lookups=lookups
        return app

    def test_default_flag_unset_is_nonrestrictive(self):
        self.assertFalse(self.app.require_cached_models)
        self.assertEqual(self.app.status()['model_readiness'],{})

    def test_all_cached_models_ready(self):
        app=self.make_gated_app({})
        readiness=app.status()['model_readiness']
        self.assertEqual(set(readiness),{'Qwen/Qwen3.5-4B','Qwen/Qwen3-8B','Qwen/Qwen3.8-27B'})
        for item in readiness.values():
            self.assertTrue(item['ready']); self.assertEqual(item['downloaded_bytes'],item['total_bytes'])
        self.assertTrue(all(rev=='a'*40 for _,_,rev in app.lookups))

    def test_missing_or_mismatched_weight_is_not_ready(self):
        app=self.make_gated_app({('Qwen/Qwen3.8-27B','b.safetensors'):None,('Qwen/Qwen3-8B','a.safetensors'):1})
        readiness=app.status()['model_readiness']
        self.assertTrue(readiness['Qwen/Qwen3.5-4B']['ready'])
        self.assertFalse(readiness['Qwen/Qwen3.8-27B']['ready'])
        self.assertEqual(readiness['Qwen/Qwen3.8-27B']['downloaded_bytes'],8)
        self.assertEqual(readiness['Qwen/Qwen3.8-27B']['total_bytes'],17)
        self.assertFalse(readiness['Qwen/Qwen3-8B']['ready'])
        self.assertIn('downloading',readiness['Qwen/Qwen3-8B']['reason'])

    def test_missing_config_manifest_or_dependency_fails_closed(self):
        app=self.make_gated_app({},present={'Qwen/Qwen3.5-4B'})
        self.assertTrue(app.model_readiness('Qwen/Qwen3.5-4B')['ready'])
        self.assertFalse(app.model_readiness('Qwen/Qwen3-8B')['ready'])
        app.manifest_path=Path(self.temp.name)/'absent.json'
        self.assertFalse(app.model_readiness('Qwen/Qwen3.5-4B')['ready'])
        app.manifest_path=Path(self.temp.name)/'manifest.json'
        def broken(*args): raise ImportError('no huggingface_hub')
        app.cache_lookup=broken
        self.assertFalse(app.model_readiness('Qwen/Qwen3.5-4B')['ready'])

    def test_uncached_compare_returns_409_without_runtime_and_ready_model_still_runs(self):
        app=self.make_gated_app({('Qwen/Qwen3.8-27B','a.safetensors'):None})
        def forbidden(): raise AssertionError('runtime must not be constructed for an uncached model')
        app.runtime=forbidden
        body=json.dumps({'model':'Qwen/Qwen3.8-27B','prompt':'hi','components':[]}).encode()
        self.app=app
        response=self.request('POST','/api/compare',{'Host':'127.0.0.1:8787','Origin':'http://127.0.0.1:8787',
                              'X-Lab-Request':'1','Content-Length':str(len(body)),'Content-Type':'application/json'},body)
        self.assertEqual(response[0],409)
        self.assertIn('still downloading',response[1]['error'])
        self.assertIsNone(app._runtime)
        class Runtime:
            def compare(self,*a,**k): return {'baseline':'b','steered':'s','metrics':{},'provenance':{}}
        del app.runtime
        app._runtime=Runtime()
        result=app.post('/api/compare',{'model':'Qwen/Qwen3.5-4B','prompt':'hi','components':[]})
        self.assertEqual(result['steered'],'s')

    def test_actual_published_atlas_integrity_and_hit_mapping(self):
        path=Path('data/sources/neuronpersonas/atlas.json')
        if not path.is_file():self.skipTest('Published atlas not downloaded')
        atlas=self.app.get('/api/atlas',{})
        raw=base64.b64decode(atlas['b64'],validate=True)
        self.assertEqual((atlas['layers'],atlas['neurons']),(36,12288))
        self.assertEqual(len(raw),36*12288)
        self.assertTrue(all((v&15)<len(atlas['codes']) and (v>>4)<len(atlas['codes']) for v in raw))
        for layer in range(36):
            for neuron in [0,127,128,12287]:
                x=(layer%6)*140+neuron%128+.5
                y=(layer//6)*128+22+neuron//128+.5
                decoded=(int(y//128)*6+int(x//140),(int(y%128)-22)*128+int(x%140))
                self.assertEqual((layer,neuron),decoded)

if __name__=='__main__':unittest.main()

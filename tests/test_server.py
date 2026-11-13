"""No-socket HTTP and API checks; inference below is an isolated test double."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from lab.server import Application, handler

class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app=Application(str(Path(self.temp.name)/'lab.sqlite3'))
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

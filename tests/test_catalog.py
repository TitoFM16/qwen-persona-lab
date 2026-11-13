import csv,json,sqlite3,tempfile,time,unittest
from pathlib import Path
from lab.catalog import Catalog,parse_neurons
from scripts.sync_catalogs import sync_neuronary,import_personas

class CatalogTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.c=Catalog(Path(self.tmp.name)/'test.sqlite3')
 def tearDown(self):self.c.close();self.tmp.cleanup()
 def recipe(self):return {'id':'a','name':'Test','model':'Qwen/Qwen3-8B','neurons':[{'layer':9,'neuron':3,'dose':-.525},{'layer':8,'neuron':7,'dose':1.25}]}
 def test_precision_and_provenance(self):
  raw=self.recipe();rid=self.c.upsert(raw,'neuronary');r=self.c.get_recipe(rid)
  self.assertEqual(r['neurons'],raw['neurons']);self.assertEqual(r['raw'],raw);self.assertEqual(r['normalization'],'neuronary')
  self.c.upsert(raw,'neuronary');self.assertEqual(self.c.counts()['imported'],1)
 def test_metadata_refresh_preserves_completed_validation(self):
  raw=self.recipe();rid=self.c.upsert(raw,'neuronary');self.c.enqueue([rid],['Who are you?'],1)
  job=self.c.claim_job();self.c.complete_job(job['id'],{'baseline':'A','steered':'B'})
  before=dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone())
  refreshed=dict(raw,name='Updated label',description='More source detail',examples=[{'output':'source example'}])
  refreshed['method']={'op':'add','apply_to':'all_tokens','site':'mlp.down_proj.input'}
  self.c.upsert(refreshed,'neuronary')
  self.assertEqual(self.c.get_recipe(rid)['status'],'completed')
  self.assertEqual(self.c.get_recipe(rid)['raw'],refreshed)
  self.assertEqual(dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()),before)
  self.assertEqual(self.c.list_experiments(),[])
 def test_dose_refresh_archives_old_evidence_and_resets_jobs(self):
  raw=self.recipe();rid=self.c.upsert(raw,'neuronary');self.c.enqueue([rid],['Who are you?'],2)
  job=self.c.claim_job();self.c.complete_job(job['id'],{'baseline':'A','steered':'B','provenance':{'model_revision':'old'}})
  job=self.c.claim_job();self.c.fail_job(job['id'],'old failure')
  original_jobs=[dict(row) for row in self.c.db.execute('SELECT * FROM jobs ORDER BY id')]
  changed=self.recipe();changed['neurons'][0]['dose']=-.125
  self.c.upsert(changed,'neuronary')
  self.assertEqual(self.c.get_recipe(rid)['status'],'pending')
  self.assertEqual(self.c.get_recipe(rid)['raw'],changed)
  for row in self.c.db.execute('SELECT * FROM jobs'):
   self.assertEqual(row['status'],'pending');self.assertEqual(row['attempts'],0)
   for key in ['result_json','error','worker','lease_until']:self.assertIsNone(row[key])
  archives=sorted(self.c.list_experiments(),key=lambda row:row['result']['archive']['job_snapshot']['id'])
  self.assertEqual(len(archives),2)
  for archived,original in zip(archives,original_jobs):
   evidence=archived['result']['archive']
   self.assertEqual(evidence['job_snapshot'],original)
   self.assertEqual(evidence['recipe_snapshot']['components'],raw['neurons'])
   self.assertEqual(evidence['recipe_snapshot']['raw'],raw)
   self.assertEqual(evidence['superseded_by']['components'],changed['neurons'])
  self.assertEqual(archives[0]['result']['steered'],'B')
  self.assertEqual(self.c.claim_job()['recipe']['components'],changed['neurons'])
 def test_intervention_refresh_rejects_active_job_atomically(self):
  raw=self.recipe();rid=self.c.upsert(raw,'neuronary');self.c.enqueue([rid],['Who are you?'],1)
  job=self.c.claim_job();before=dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone())
  changed=self.recipe();changed['neurons'][0]['dose']=.5
  with self.assertRaisesRegex(ValueError,'running jobs'):self.c.upsert(changed,'neuronary')
  self.assertEqual(self.c.get_recipe(rid)['raw'],raw)
  self.assertEqual(dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()),before)
  self.assertEqual(self.c.list_experiments(),[])
 def test_refresh_archive_and_reset_roll_back_together(self):
  raw=self.recipe();rid=self.c.upsert(raw,'neuronary');self.c.enqueue([rid],['Who are you?'],1)
  job=self.c.claim_job();self.c.complete_job(job['id'],{'baseline':'A','steered':'B'})
  before=dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone())
  self.c.db.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON recipes BEGIN SELECT RAISE(ABORT,'fixture rejection'); END")
  changed=self.recipe();changed['neurons'][0]['dose']=.5
  with self.assertRaises(sqlite3.IntegrityError):self.c.upsert(changed,'neuronary')
  self.assertEqual(self.c.get_recipe(rid)['raw'],raw)
  self.assertEqual(self.c.get_recipe(rid)['status'],'completed')
  self.assertEqual(dict(self.c.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()),before)
  self.assertEqual(self.c.list_experiments(),[])
 def test_signed_csv(self):
  p=Path(self.tmp.name)/'input.csv';p.write_text('layer,neuron,sign,strength,persona\n8,99,-1,0.75,Coffee\n8,99,1,1.25,Tea\n')
  report=self.c.import_csv(p,source='neuronpersonas-manual',default_model='Qwen/Qwen3-8B')
  self.assertEqual(report['imported'],2);self.assertEqual(self.c.list_recipes()[0]['neurons'][0]['dose'],-.75)
 def test_csv_embedded_newline_preserved(self):
  p=Path(self.tmp.name)/'multiline.csv'
  with p.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['name','recipe','description']);w.writeheader();w.writerow({'name':'Multiline','recipe':'L8N3-0.75','description':'first\nsecond'})
  self.c.import_csv(p,default_model='Qwen/Qwen3-8B')
  self.assertEqual(self.c.list_recipes()[0]['raw']['description'],'first\nsecond')
 def test_starter_csv_columns_preserve_method_and_name(self):
  raw={'model':'Qwen/Qwen3.5-4B','personality':'Narrator','components_json':'[{"layer":12,"neuron":4774,"dose":-1.2}]','method_op':'mul','method_site':'mlp.down_proj.input','apply_to':'generated_tokens','normalization':'layer_residual_median_over_down_column_l2'}
  r=self.c.get_recipe(self.c.upsert(raw,'starter'))
  self.assertEqual(r['name'],'Narrator');self.assertEqual(r['components'][0]['dose'],-1.2)
  self.assertEqual(r['method'],{'op':'mul','site':'mlp.down_proj.input','apply_to':'generated_tokens'})
  self.assertEqual(r['raw'],raw)
 def test_neuron_filter_matches_same_component(self):
  rid=self.c.upsert(self.recipe(),'test')
  self.assertEqual([r['id'] for r in self.c.list_recipes(layer=9,neuron=3)],[rid])
  self.assertEqual(self.c.list_recipes(layer=9,neuron=7),[])
  self.assertEqual(len(self.c.list_recipes(neuron=7)),1)
  self.assertEqual(self.c.list_recipes(model='Qwen/Qwen3.5-4B',layer=9,neuron=3),[])
 def test_exact_source_filter(self):
  self.c.upsert(self.recipe(),'starter');self.c.upsert(self.recipe(),'neuronary')
  self.assertEqual(len(self.c.list_recipes(source='starter')),1)
  self.assertEqual(self.c.list_recipes(source='starter')[0]['source'],'starter')
  self.assertEqual(len(self.c.list_recipes()),2)
 def test_history_prioritizes_actual_work_over_pending_catalog(self):
  rid=self.c.upsert(self.recipe(),'starter');self.c.enqueue([rid],['First','Second','Third'],2)
  job=self.c.claim_job();self.c.complete_job(job['id'],{'baseline':'a','steered':'b'})
  history=self.c.list_jobs(limit=1)
  self.assertEqual(history[0]['id'],job['id']);self.assertEqual(history[0]['status'],'completed')
 def test_invalid_not_silently_coerced(self):
  for value in ['L9N3', 'junk L9N3+1', [{'layer':1.1,'neuron':2,'dose':1}], [{'layer':1,'neuron':2,'dose':float('nan')}]]:
   with self.assertRaises(ValueError):parse_neurons(value)
 def test_jobs_resume_and_no_automatic_reproduction(self):
  rid=self.c.upsert(self.recipe(),'test');self.assertEqual(self.c.enqueue([rid],['Who are you?'],2),2);self.assertEqual(self.c.enqueue([rid],['Who are you?'],2),0)
  j=self.c.claim_job('a',lease_seconds=-1);j2=self.c.claim_job('b');self.assertEqual(j['id'],j2['id']);self.assertEqual(j2['attempts'],2)
  self.c.complete_job(j2['id'],{'changed':True});j=self.c.claim_job();self.c.fail_job(j['id'],'OOM',retry=True);j=self.c.claim_job();self.c.complete_job(j['id'],{'changed':True})
  self.assertEqual(self.c.counts()['completed'],1);self.assertEqual(self.c.counts()['reproduced'],0)
 def test_paginated_full_recipe(self):
  calls=[]
  def fetch(url,path):
   calls.append(url)
   if 'offset=0' in url:return {'count':2,'recipes':[{'id':'a'}]}
   if 'offset=1' in url:return {'count':2,'recipes':[{'id':'b'}]}
   r=self.recipe();r['id']=url.rsplit('/',1)[1];r['examples']=[{'prompt':'x','output':'y'}];return r
  report=sync_neuronary(self.c,self.tmp.name,fetch);self.assertEqual(report['imported'],2);self.assertEqual(len(calls),4)
 def test_persona_categories_no_filter(self):
  p=Path(self.tmp.name)/'lib.json';p.write_text(json.dumps({'fields':['L','n','s','persona','cat','w','c','tags','text','stop','a','before'],'categories':['persona','other'],'rows':[[8,2,-1,'Ghost',0,8,9,[],'text','eos',.75,[]],[9,3,1,'',1,4,9,[],'text','eos',1.25,[]]]}))
  report=import_personas(self.c,p,Path(self.tmp.name)/'out.csv');self.assertEqual(report['imported'],2);self.assertEqual(self.c.list_recipes()[0]['normalization'],'persona_library')
if __name__=='__main__':unittest.main()

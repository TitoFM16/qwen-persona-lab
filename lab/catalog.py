"""Persistent provenance-preserving catalog and leased validation queue (stdlib only)."""
from __future__ import annotations
import csv, hashlib, io, json, math, re, sqlite3, time
from pathlib import Path

NEURONARY_API = 'https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api'
STANDARD_PROMPTS = ['Who are you?', 'Explain why the sky looks blue in three sentences.', 'A colleague disagrees with my idea. How should I respond?', 'What is 17 multiplied by 19? Explain briefly.']

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def parse_neurons(value):
    if isinstance(value, str):
        try: value = json.loads(value)
        except (ValueError, TypeError):
            pattern = r'L\s*(\d+)\s*(?:E\s*(\d+)\s*)?N\s*(\d+)\s*([+\-−]\s*\d*\.?\d+(?:[eE][+\-]?\d+)?)'
            matches = list(re.finditer(pattern, value, re.I))
            remainder = re.sub(pattern, '', value, flags=re.I).strip(' ,;|+\t\n')
            if not matches or remainder: raise ValueError('Recipe must specify every layer, neuron and signed dose')
            value = [dict(layer=int(m[1]), neuron=int(m[3]), dose=float(m[4].replace(' ', '').replace('−','-')), **({'expert':int(m[2])} if m[2] else {})) for m in matches]
    if not isinstance(value, list) or not value: raise ValueError('Missing neuron recipe')
    result = []
    for item in value:
        n = dict(item)
        for key in ('layer', 'neuron'):
            v = float(n[key])
            if not v.is_integer() or v < 0: raise ValueError(f'{key} must be a nonnegative integer')
            n[key] = int(v)
        n['dose'] = float(n['dose'])
        if not math.isfinite(n['dose']): raise ValueError('Dose must be finite')
        if n.get('expert') is not None:
            v = float(n['expert'])
            if not v.is_integer() or v < 0: raise ValueError('expert must be a nonnegative integer')
            n['expert'] = int(v)
        result.append(n)
    return result

class Catalog:
    def __init__(self, db_path='data/lab.sqlite3'):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS recipes (
         id TEXT PRIMARY KEY, source_key TEXT NOT NULL, name TEXT NOT NULL, model TEXT NOT NULL,
         description TEXT NOT NULL, neurons_json TEXT NOT NULL, method_json TEXT NOT NULL,
         normalization TEXT NOT NULL, source TEXT NOT NULL, raw_json TEXT NOT NULL,
         status TEXT NOT NULL DEFAULT 'pending', imported_at REAL NOT NULL, updated_at REAL NOT NULL,
         UNIQUE(source,source_key));
        CREATE INDEX IF NOT EXISTS recipes_model ON recipes(model);
        CREATE TABLE IF NOT EXISTS imports (id INTEGER PRIMARY KEY, source TEXT, path TEXT, sha256 TEXT, imported INTEGER, errors_json TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS jobs (
         id INTEGER PRIMARY KEY, recipe_id TEXT REFERENCES recipes(id), prompt TEXT NOT NULL,
         seed INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
         worker TEXT, lease_until REAL, error TEXT, result_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
         UNIQUE(recipe_id,prompt,seed));
        CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,lease_until);
        CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, recipe_id TEXT, model TEXT,
         prompt TEXT, result_json TEXT NOT NULL, created_at REAL NOT NULL);
        ''')
        self.db.commit()

    def close(self): self.db.close()

    def upsert(self, raw, source, source_key=None, default_model=None):
        lowered = {str(k).strip().lower():v for k,v in raw.items()}
        model = lowered.get('model') or lowered.get('model_id') or default_model
        if model and '/' not in model and model.startswith('Qwen'): model = 'Qwen/' + model
        if not model: raise ValueError('Model is required; select it explicitly for model-less CSV')
        neurons = lowered.get('neurons') or lowered.get('components') or lowered.get('components_json') or lowered.get('recipe') or lowered.get('steering_recipe')
        if not neurons and ('layer' in lowered or 'l' in lowered):
            sign = str(lowered.get('sign', '1')).strip()
            strength = lowered.get('dose')
            if strength in (None, ''):
                strength = lowered.get('strength', lowered.get('a', lowered.get('strength_k')))
                if strength in (None, ''): raise ValueError('Missing dose/strength')
                strength = float(strength) * (-1 if sign in ('-1','-','−','down') else 1)
            neurons = [dict(layer=lowered.get('layer',lowered.get('l')), neuron=lowered.get('neuron',lowered.get('n')), dose=strength)]
        neurons = parse_neurons(neurons)
        method = lowered.get('method') or {k: lowered[column] for k, column in
                  [('op','method_op'), ('site','method_site'), ('apply_to','apply_to')]
                  if lowered.get(column)} or {'op':'add'}
        if isinstance(method,str):
            try: method=json.loads(method)
            except ValueError: method={'op':method}
        if not isinstance(method,dict): raise ValueError('method must be an object')
        normalization = lowered.get('normalization') or ('persona_library' if source.startswith('neuronpersonas') else 'neuronary' if source.startswith('neuronary') else 'unspecified')
        source_key = str(source_key or lowered.get('id') or hashlib.sha256(dumps(raw).encode()).hexdigest())
        rid = source + ':' + source_key
        name = lowered.get('name') or lowered.get('persona') or lowered.get('personality') or lowered.get('title') or source_key
        now=time.time()
        old=self.db.execute('SELECT raw_json,status FROM recipes WHERE id=?',(rid,)).fetchone()
        status=old['status'] if old and old['raw_json']==dumps(raw) else 'pending'
        self.db.execute('''INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET name=excluded.name,model=excluded.model,description=excluded.description,
         neurons_json=excluded.neurons_json,method_json=excluded.method_json,normalization=excluded.normalization,
         raw_json=excluded.raw_json,status=excluded.status,updated_at=excluded.updated_at''',
         (rid,source_key,str(name),str(model),str(lowered.get('description', lowered.get('desc',''))),dumps(neurons),dumps(method),normalization,source,dumps(raw),status,now,now))
        self.db.commit()
        return rid

    def import_csv(self, path, source='manual', default_model=None):
        path=Path(path); data=path.read_bytes(); errors=[]; imported=0
        for i,row in enumerate(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')),2):
            try:
                self.upsert(row,source,source_key=f'{hashlib.sha256(data).hexdigest()[:16]}:{i}',default_model=default_model)
                imported+=1
            except (ValueError,KeyError,TypeError) as e: errors.append({'line':i,'error':str(e),'raw':row})
        self.db.execute('INSERT INTO imports(source,path,sha256,imported,errors_json,created_at) VALUES (?,?,?,?,?,?)',(source,str(path),hashlib.sha256(data).hexdigest(),imported,dumps(errors),time.time()));self.db.commit()
        return {'imported':imported,'errors':errors}

    @staticmethod
    def _recipe(row):
        if row is None: return None
        r=dict(row)
        for key,target in [('neurons_json','neurons'),('method_json','method'),('raw_json','raw')]: r[target]=json.loads(r.pop(key))
        r['components']=r['neurons'];return r

    def get_recipe(self,id): return self._recipe(self.db.execute('SELECT * FROM recipes WHERE id=?',(id,)).fetchone())

    def list_recipes(self,query='',model=None,status=None,limit=50,offset=0,layer=None,neuron=None):
        clauses=[];args=[]
        if query:
            clauses.append('(name LIKE ? OR description LIKE ? OR raw_json LIKE ?)');args += ['%'+query+'%']*3
        if model: clauses.append('model=?');args.append(model)
        if status: clauses.append('status=?');args.append(status)
        components=[]
        if layer is not None: components.append("json_extract(n.value,'$.layer')=?");args.append(int(layer))
        if neuron is not None: components.append("json_extract(n.value,'$.neuron')=?");args.append(int(neuron))
        if components: clauses.append('EXISTS (SELECT 1 FROM json_each(recipes.neurons_json) n WHERE '+' AND '.join(components)+')')
        sql='SELECT * FROM recipes'+(' WHERE '+' AND '.join(clauses) if clauses else '')+' ORDER BY name,id LIMIT ? OFFSET ?'
        return [self._recipe(r) for r in self.db.execute(sql,args+[min(max(1,int(limit)),1000),max(0,int(offset))])]

    def counts(self):
        counts={k:0 for k in ('imported','completed','reproduced','failed','pending','running')}
        for r in self.db.execute('SELECT status,count(*) n FROM recipes GROUP BY status'):
            counts['imported']+=r['n'];counts[r['status']]=r['n']
        counts['jobs']={r['status']:r['n'] for r in self.db.execute('SELECT status,count(*) n FROM jobs GROUP BY status')}
        counts['sources']={r['source']:r['n'] for r in self.db.execute('SELECT source,count(*) n FROM recipes GROUP BY source')}
        return counts
    stats=counts

    def enqueue(self,recipe_ids=None,prompts=None,repeats=2):
        if recipe_ids is None: recipe_ids=[r[0] for r in self.db.execute('SELECT id FROM recipes')]
        if isinstance(recipe_ids,str): recipe_ids=[recipe_ids]
        if not 1<=int(repeats)<=20: raise ValueError('repeats must be 1..20')
        before=self.db.total_changes;now=time.time()
        for rid in recipe_ids:
            for prompt in prompts or STANDARD_PROMPTS:
                for seed in range(int(repeats)):
                    self.db.execute('INSERT OR IGNORE INTO jobs(recipe_id,prompt,seed,created_at,updated_at) VALUES (?,?,?,?,?)',(rid,prompt,seed,now,now))
        added=self.db.total_changes-before;self.db.commit();return added

    def claim_job(self,worker='worker',lease_seconds=600,model=None):
        now=time.time();self.db.execute('BEGIN IMMEDIATE')
        try:
            sql="SELECT j.* FROM jobs j JOIN recipes r ON r.id=j.recipe_id WHERE (j.status='pending' OR (j.status='running' AND j.lease_until<?))"
            args=[now]
            if model: sql+=' AND r.model=?';args.append(model)
            row=self.db.execute(sql+' ORDER BY j.id LIMIT 1',args).fetchone()
            if not row: self.db.commit();return None
            self.db.execute("UPDATE jobs SET status='running',attempts=attempts+1,worker=?,lease_until=?,updated_at=? WHERE id=?",(worker,now+lease_seconds,now,row['id']))
            self.db.execute("UPDATE recipes SET status='running' WHERE id=?",(row['recipe_id'],));self.db.commit()
            result=dict(row);result.update(status='running',attempts=row['attempts']+1,recipe=self.get_recipe(row['recipe_id']));return result
        except BaseException: self.db.rollback();raise

    def heartbeat(self,id,lease_seconds=600):
        self.db.execute('UPDATE jobs SET lease_until=?,updated_at=? WHERE id=? AND status=?',(time.time()+lease_seconds,time.time(),id,'running'));self.db.commit()

    def _refresh(self,rid):
        rows=self.db.execute('SELECT status FROM jobs WHERE recipe_id=?',(rid,)).fetchall();states=[r[0] for r in rows]
        status='running' if 'running' in states else 'pending' if 'pending' in states else 'failed' if 'failed' in states else 'completed'
        # Reproduction is an explicit evaluated claim, never inferred from changed text.
        if status=='completed':
            results=[json.loads(r[0]) for r in self.db.execute('SELECT result_json FROM jobs WHERE recipe_id=? AND result_json IS NOT NULL',(rid,))]
            if results and all(r.get('reproduced') is True for r in results): status='reproduced'
        self.db.execute('UPDATE recipes SET status=?,updated_at=? WHERE id=?',(status,time.time(),rid))

    def complete_job(self,id,result,status='completed'):
        if status not in ('completed','failed'): raise ValueError('Unsupported completion status')
        row=self.db.execute('SELECT recipe_id FROM jobs WHERE id=?',(id,)).fetchone()
        if not row: raise KeyError(id)
        self.db.execute('UPDATE jobs SET status=?,result_json=?,error=NULL,lease_until=NULL,updated_at=? WHERE id=?',(status,dumps(result),time.time(),id));self._refresh(row[0]);self.db.commit()

    def fail_job(self,id,error,retry=False):
        row=self.db.execute('SELECT recipe_id FROM jobs WHERE id=?',(id,)).fetchone()
        if not row: raise KeyError(id)
        self.db.execute('UPDATE jobs SET status=?,error=?,lease_until=NULL,updated_at=? WHERE id=?',('pending' if retry else 'failed',str(error),time.time(),id));self._refresh(row[0]);self.db.commit()

    def retry_failed(self):
        ids=[r[0] for r in self.db.execute("SELECT DISTINCT recipe_id FROM jobs WHERE status='failed'")]
        n=self.db.execute("UPDATE jobs SET status='pending',updated_at=? WHERE status='failed'",(time.time(),)).rowcount
        for rid in ids:self._refresh(rid)
        self.db.commit();return n

    def list_jobs(self,limit=100):
        rows=[dict(r) for r in self.db.execute('SELECT * FROM jobs ORDER BY id DESC LIMIT ?',(limit,))]
        for r in rows:
            if r['result_json']:r['result']=json.loads(r['result_json'])
        return rows

    def record_experiment(self,recipe_id,model,prompt,result):
        cur=self.db.execute('INSERT INTO experiments(recipe_id,model,prompt,result_json,created_at) VALUES (?,?,?,?,?)',(recipe_id,model,prompt,dumps(result),time.time()));self.db.commit();return cur.lastrowid

    def list_experiments(self,limit=100):
        rows=[dict(r) for r in self.db.execute('SELECT * FROM experiments ORDER BY id DESC LIMIT ?',(limit,))]
        for r in rows:r['result']=json.loads(r.pop('result_json'))
        return rows

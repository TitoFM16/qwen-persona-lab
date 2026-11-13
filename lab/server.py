"""Loopback-only, dependency-free browser application. Run python -m lab.server."""
from __future__ import annotations
import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent / 'static'
MODELS = ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3-8B', 'Qwen/Qwen3.8-27B']

class Application:
    def __init__(self, db_path, device='cuda'):
        from lab.catalog import Catalog
        self.db_path = db_path
        self._catalog_local = threading.local()
        initial = Catalog(db_path)
        initial.db.close()
        self.device = device
        self._runtime = None
        self._runtime_lock = threading.Lock()

    @property
    def catalog(self):
        if not hasattr(self._catalog_local, 'catalog'):
            from lab.catalog import Catalog
            self._catalog_local.catalog = Catalog(self.db_path)
        return self._catalog_local.catalog

    def runtime(self):
        with self._runtime_lock:
            if self._runtime is None:
                from lab.steering import ModelRuntime
                self._runtime = ModelRuntime(device=self.device)
            return self._runtime

    def status(self):
        runtime = {'loaded': False, 'device': self.device, 'model': None}
        if self._runtime is not None:
            try:
                status = self._runtime.status
                runtime = status() if callable(status) else status
            except Exception as exc:
                runtime = {'loaded': False, 'error': str(exc), 'device': self.device}
        return {'counts': self.catalog.stats(), 'runtime': runtime, 'models': MODELS,
                'atlas': {'Qwen/Qwen3.5-4B': {'available': False, 'assets_present': (ROOT.parent.parent / 'vendor/neuron-explorer/explorer/cache').exists(), 'reason': 'Original upstream 3D source and assets are preserved under vendor/neuron-explorer. Launch its local server separately using the documented safe launcher, then forward port 8791.'},
                          'Qwen/Qwen3-8B': {'available': (ROOT.parent.parent / 'data/sources/neuronpersonas/atlas.json').is_file(), 'reason': 'Published model-specific 2D neuron atlas imported from neuronpersonas.com. This is an indexed category atlas, not a 3D embedding.'},
                          'Qwen/Qwen3.8-27B': {'available': False, 'reason': 'No verified model-specific visualization data. Steering support must be validated on GPU.'}}}

    def get(self, path, q):
        one = lambda key, default=None: q.get(key, [default])[0]
        if path == '/api/status': return self.status()
        if path == '/api/atlas':
            atlas = ROOT.parent.parent / 'data/sources/neuronpersonas/atlas.json'
            if not atlas.is_file(): raise LookupError('Published Qwen3-8B atlas not imported')
            return json.loads(atlas.read_text())
        if path == '/api/recipes':
            limit = min(100, max(1, int(one('limit',50))))
            offset = max(0,int(one('offset',0)))
            return {'recipes': self.catalog.list_recipes(query=one('q',''), model=one('model'), status=one('status'), limit=limit, offset=offset, layer=int(one('layer')) if one('layer') is not None else None, neuron=int(one('neuron')) if one('neuron') is not None else None)}
        if path.startswith('/api/recipes/'):
            recipe = self.catalog.get_recipe(path.rsplit('/',1)[1])
            if recipe is None: raise LookupError('Recipe not found')
            return recipe
        if path == '/api/history': return {'experiments': self.catalog.list_experiments(limit=100), 'jobs': self.catalog.list_jobs(limit=100)}
        raise LookupError('Route not found')

    def post(self, path, body):
        if path == '/api/import':
            text = body.get('csv','')
            if not isinstance(text,str) or len(text)>20_000_000: raise ValueError('CSV must be text, at most 20 MB')
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',encoding='utf8') as file:
                file.write(text); file.flush()
                result = self.catalog.import_csv(file.name, source=body.get('source','manual CSV'), default_model=body.get('model'))
            return {'imported': result}
        if path == '/api/queue':
            ids = body.get('recipe_ids')
            if ids is not None and (not isinstance(ids,list) or len(ids)>100000): raise ValueError('Invalid recipe list')
            count = self.catalog.enqueue(recipe_ids=ids, repeats=max(1,min(5,int(body.get('repeats',2)))))
            return {'queued': count}
        if path == '/api/compare':
            model = body.get('model')
            if model not in MODELS: raise ValueError('Unsupported model')
            prompt = body.get('prompt','')
            if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>24000: raise ValueError('Enter a prompt of at most 24,000 characters')
            components = body.get('components',[])
            if not isinstance(components,list) or len(components)>100: raise ValueError('At most 100 neuron components')
            result = self.runtime().compare(model, prompt, components,
                       normalization=body.get('normalization','neuronary'), method=body.get('method'),
                       seed=int(body.get('seed',0)), max_new_tokens=max(1,min(1024,int(body.get('max_new_tokens',160)))),
                       messages=body.get('messages'), baseline_messages=body.get('baseline_messages'),
                       steered_messages=body.get('steered_messages'))
            self.catalog.record_experiment(body.get('recipe_id'),model,prompt,result)
            return result
        raise LookupError('Route not found')


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Avoid logging request bodies, prompts, recipes, and credentials.
            pass
        def send_data(self, status, payload, mime='application/json; charset=utf-8'):
            data = json.dumps(payload, ensure_ascii=False, default=str).encode() if not isinstance(payload,bytes) else payload
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-src http://127.0.0.1:* http://localhost:*; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers(); self.wfile.write(data)
        def check_host(self):
            host = self.headers.get('Host','')
            return host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        def do_GET(self):
            if not self.check_host(): return self.send_data(403, {'error':'Invalid host'})
            parsed = urlparse(self.path)
            try:
                if parsed.path.startswith('/api/'):
                    return self.send_data(200,app.get(parsed.path, parse_qs(parsed.query)))
                name = 'index.html' if parsed.path == '/' else parsed.path.lstrip('/')
                path = (ROOT/name).resolve()
                if not path.is_relative_to(ROOT) or not path.is_file(): raise LookupError('File not found')
                self.send_data(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            except LookupError as exc: self.send_data(404,{'error':str(exc)})
            except (ValueError,TypeError) as exc: self.send_data(400,{'error':str(exc)})
            except Exception as exc: self.send_data(500,{'error':str(exc)})
        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.check_host() or (origin and origin not in allowed) or self.headers.get('X-Lab-Request')!='1':
                return self.send_data(403,{'error':'Same-origin lab request required'})
            try:
                size = int(self.headers.get('Content-Length',0))
                if size<1 or size>21_000_000: raise ValueError('Invalid request size')
                if not self.headers.get('Content-Type','').startswith('application/json'): raise ValueError('JSON required')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body,dict): raise ValueError('JSON object required')
                self.send_data(200,app.post(urlparse(self.path).path,body))
            except LookupError as exc: self.send_data(404,{'error':str(exc)})
            except (ValueError,TypeError) as exc: self.send_data(400,{'error':str(exc)})
            except Exception as exc: self.send_data(503,{'error':str(exc),'hint':'Inference requires a configured GPU runtime and downloaded official model weights.'})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default='data/lab.sqlite3')
    parser.add_argument('--port',type=int,default=8787)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(Application(args.db,args.device)))
    print(f'Qwen Persona Laboratory: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()

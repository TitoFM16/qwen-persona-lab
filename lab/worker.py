"""Resumable single-GPU validation worker; SQLite leases survive interruptions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time

from lab.catalog import Catalog
from lab.steering import ModelRuntime


def _previous_outputs(db, recipe_id, prompt):
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT result_json FROM jobs WHERE recipe_id=? AND prompt=? AND status='completed'",
                           (recipe_id, prompt)).fetchall()
    return [json.loads(r[0])['steered'] for r in rows if r[0]]


def run(db='data/lab.sqlite3', model=None, device='cuda', max_jobs=0, max_new_tokens=160,
        stop=None, poll=False, source=None, status_file=None):
    catalog, runtime = Catalog(db), ModelRuntime(device=device)
    stop = stop or threading.Event()
    completed = failed = 0
    current = {'state': 'starting', 'job_id': None, 'model': model, 'source': source}
    monitor_stop = threading.Event()
    status_lock = threading.Lock()

    def snapshot():
        if status_file is None:
            return
        with status_lock:
            try:
                status = runtime.status
            except Exception as exc:
                status = {'gpu': None, 'error': str(exc)}
            payload = dict(current, updated_at=time.time(), completed_jobs=completed,
                           failed_jobs=failed, gpu=status.get('gpu'), runtime=status, pid=os.getpid())
            path = Path(status_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps(payload))
            temp.replace(path)

    def monitor():
        # SQLite connections must be created inside the thread that owns them.
        lease_catalog = Catalog(db)
        try:
            while not monitor_stop.wait(15):
                job_id = current.get('job_id')
                if job_id is not None:
                    lease_catalog.heartbeat(job_id, lease_seconds=1800)
                snapshot()
        finally:
            lease_catalog.close()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    snapshot()
    try:
        while not stop.is_set() and (not max_jobs or completed + failed < max_jobs):
            job = catalog.claim_job(worker='gpu-worker', lease_seconds=1800, model=model, source=source)
            if job is None:
                current.update(state='idle', job_id=None)
                snapshot()
                if poll:
                    stop.wait(3)
                    continue
                break
            recipe = job['recipe']
            current.update(state='loading_or_inference', job_id=job['id'], model=recipe['model'])
            snapshot()
            try:
                raw = recipe.get('raw', recipe.get('raw_json', {}))
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if raw.get('conditional'):
                    raise ValueError('Conditional recipes require a supported conditional adapter; not silently approximated')
                result = runtime.compare(
                    recipe['model'], job['prompt'], recipe.get('components', recipe.get('neurons', [])),
                    normalization=recipe.get('normalization') or 'neuronary',
                    method=recipe.get('method') or {}, max_new_tokens=max_new_tokens, seed=job['seed'])
                prior = _previous_outputs(db, job['recipe_id'], job['prompt'])
                result['metrics']['consistency_prior_runs'] = len(prior)
                result['metrics']['consistency_exact_match_fraction'] = (
                    sum(text == result['steered'] for text in prior) / len(prior) if prior else None)
                # Text difference alone cannot establish a persona, coherence, or factual accuracy.
                result['review'] = {'reproduced': None, 'personality': None, 'coherence': None,
                                    'accuracy': None, 'notes': 'Awaiting human review against recipe claim.'}
                result['provenance']['source_sampling'] = raw.get('sampling')
                result['provenance']['source_sampling_policy'] = (
                    'Standardized deterministic greedy baseline comparison; source sampling is retained '
                    'as metadata rather than silently changing decoding between recipes.')
                catalog.complete_job(job['id'], result)
                completed += 1
                current.update(state='completed', job_id=None)
                snapshot()
                print(json.dumps({'job': job['id'], 'status': 'completed', 'gpu': runtime.status['gpu']}), flush=True)
            except Exception as exc:
                catalog.fail_job(job['id'], f'{type(exc).__name__}: {exc}', retry=False)
                failed += 1
                current.update(state='failed', job_id=None, error=f'{type(exc).__name__}: {exc}')
                snapshot()
                print(json.dumps({'job': job['id'], 'status': 'failed', 'error': str(exc)}), flush=True)
                if isinstance(exc, (ImportError, ModuleNotFoundError)) or 'CUDA unavailable' in str(exc):
                    # A missing runtime is systemic; preserve the remaining queue.
                    break
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        runtime.unload()
        current.update(state='stopped', job_id=None)
        snapshot()
        catalog.close()
    return {'completed_jobs': completed, 'failed_jobs': failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/lab.sqlite3')
    parser.add_argument('--model')
    parser.add_argument('--source', help='Exact catalog source, e.g. starter, neuronary, neuronpersonas')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--max-jobs', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--poll', action='store_true')
    parser.add_argument('--status-file', default='.state/worker-status.json')
    args = parser.parse_args()
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    print(json.dumps(run(args.db, args.model, args.device, args.max_jobs, args.max_new_tokens,
                         stop, args.poll, args.source, args.status_file)))


if __name__ == '__main__':
    main()

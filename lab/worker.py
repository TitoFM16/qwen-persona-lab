"""Resumable single-GPU validation worker; SQLite leases survive interruptions."""
from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import threading
import time
import traceback

from lab.catalog import Catalog
from lab.steering import ModelRuntime


def _previous_outputs(db, recipe_id, prompt):
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT result_json FROM jobs WHERE recipe_id=? AND prompt=? AND status='completed'",
                           (recipe_id, prompt)).fetchall()
    return [json.loads(r[0])['steered'] for r in rows if r[0]]


def run(db='data/lab.sqlite3', model=None, device='cuda', max_jobs=0, max_new_tokens=160,
        stop=None, poll=False):
    catalog, runtime = Catalog(db), ModelRuntime(device=device)
    stop = stop or threading.Event()
    completed = failed = 0
    try:
        while not stop.is_set() and (not max_jobs or completed + failed < max_jobs):
            job = catalog.claim_job(worker='gpu-worker', lease_seconds=1800, model=model)
            if job is None:
                if poll:
                    stop.wait(3)
                    continue
                break
            recipe = job['recipe']
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
                catalog.complete_job(job['id'], result)
                completed += 1
                print(json.dumps({'job': job['id'], 'status': 'completed', 'gpu': runtime.status['gpu']}), flush=True)
            except Exception as exc:
                catalog.fail_job(job['id'], f'{type(exc).__name__}: {exc}', retry=False)
                failed += 1
                print(json.dumps({'job': job['id'], 'status': 'failed', 'error': str(exc)}), flush=True)
                if isinstance(exc, (ImportError, ModuleNotFoundError)) or 'CUDA unavailable' in str(exc):
                    # A missing runtime is systemic; preserve the remaining queue.
                    break
    finally:
        runtime.unload()
    return {'completed_jobs': completed, 'failed_jobs': failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/lab.sqlite3')
    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--max-jobs', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--poll', action='store_true')
    args = parser.parse_args()
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    print(json.dumps(run(args.db, args.model, args.device, args.max_jobs, args.max_new_tokens, stop, args.poll)))


if __name__ == '__main__':
    main()

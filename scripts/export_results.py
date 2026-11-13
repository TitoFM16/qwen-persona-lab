#!/usr/bin/env python3
"""Export a consistent read-only validation snapshot, without upgrading proxy scores to verdicts.

python scripts/export_results.py --db data/lab.sqlite3 --output outputs/results.jsonl \
    --report outputs/result-report.json --backup outputs/lab-backup.sqlite3
--limit limits JSONL output only; report counts and metrics cover the full snapshot.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

MATH_PROMPT = 'What is 17 multiplied by 19? Explain briefly.'


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def text_proxies(text, prompt):
    if not isinstance(text, str):
        return None
    words = re.findall(r'\w+', text.casefold())
    longest = run = 0
    previous = None
    for word in words:
        run = run + 1 if word == previous else 1
        longest = max(longest, run)
        previous = word
    return {
        'nonempty_text_proxy': bool(text.strip()),
        'word_count': len(words),
        'max_consecutive_repeated_word_run': longest,
        'repeated_word_run_at_least_8_proxy': longest >= 8,
        'math_expected_323_presence_proxy': bool(re.search(r'(?<!\d)323(?!\d)', text))
            if prompt == MATH_PROMPT else None,
    }


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    return con


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def consistent_backup(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    origin, target = readonly(source), sqlite3.connect(temporary)
    try:
        origin.backup(target, pages=1024)
    finally:
        target.close()
        origin.close()
    os.replace(temporary, destination)


def export_results(db='data/lab.sqlite3', output='outputs/results.jsonl',
                   report_path='outputs/result-report.json', backup=None, limit=None,
                   source=None, model=None):
    db, output, report_path = Path(db), Path(output), Path(report_path)
    if limit is not None and limit < 0:
        raise ValueError('limit must be nonnegative')
    paths = [db, output, report_path] + ([Path(backup)] if backup else [])
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Database, backup, JSONL, and report must use different paths')
    if backup:
        consistent_backup(db, backup)
    con = readonly(backup or db)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    clauses, arguments = [], []
    if source is not None:
        clauses.append('r.source=?'); arguments.append(source)
    if model is not None:
        clauses.append('r.model=?'); arguments.append(model)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    counters = {'completed_results': 0, 'changed_text': 0, 'unchanged_text': 0,
                'baseline_nonempty': 0, 'steered_nonempty': 0,
                'baseline_repeated_word_run_at_least_8': 0,
                'steered_repeated_word_run_at_least_8': 0,
                'math_prompt_results': 0, 'baseline_contains_323': 0, 'steered_contains_323': 0}
    consistency = []
    peaks = []
    exported = 0
    malformed = []
    try:
        con.execute('BEGIN')  # All queries share one snapshot while workers continue writing.
        recipe_counts = {row['status']: row['n'] for row in con.execute(
            'SELECT r.status,count(*) n FROM recipes r' + where + ' GROUP BY r.status', arguments)}
        job_counts = {row['status']: row['n'] for row in con.execute(
            'SELECT j.status,count(*) n FROM jobs j JOIN recipes r ON r.id=j.recipe_id' + where + ' GROUP BY j.status', arguments)}
        grouped = []
        for row in con.execute('SELECT r.source,r.model,r.status,count(*) n FROM recipes r' + where +
                               ' GROUP BY r.source,r.model,r.status ORDER BY r.source,r.model,r.status', arguments):
            grouped.append(dict(row))
        grouped_jobs = [dict(row) for row in con.execute(
            'SELECT r.source,r.model,j.status,count(*) n FROM jobs j JOIN recipes r ON r.id=j.recipe_id' +
            where + ' GROUP BY r.source,r.model,j.status ORDER BY r.source,r.model,j.status', arguments)]
        rows = con.execute('''SELECT j.*,r.name,r.description,r.source,r.source_key,r.model,
                              r.neurons_json,r.method_json,r.normalization,r.raw_json
                              FROM jobs j JOIN recipes r ON r.id=j.recipe_id''' + where +
                           (' AND ' if clauses else ' WHERE ') +
                           "j.status IN ('completed','failed','running') ORDER BY j.id", arguments)
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                r = dict(row)
                result = None
                if r['result_json']:
                    try:
                        result = json.loads(r['result_json'])
                    except ValueError as exc:
                        malformed.append({'job_id': r['id'], 'error': str(exc)})
                baseline = text_proxies(result.get('baseline'), r['prompt']) if isinstance(result, dict) else None
                steered = text_proxies(result.get('steered'), r['prompt']) if isinstance(result, dict) else None
                changed = (result['baseline'] != result['steered']) if (
                    isinstance(result, dict) and isinstance(result.get('baseline'), str) and
                    isinstance(result.get('steered'), str)) else None
                record = {
                    'job': {key: r[key] for key in ['id','recipe_id','prompt','seed','status','attempts',
                                                   'worker','lease_until','error','created_at','updated_at']},
                    'recipe': {'id':r['recipe_id'], 'name':r['name'], 'description':r['description'],
                               'source':r['source'], 'source_key':r['source_key'], 'model':r['model'],
                               'components':json.loads(r['neurons_json']), 'method':json.loads(r['method_json']),
                               'normalization':r['normalization'], 'raw':json.loads(r['raw_json'])},
                    'result':result, 'proxies':{'changed_text':changed, 'baseline':baseline,'steered':steered},
                }
                if r['result_json'] and result is None:
                    record['unparsed_result_json'] = r['result_json']
                if r['status'] == 'completed' and isinstance(result, dict):
                    counters['completed_results'] += 1
                    if changed is not None:
                        counters['changed_text' if changed else 'unchanged_text'] += 1
                    for label, proxy in [('baseline',baseline),('steered',steered)]:
                        if proxy:
                            counters[label+'_nonempty'] += int(proxy['nonempty_text_proxy'])
                            counters[label+'_repeated_word_run_at_least_8'] += int(proxy['repeated_word_run_at_least_8_proxy'])
                            if proxy['math_expected_323_presence_proxy'] is not None:
                                counters[label+'_contains_323'] += int(proxy['math_expected_323_presence_proxy'])
                    if r['prompt'] == MATH_PROMPT:
                        counters['math_prompt_results'] += 1
                    metric = result.get('metrics',{}).get('consistency_exact_match_fraction')
                    if isinstance(metric,(float,int)) and not isinstance(metric,bool):
                        consistency.append(metric)
                    gpu = result.get('provenance',{}).get('telemetry',{}).get('gpu') or {}
                    peak = gpu.get('peak_allocated_bytes')
                    if isinstance(peak,int) and peak >= 0:
                        peaks.append(peak)
                if limit is None or exported < limit:
                    handle.write(dumps(record)+'\n')
                    exported += 1
        os.replace(temporary,output)
        maximum_updated = con.execute('SELECT max(j.updated_at) FROM jobs j JOIN recipes r ON r.id=j.recipe_id'+where,arguments).fetchone()[0]
        report = {
            'kind':'readonly_validation_export', 'exported_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'database':str(db), 'snapshot_max_job_updated_at':maximum_updated,
            'filters':{'source':source,'model':model},
            'recipes':dict(imported=sum(recipe_counts.values()),**{key:recipe_counts.get(key,0) for key in
                            ['completed','reproduced','failed','pending','running']}),
            'recipe_status_counts':recipe_counts, 'job_status_counts':job_counts,
            'recipes_grouped_source_model_status':grouped, 'jobs_grouped_source_model_status':grouped_jobs,
            'proxy_counts':counters,
            'consistency_recorded_comparisons':len(consistency),
            'consistency_recorded_fraction_mean':sum(consistency)/len(consistency) if consistency else None,
            'gpu_peak_allocated_bytes_max':max(peaks) if peaks else None,
            'gpu_peak_measurement_count':len(peaks),
            'jsonl':{'path':str(output),'sha256':sha256(output),'exported_records':exported,'limit':limit,
                     'statuses':['completed','failed','running'],'order':'job_id ascending'},
            'malformed_results':malformed,
            'limits':['Text differences are not personality reproduction.',
                      'Nonempty text and repeated word runs are narrow coherence proxies, not coherence ratings.',
                      'Presence of 323 on the fixed arithmetic prompt is a narrow accuracy proxy, not general correctness.',
                      'Reproduction counts come only from stored recipe status; no verdict is inferred.',
                      'Report metrics use every completed job in the filtered snapshot, independently of JSONL limit.'],
        }
        if backup:
            report['backup']={'path':str(backup),'sha256':sha256(backup),'method':'SQLite online backup API'}
        report_temp=report_path.with_name(report_path.name+'.tmp')
        report_temp.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
        os.replace(report_temp,report_path)
        return report
    finally:
        con.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',default='data/lab.sqlite3')
    p.add_argument('--output',default='outputs/results.jsonl')
    p.add_argument('--report',default='outputs/result-report.json')
    p.add_argument('--backup',nargs='?',const='outputs/lab-backup.sqlite3')
    p.add_argument('--limit',type=int)
    p.add_argument('--source')
    p.add_argument('--model')
    args=p.parse_args()
    report=export_results(args.db,args.output,args.report,args.backup,args.limit,args.source,args.model)
    print(json.dumps({key:report[key] for key in ['recipes','job_status_counts','proxy_counts','jsonl']},indent=2))

if __name__=='__main__': main()

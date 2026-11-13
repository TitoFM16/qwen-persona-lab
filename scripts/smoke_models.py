#!/usr/bin/env python3
"""Run real inference smoke checks on each official model, recording failures honestly.

Coordinates below are arbitrary model-local index-valid probes. They are NOT persona
recipes, coordinates copied from another model, or claims of behavioral reproduction.
Run from repo root: python scripts/smoke_models.py --output data/smoke_models.json
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.steering import MODELS, ModelRuntime


def write_report(path, report):
    """Atomic checkpoint: a killed model download cannot erase earlier results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def run(device='cuda', max_new_tokens=32, models=None, output=None, resume=False):
    selected = list(models or MODELS)
    if any(model not in MODELS for model in selected):
        raise ValueError('Unsupported smoke model')
    invocation = {'device': device, 'max_new_tokens': max_new_tokens, 'models': selected}
    previous = {}
    if resume and output and Path(output).exists():
        saved = json.loads(Path(output).read_text())
        if saved.get('invocation') != invocation:
            raise ValueError('Existing smoke report uses different settings; select a new --output or omit --resume')
        previous = {r['model']: r for r in saved['models'] if r['status'] == 'passed'}
    runtime = ModelRuntime(device=device)
    reports = []

    def checkpoint():
        result = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'kind': 'inference_smoke_checks_not_persona_validation',
                  'invocation': invocation, 'models': reports,
                  'passed': sum(r['status'] == 'passed' for r in reports)}
        if output:
            write_report(output, result)
        return result

    for model in selected:
        if model in previous:
            reports.append(previous[model])
            checkpoint()
            print(json.dumps({'model': model, 'status': 'passed', 'resumed': True}), flush=True)
            continue
        report = {'model': model, 'coordinate_purpose': 'arbitrary model-local hook and shape probes; no persona claim',
                  'status': 'running', 'checks': {}}
        reports.append(report)
        checkpoint()
        started = time.monotonic()
        try:
            # Shape proportions are computed independently from each model architecture.
            dims = MODELS[model]
            first = {'layer': dims['layers'] // 3, 'neuron': dims['neurons'] // 7, 'dose': .05}
            second = {'layer': dims['layers'] // 2, 'neuron': dims['neurons'] // 11, 'dose': -.05}
            zero = dict(first, dose=0.)
            base = runtime.compare(model, 'Who are you?', [], max_new_tokens=max_new_tokens)
            zero_result = runtime.compare(model, 'Who are you?', [zero], max_new_tokens=max_new_tokens)
            single = runtime.compare(model, 'Who are you?', [first], max_new_tokens=max_new_tokens)
            multi = runtime.compare(model, 'Who are you?', [first, second], max_new_tokens=max_new_tokens)
            cleanup = runtime.compare(model, 'Who are you?', [], max_new_tokens=max_new_tokens)
            report['checks'] = {
                'baseline_has_text': bool(base['baseline']),
                'zero_dose_equals_baseline': zero_result['baseline'] == zero_result['steered'] == base['baseline'],
                'single_hook_completed': bool(single['steered']),
                'multi_hook_completed': bool(multi['steered']),
                'baseline_after_hook_cleanup_unchanged': cleanup['baseline'] == cleanup['steered'] == base['baseline'],
            }
            report['results'] = {'baseline': base, 'zero': zero_result, 'single': single, 'multi': multi,
                                 'cleanup': cleanup}
            memory = [result.get('provenance', {}).get('telemetry', {}).get('gpu')
                      for result in report['results'].values()]
            memory = [gpu for gpu in memory if gpu]
            report['measured_memory'] = {
                'peak_allocated_bytes': max((gpu['peak_allocated_bytes'] for gpu in memory), default=None),
                'gpu_total_bytes': memory[0]['total_bytes'] if memory else None,
                'gpu_name': memory[0]['name'] if memory else None,
                'source': 'torch.cuda telemetry during actual inference; null when unavailable',
            }
            report['status'] = 'passed' if all(report['checks'].values()) else 'failed'
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
            report['error'] = 'KeyboardInterrupt; resume retries this model and preserves prior passes'
            raise
        except Exception as exc:
            report['status'] = 'failed'
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            report['elapsed_seconds'] = time.monotonic() - started
            runtime.unload()
            checkpoint()
        print(json.dumps({'model': model, 'status': report['status'], 'error': report.get('error')}), flush=True)
    return checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--output', default='data/smoke_models.json')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), help='Subset of official models to smoke test')
    parser.add_argument('--resume', action='store_true', help='Keep prior passes with identical settings; retry failed/interrupted models')
    args = parser.parse_args()
    report = run(args.device, args.max_new_tokens, args.models, args.output, args.resume)
    return 0 if report['passed'] == len(args.models or MODELS) else 1


if __name__ == '__main__':
    raise SystemExit(main())

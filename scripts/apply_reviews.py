"""Attach explicitly labeled Codex rubric assessments to matching completed outputs.

Never generates outputs or infers reproduction from text changes. Evidence JSONL
must match the current baseline and steered text before any review is applied.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.catalog import Catalog


def apply_reviews(db, assessment, evidence):
    assessment, evidence = Path(assessment), Path(evidence)
    document = json.loads(assessment.read_text())
    snapshots = {r['job']['id']: r for r in
                 (json.loads(line) for line in evidence.read_text().splitlines() if line.strip())}
    catalog = Catalog(db)
    changes = []
    seen = set()
    try:
        for recipe in document['recipes']:
            for review in recipe['per_job']:
                jid = review['job_id']
                if jid in seen:
                    raise ValueError(f'Duplicate job {jid}')
                seen.add(jid)
                verdict = review.get('reproduced')
                if verdict is not None and not isinstance(verdict, bool):
                    raise ValueError('Reproduction must be true, false, or null')
                for field, maximum in [('personality_fidelity', 3), ('coherence', 4)]:
                    score = review[field]
                    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= maximum:
                        raise ValueError(f'Invalid {field}')
                if verdict is True and (review['personality_fidelity'] != 3 or review['coherence'] < 3):
                    raise ValueError('Positive reproduction requires clear target fidelity and coherent output')
                old = catalog.db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
                snap = snapshots.get(jid)
                if not old or old['status'] != 'completed' or not snap:
                    raise ValueError(f'Job {jid} is not a completed evidenced result')
                if old['recipe_id'] != recipe['recipe_id'] or snap['recipe']['id'] != recipe['recipe_id']:
                    raise ValueError(f'Recipe mismatch for job {jid}')
                result = json.loads(old['result_json'])
                if any(result.get(k) != snap['result'].get(k) for k in ('baseline', 'steered')):
                    raise ValueError(f'Output mismatch for job {jid}; export fresh evidence')
                provenance = result.setdefault('provenance', {})
                snapshot_review = snap['result'].get('review') or {}
                snapshot_explicit_review = isinstance(snapshot_review, dict) and any(
                    key in snapshot_review for key in ('reviewer', 'review_type', 'kind'))
                generated_at = snap['job'].get('updated_at')
                if ('generation_completed_at_unix' not in provenance and not snapshot_explicit_review
                        and type(generated_at) in (int, float) and math.isfinite(generated_at) and generated_at > 0):
                    provenance['generation_completed_at_unix'] = generated_at
                    provenance['generation_completed_at_resolution_note'] = (
                        'Recovered from exact-output-matching evidence snapshot job.updated_at '
                        'recorded before any explicit review; later review writes retain this timestamp.')
                result['review'] = dict(review, reviewer='Codex', kind='exploratory_agent_rubric',
                    reviewed_at=time.time(), rubric_version='exploratory-v1',
                    assessment_sha256=hashlib.sha256(assessment.read_bytes()).hexdigest(),
                    evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    limitations='Interpretive agent assessment of the recorded prompts; not independent human validation.')
                result['reproduced'] = verdict
                result.setdefault('metrics', {}).update(personality=review['personality_fidelity'],
                    coherence=review['coherence'], accuracy=review.get('accuracy'),
                    rating_status='exploratory_codex_review')
                changes.append((jid, result))
        # Validate the whole document before writing any assessment.
        for jid, result in changes:
            catalog.complete_job(jid, result)
        return {'reviewed_jobs': len(changes), 'counts': catalog.counts(),
                'kind': 'exploratory_agent_rubric_not_independent_human_validation'}
    finally:
        catalog.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('assessment')
    p.add_argument('--evidence', required=True)
    p.add_argument('--db', default='data/lab.sqlite3')
    a = p.parse_args()
    print(json.dumps(apply_reviews(a.db, a.assessment, a.evidence), indent=2))


if __name__ == '__main__':
    main()

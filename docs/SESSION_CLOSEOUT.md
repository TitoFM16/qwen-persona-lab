# Presentation session closeout — November 13, 2026

Vast.ai instance **60502794** was destroyed at the user’s request. Destruction was confirmed at **2026-11-13T20:43:10.620349+00:00**, and a separate provider account listing returned **[]**. No rented compute or instance disk remains. Live inference and the GPU-backed explorer are offline.

## Preserved locally

The shutdown command exported results using the SQLite online backup API and downloaded them before destruction:

- `outputs/presentation-60502794/presentation-backup.sqlite3`
- `outputs/presentation-60502794/presentation-results.jsonl`
- `outputs/presentation-60502794/presentation-result-report.json`
- `outputs/presentation-60502794/evidence.tar.gz`

Backup integrity check: **ok**. SHA-256: `ae9c286660653512cc127b41ff4fd4fe8ca93e8d012f7e5a2bfb946eec522977`. The backup hash matches the export report. These artifacts are excluded from Git; a Git clone alone does not include the local experiment database, downloaded catalogs, credentials or model weights.

## Final exported validation counts

| Recipe status | Count |
|---|---:|
| Total records, including saved custom personas | 78,431 |
| Completed without reproduction acceptance | 301 |
| Reproduced under the exploratory model-assisted rubric | 3 |
| Failed | 4 |
| Pending | 78,123 |
| Running | 0 |

The total includes 78,419 source records and 12 saved custom personas. Standardized validation jobs: **2,432 completed**, **32 failed**, **624,896 pending**. Pending jobs are persisted queue entries, not active work. Exploratory live-chat comparisons and persona screening are separate from those standardized job counts. Reproduction ratings are not independent human or scientific validation.

All three BF16 models were verified resident together during the session. Historical measurements and limitations remain in [verification.md](verification.md). No new GPU tests were run for this documentation update.

To launch another session, follow [RUNBOOK.md](RUNBOOK.md), selecting the latest validated local backup. The runbook caps rental quotes below US$2/hour and provides a bounded shutdown lifecycle. Starting another session incurs new provider charges.

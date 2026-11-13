# Presentation launch and bounded rental lifecycle

The completed experiment evidence and database backup are saved on this Mac. There is no reason to leave a GPU idle overnight. Keep `outputs/gpu-final/` and the source folder together; generated artifacts and model weights are not stored in Git.

## One command at presentation time

### Authorized November 13 session: until 4 PM Bogotá

For the explicitly authorized longer session, use the absolute deadline **2026-11-13 21:00 UTC**, which is **4 PM Bogotá**. The estimate cap below is US$20, leaving a margin against the stated US$23 credit. This deadline is intentionally date-specific; do not reuse it on a later day.

```bash
cd /Users/juanfigueroa/workspace/evil_qwen
mkdir -p outputs .state
nohup python3 scripts/presentation.py \
  --env-file /Users/juanfigueroa/workspace/johan_kaseya/sleeper_demo/.env \
  --until 2026-11-13T21:00:00Z --max-hourly 2 --max-total 20 \
  --prefetch-models Qwen/Qwen3.5-4B Qwen/Qwen3-8B Qwen/Qwen3.8-27B \
  --fast-kernels > outputs/presentation-controller.log 2>&1 < /dev/null &
echo "$!" > .state/presentation-controller.pid
```

This detached controller respects `nohup`'s SIGHUP policy and can remain active after the launching Terminal exits. Watch `outputs/presentation-controller.log` and `outputs/presentation-setup.log`. Its process ID, contract ID, UTC deadline and full-session cost estimate are persisted in `.state/presentation.json`. No raw credential is passed on the command line or stored in that state.

The local `caffeinate` child prevents ordinary idle sleep while the controller is alive. A separate remote watchdog is armed before dependency installation and uses only the provider-injected container credential to delete its own instance at the same absolute deadline, independently of the Mac. It is not an external provider scheduler: host failure or provider API failure can still delay cleanup. The timer falls back to stopping the instance after bounded deletion failures; stopped disk can retain storage charges.

### Default short-session command

Run this from a normal Mac Terminal, with this project as the working directory:

The saved-results catalog viewer is currently a local process only, with PID **30684** recorded in `.state/local-viewer.pid`. Free port 8787 before launching the GPU tunnel. Verify that the recorded PID still belongs to this lab's `python -m lab.server`, then stop that viewer:

```bash
ps -p "$(cat .state/local-viewer.pid)" -o pid=,command=
# Only if the displayed command is this lab's viewer:
kill -TERM "$(cat .state/local-viewer.pid)"
```

If the PID is absent or now belongs to another application, do not kill it. The presentation controller refuses occupied ports before renting.

```bash
cd /Users/juanfigueroa/workspace/evil_qwen
python3 scripts/presentation.py \
  --env-file /Users/juanfigueroa/workspace/johan_kaseya/sleeper_demo/.env \
  --max-hourly 2 --duration-hours 2 --open-browser \
  --prefetch-models Qwen/Qwen3.5-4B Qwen/Qwen3-8B Qwen/Qwen3.8-27B
```

The env file is read on the Mac only. Only its `VAST_AI_API_KEY` entry is used; the file is never executed or transferred. The default SSH public key is `~/.ssh/id_ed25519.pub`; use `--key /path/to/public_key.pub` if needed. The matching private key stays on the Mac, or your SSH agent provides authentication.

The controller:

1. Refuses to rent if a previous presentation is unresolved or either local UI port is occupied.
2. Creates a sanitized source/data/vendor bundle and a consistent SQLite snapshot from the newest schema-validated lab database under `outputs/**/*backup*.sqlite3`. Use `--backup PATH` to choose a specific verified backup.
3. Searches verified one-GPU RTX PRO 6000 offers with at least 95,000 MiB VRAM and enough disk, then rejects a conservative compute-plus-180GB-storage estimate above US$2/hour.
4. Rents the selected offer, records its contract ID immediately, and verifies its SSH public key. An exact account-registered key avoids redundant attachment; an explicit duplicate-key response still requires actual SSH authentication. Unknown attachment failures are surfaced with redacted error details. SSH readiness is bounded to ten minutes.
5. Transfers the source, authentic dataset snapshots, original explorer assets and backed-up experiment database. It excludes `.env*`, `.state`, `.git`, `.venv`, credential key files, logs and local model caches.
6. Arms and verifies the remote self-deletion watchdog immediately after unpacking, then installs tested dependencies using the CUDA image's working PyTorch. Setup refuses to continue if the watchdog does not arm. It runs ordinary software tests, but **does not launch validation, an automated judge or exhaustive model smoke checks**.
7. Downloads the requested official weights at the immutable revisions in `data/model-cache-manifest.json`, starts the lab and safe original explorer, and opens loopback SSH forwards. Bootstrap constrains Transformers to the tested **5.22.2** release. The command above prefetches all three models so the **4B OG Evil** and **27B Evil** demonstrations do not need a large first-use download; Qwen3-8B is available for its separate persona examples. Other recipe/model selections remain available in the lab.

Open **http://127.0.0.1:8787** for the persona lab and **http://127.0.0.1:8791** for the original 3D explorer. Optional `--fast-kernels` installs the tested GCC/G++ and pinned FLA packages before launch. To prefetch every model, add:

```bash
--models Qwen/Qwen3.5-4B Qwen/Qwen3-8B Qwen/Qwen3.8-27B
```

There are no local copies of the roughly 76 GB GPU model cache. A fresh rental downloads the requested models again. Allow at least 15 minutes for the small-model setup and plan **30–60 minutes of lead time when prefetching all three**, depending on image pulls, offer networking, Hugging Face throughput and package downloads. The script's default is only 4B, but the command above explicitly requests all three. Follow `outputs/presentation-setup.log` from another Terminal while setup runs.

## Adopt a running rental (`--resume-existing`) and controller handoff

Use this only when a controller process is already running a rental whose setup is stuck behind the old fixed one-hour SSH limit (new controllers no longer have it; setup waits are bounded by the absolute deadline). Adoption never rents, never extends the deadline, never bootstraps and never re-arms the remote timer.

Required: `--expected-instance-id ID` and `--until` equal to the saved deadline. Example (instance 60502794, deadline 2026-11-13T21:00:00Z):

```bash
caffeinate -i .venv/bin/python scripts/presentation.py --env-file /private/path/keys.env \
  --resume-existing --expected-instance-id 60502794 --until 2026-11-13T21:00:00Z
```

Phase 1 (read-only, old controller still protects the rental): validates saved purpose/ID/label/unexpired deadline, finds the instance with the same ID and label in the provider account (running, same SSH endpoint), connects over strict-host-key SSH using the saved known-hosts, and reads `.state/remote-deadline.json` (must be `armed`, same instance, same deadline, timer process alive). Any failure exits without touching state, the lock, the old controller or the rental. It then writes `.state/presentation-handoff.json` (`phase: ready_for_takeover`, `old_pid`, `new_pid`, validated timer) and polls the `.state/presentation.lock` flock.

Phase 2 (root/operator):

1. Check the marker: `cat .state/presentation-handoff.json` shows `ready_for_takeover`, the expected `old_pid` and live `new_pid`.
2. Kill **only the old local controller**: `kill -KILL OLD_PID`. **Never SIGTERM/SIGINT/Ctrl+C/`--stop` on the old controller**: its cleanup destroys the rental. Cancel an unwanted replacement with Ctrl+C on the *replacement* while the marker says `ready_for_takeover`; that touches nothing.
3. The replacement acquires the lock only. While holding it, it revalidates account/ID/label and the unchanged armed timer, then atomically commits its PID to `.state/presentation.json`. **Only after that commit** does it own cleanup and set the marker to `taken_over`. Any failure, timeout or cancel before the commit releases the lock, removes the marker, leaves main state untouched and makes zero DELETE calls. From then on Ctrl+C/deadline/`--stop` destroy the rental (bounded backup first). Explicit `--stop` is validated (purpose, label, integer-or-absent instance ID) before it may delete; `destroy()` is inert for any controller without ownership.
4. After the commit, and before `start_services.sh`, the controller copies the newest local `lab/server.py`, `lab/static/{app.js,index.html,style.css}` and `scripts/start_services.sh` to `/workspace/persona-lab` over strict-host-key scp (no credentials, `.state` or `.env`). All five local files must exist before the marker is written. The existing remote downloader is never restarted.

Models: readiness uses the pinned revisions in `data/model-cache-manifest.json`; Qwen3.5-4B is ready only when config, tokenizer files and every unique shard in `model.safetensors.index.json` exist as complete non-empty files (broken links and `.incomplete` blobs are not ready). As soon as 4B is ready the controller runs `scripts/start_services.sh` and opens tunnels; 8B/27B continue in the background (default `--models` is all three when adopting). It launches one trusted `nohup` pinned downloader only if no `snapshot_download` process is already running; it never signals existing remote processes. Caveat: if the old SSH session's death hangs up a downloader that is not `nohup`/detached, that is outside this controller; check the remote `pgrep -af snapshot_download` after the SIGKILL.

## End the presentation and remove the rental

Press **Ctrl+C** in the controller Terminal. It closes only its own tunnel and **destroys its recorded rental**, rather than merely stopping the GPU and retaining paid disk storage. It confirms the instance is absent from the provider listing. Setup errors, normal termination and the wall-clock deadline use the same cleanup path.

If that Terminal has disappeared, run:

```bash
python3 scripts/presentation.py \
  --env-file /Users/juanfigueroa/workspace/johan_kaseya/sleeper_demo/.env --stop
```

`--stop` means destroy this controller's recorded presentation rental. It does not act on arbitrary instances, the older completed experiment rental or unrelated account resources. Ownership state is kept in `.state/presentation.json`; no API key is stored there. If cleanup cannot be confirmed, the script prints an explicit error, retains the instance ID and marks `cleanup_failed`. Open the Vast Instances page and destroy that recorded instance promptly if the API is unavailable.

The default two-hour deadline **includes setup time**. Use a smaller `--duration-hours`, for example `0.5`, for a short demo. A longer session is allowed only through an explicit timezone-aware `--until`, at most twelve hours in the future; naive, expired and more distant timestamps are rejected. US$2/hour remains a hard quote ceiling. Offers are also rejected when their conservative hourly estimate multiplied by the whole requested session exceeds `--max-total` (default US$23). This is an estimated compute/storage budget, not a provider-enforced account-spending limit; transfer charges and marketplace billing remain provider-dependent.

About 130 seconds before the deadline, the controller attempts a bounded online SQLite backup and JSONL/report export while both UIs remain available. It then keeps the session live until the exact deadline. Ctrl+C/termination also attempts a bounded backup when enough time remains. Evidence is copied into `outputs/presentation-INSTANCE_ID/`; a backup failure or timeout **never prevents rental destruction**. The controller never deliberately extends the rental deadline to finish a download. Very late edits, unavailable networking or interrupted backup may not be captured; retain important presentation changes locally as well.

**Keep the Mac awake and the controller Terminal open.** On macOS the controller automatically runs `caffeinate -i -w PID` to prevent ordinary idle sleep for its own lifetime. This cannot guarantee execution when the lid closes, the Mac loses power/network, somebody forces sleep, or the process is force-killed. With a stable network the controller checks the deadline every five seconds; after waking it notices an expired deadline and cleans up. You can also wrap the command manually:

```bash
caffeinate -i python3 scripts/presentation.py --env-file /private/path/keys.env
```

Do not leave a presentation rental unattended overnight. Two independent watchdogs improve cleanup reliability but cannot guarantee provider billing behavior during host or API outages.

## Host keys and credentials

A fresh rental has no preverified SSH host key. The controller uses OpenSSH `StrictHostKeyChecking=accept-new` once into a dedicated per-rental known-hosts file, then switches setup and tunnel operations to strict verification. This is transparent trust-on-first-use, not an out-of-band fingerprint guarantee. Changed keys are never automatically accepted. Inspect the provider/SSH fingerprint if your threat model requires independent verification.

The Vast token remains only in local controller memory; child command environments omit provider and Hugging Face tokens. No account key is installed on the GPU. Models are public official repositories. The lab's generated text has no shell-execution permissions. The original explorer starts through its agent-blocking wrapper.

## Remote self-deletion and remaining provider limits

Vast documents a provider-injected `CONTAINER_API_KEY` and `CONTAINER_ID` that permit an instance to delete itself. `scripts/remote_deadline.py` uses only this narrow credential, verifies the expected instance ID, records a secret-free armed status file and waits for the absolute deadline. It tries deletion with bounded timeouts/retries, then attempts a stopped-state fallback if deletion remains unavailable. No user account API key is copied to the remote host. The local controller independently backs up and destroys the same owned rental.

The official CLI also has a generic `/commands/schedule_job/` implementation used for scheduled cloud copy, constrained execute and reboot operations. Scheduling an arbitrary DELETE through that external scheduler was not verified and is not claimed here. The remote watchdog is an instance-side process, so a frozen/offline host or failed provider API can still defeat timely deletion; inspect the provider if either watchdog reports failure.

References: [official container self-deletion documentation](https://docs.vast.ai/guides/instances/docker-environment#using-the-cli-from-inside-an-instance), [official Vast CLI source](https://github.com/vast-ai/vast-python/blob/master/vast.py), [create-instance API](https://docs.vast.ai/api-reference/instances/create-instance), [destroy-instance API](https://docs.vast.ai/api-reference/instances/destroy-instance), [instance lifecycle and storage charges](https://docs.vast.ai/guides/instances/manage-instances).

The controller's setup/cleanup failure paths are covered by fake-API tests. A future presentation launch is a real billable rental, not one of those tests. After this script was added, no new rental was created merely to test it.

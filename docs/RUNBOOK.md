# Tomorrow's presentation: rent only when needed

The completed experiment evidence and database backup are saved on this Mac. There is no reason to leave a GPU idle overnight. Keep `outputs/gpu-final/` and the source folder together; generated artifacts and model weights are not stored in Git.

## One command at presentation time

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
4. Rents the selected offer, records its contract ID immediately, attaches your public key, and waits up to ten minutes for SSH.
5. Transfers the source, authentic dataset snapshots, original explorer assets and backed-up experiment database. It excludes `.env*`, `.state`, `.git`, `.venv`, credential key files, logs and local model caches.
6. Installs the tested dependencies using the CUDA image's working PyTorch. It runs ordinary software tests, but **does not launch validation, an automated judge or exhaustive model smoke checks**.
7. Downloads the requested official weights, starts the lab and safe original explorer, and opens loopback SSH forwards. The command above prefetches all three models so the **4B OG Evil** and **27B Evil** demonstrations do not need a large first-use download; Qwen3-8B is available for its separate persona examples. Other recipe/model selections remain available in the lab.

Open **http://127.0.0.1:8787** for the persona lab and **http://127.0.0.1:8791** for the original 3D explorer. Optional `--fast-kernels` installs the tested GCC/G++ and pinned FLA packages before launch. To prefetch every model, add:

```bash
--models Qwen/Qwen3.5-4B Qwen/Qwen3-8B Qwen/Qwen3.8-27B
```

There are no local copies of the roughly 76 GB GPU model cache. A fresh rental downloads the requested models again. Allow at least 15 minutes for the small-model setup and plan **30–60 minutes of lead time when prefetching all three**, depending on image pulls, offer networking, Hugging Face throughput and package downloads. The script's default is only 4B, but the command above explicitly requests all three. Follow `outputs/presentation-setup.log` from another Terminal while setup runs.

## End the presentation and remove the rental

Press **Ctrl+C** in the controller Terminal. It closes only its own tunnel and **destroys its recorded rental**, rather than merely stopping the GPU and retaining paid disk storage. It confirms the instance is absent from the provider listing. Setup errors, normal termination and the wall-clock deadline use the same cleanup path.

If that Terminal has disappeared, run:

```bash
python3 scripts/presentation.py \
  --env-file /Users/juanfigueroa/workspace/johan_kaseya/sleeper_demo/.env --stop
```

`--stop` means destroy this controller's recorded presentation rental. It does not act on arbitrary instances, the older completed experiment rental or unrelated account resources. Ownership state is kept in `.state/presentation.json`; no API key is stored there. If cleanup cannot be confirmed, the script prints an explicit error, retains the instance ID and marks `cleanup_failed`. Open the Vast Instances page and destroy that recorded instance promptly if the API is unavailable.

The default two-hour deadline **includes setup time**. It is enforced by this local process regardless of whether anybody is using the UI. Use a smaller `--duration-hours`, for example `0.5`, for a short demo. Two hours and US$2/hour are hard argument ceilings in this controller. Network/transfer billing and marketplace availability remain provider-dependent.

**Keep the Mac awake and the controller Terminal open.** On macOS the controller automatically runs `caffeinate -i -w PID` to prevent ordinary idle sleep for its own lifetime. This cannot guarantee execution when the lid closes, the Mac loses power/network, somebody forces sleep, or the process is force-killed. With a stable network the controller checks the deadline every five seconds; after waking it notices an expired deadline and cleans up. You can also wrap the command manually:

```bash
caffeinate -i python3 scripts/presentation.py --env-file /private/path/keys.env
```

Do not leave a presentation rental unattended overnight. The helper cannot make a provider-side billing guarantee.

## Host keys and credentials

A fresh rental has no preverified SSH host key. The controller uses OpenSSH `StrictHostKeyChecking=accept-new` once into a dedicated per-rental known-hosts file, then switches setup and tunnel operations to strict verification. This is transparent trust-on-first-use, not an out-of-band fingerprint guarantee. Changed keys are never automatically accepted. Inspect the provider/SSH fingerprint if your threat model requires independent verification.

The Vast token remains only in local controller memory; child command environments omit provider and Hugging Face tokens. No account key is installed on the GPU. Models are public official repositories. The lab's generated text has no shell-execution permissions. The original explorer starts through its agent-blocking wrapper.

## Provider scheduling research and remaining limit

The official Vast CLI source has a generic `/commands/schedule_job/` implementation used for scheduled cloud copy, constrained execute and reboot operations. Its `stop instance` and `destroy instance` commands do not expose scheduling options. The public create-instance API does not document a user-selected automatic-destruction deadline. Support for scheduling an arbitrary DELETE request was not verified, so this controller does **not** claim to have configured a provider-side kill switch or copy a Vast key onto the remote host.

References: [official Vast CLI source](https://github.com/vast-ai/vast-python/blob/master/vast.py), [create-instance API](https://docs.vast.ai/api-reference/instances/create-instance), [destroy-instance API](https://docs.vast.ai/api-reference/instances/destroy-instance), [instance lifecycle and storage charges](https://docs.vast.ai/guides/instances/manage-instances).

The controller's setup/cleanup failure paths are covered by fake-API tests. A future presentation launch is a real billable rental, not one of those tests. After this script was added, no new rental was created merely to test it.

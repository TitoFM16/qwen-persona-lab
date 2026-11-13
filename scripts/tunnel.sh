#!/usr/bin/env bash
# Keep both lab forwards on loopback. Ctrl+C stops only this script's SSH child.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 SSH_HOST SSH_PORT [KNOWN_HOSTS_FILE]" >&2
  exit 2
fi
ssh_host=$1
ssh_port=$2
known_hosts=${3:-.state/known_hosts}
lab_port=${LAB_LOCAL_PORT:-8787}
explorer_port=${EXPLORER_LOCAL_PORT:-8791}
for port in "$ssh_port" "$lab_port" "$explorer_port"; do
  if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "SSH and local ports must be integers between 1 and 65535." >&2
    exit 2
  fi
done
if [ "$lab_port" = "$explorer_port" ]; then
  echo "The lab and explorer need different local ports." >&2
  exit 2
fi
if [ ! -f "$known_hosts" ]; then
  echo "Known-hosts file is missing: $known_hosts" >&2
  echo "Verify the server fingerprint and add it to that file before opening this tunnel." >&2
  exit 2
fi

child_pid=''
error_file=$(mktemp "${TMPDIR:-/tmp}/qwen-tunnel.XXXXXX") || exit 1
cleanup() {
  if [ -n "$child_pid" ]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=''
  fi
  rm -f "$error_file"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ssh_options=(-N -o BatchMode=yes -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3
  -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$known_hosts"
  -p "$ssh_port"
  -L "127.0.0.1:$lab_port:127.0.0.1:8787"
  -L "127.0.0.1:$explorer_port:127.0.0.1:8791")
if [ -n "${SSH_HOST_KEY_ALIAS:-}" ]; then
  ssh_options+=(-o "HostKeyAlias=$SSH_HOST_KEY_ALIAS")
fi

echo "Forwarding lab http://127.0.0.1:$lab_port and explorer http://127.0.0.1:$explorer_port"
echo "Press Ctrl+C to stop this tunnel. Existing listeners will never be killed."
while true; do
  : > "$error_file"
  LC_ALL=C ssh "${ssh_options[@]}" -- "root@$ssh_host" 2> "$error_file" &
  child_pid=$!
  wait "$child_pid"
  ssh_status=$?
  child_pid=''
  cat "$error_file" >&2
  bind_failed=0
  while IFS= read -r message; do
    if [[ "$message" == *"Address already in use"* || "$message" == *"cannot listen to port"* ||
          "$message" == *"Could not request local forwarding"* ]]; then
      bind_failed=1
    fi
  done < "$error_file"
  if [ "$bind_failed" -eq 1 ]; then
    echo "A local forwarding port is occupied. Close your old tunnel, or choose different LAB_LOCAL_PORT and EXPLORER_LOCAL_PORT values." >&2
    echo "No other listener has been stopped. Exiting instead of repeatedly retrying the occupied port." >&2
    exit 1
  fi
  echo "SSH exited with status $ssh_status. Reconnecting in 3 seconds; Ctrl+C stops retries." >&2
  sleep 3
done

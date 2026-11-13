"""Self-delete only this Vast instance at a bounded UTC deadline.

Reads only the provider-injected CONTAINER_API_KEY and CONTAINER_ID. The account
key is never accepted. This independent timer survives a disconnected Mac, but
provider/network failures can still prevent destruction; the Mac watchdog remains
a second safeguard. Stop fallback preserves disk and therefore storage charges.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def parse_deadline(until, now=None):
    now = time.time() if now is None else now
    parsed = datetime.datetime.fromisoformat(until.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Deadline must include a timezone')
    deadline = parsed.timestamp()
    if not 0 < deadline - now <= 12 * 3600:
        raise ValueError('Deadline must be in the future and no more than12 hours away')
    return deadline


def provider_credentials(instance_id, environ=None, proc_path='/proc/1/environ'):
    if type(instance_id) is not int or instance_id <= 0:
        raise ValueError('Expected instance ID must be a positive integer')
    environ = os.environ if environ is None else environ
    # Deliberately never read VAST_AI_API_KEY, VAST_API_KEY or account key files.
    key = environ.get('CONTAINER_API_KEY', '').strip()
    identifiers = [environ.get('CONTAINER_ID', '').strip()]
    if not key or not identifiers[0]:
        try:
            for record in Path(proc_path).read_bytes().split(b'\0'):
                name, separator, value = record.partition(b'=')
                if not separator: continue
                if name == b'CONTAINER_API_KEY' and not key:
                    key = value.decode('utf-8').strip()
                elif name == b'CONTAINER_ID':
                    identifiers.append(value.decode('utf-8').strip())
        except (OSError, UnicodeError):
            pass
    for identifier in filter(None, identifiers):
        if not identifier.isdecimal() or int(identifier) != instance_id:
            raise ValueError('Provider CONTAINER_ID does not match expected instance; refusing all actions')
    if not key:
        raise ValueError('Provider CONTAINER_API_KEY is unavailable; no account-key fallback is allowed')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def provider_request(instance_id, key, method):
    data = json.dumps({'state': 'stopped'}).encode() if method == 'PUT' else None
    request = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{instance_id}/',
        data=data, method=method,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Reject redirects rather than forwarding a credential to another origin.
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=15) as response:
        result = json.load(response)
    return isinstance(result, dict) and result.get('success') is True


def write_status(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle: json.dump(state, handle, sort_keys=True)
    os.replace(temporary, path)


def run(instance_id, until, status_file, *, environ=None, proc_path='/proc/1/environ',
        clock=time.time, sleeper=time.sleep, api=provider_request, attempts=5):
    if not 3 <= attempts <= 5: raise ValueError('Delete attempts must be bounded to3..5')
    deadline = parse_deadline(until, clock())
    key = provider_credentials(instance_id, environ, proc_path)
    state = {'instance_id': instance_id, 'pid': os.getpid(), 'state': 'armed',
             'deadline_unix': deadline, 'deadline_utc': datetime.datetime.fromtimestamp(
                 deadline, datetime.timezone.utc).isoformat(),
             'armed_at_unix': clock(), 'updated_at': clock(), 'delete_attempts': 0,
             'stop_requested': False}
    write_status(status_file, state)
    # Bounded increments avoid relying on one long sleep across clock changes.
    while True:
        remaining = deadline - clock()
        if remaining <= 0: break
        sleeper(min(30, remaining))
    for attempt in range(1, attempts + 1):
        state.update(state='delete_attempt', delete_attempts=attempt, updated_at=clock())
        write_status(status_file, state)
        try:
            if api(instance_id, key, 'DELETE'):
                state.update(state='destroy_requested', updated_at=clock())
                write_status(status_file, state)
                return 0
            state['last_error'] = 'Provider did not confirm deletion'
        except Exception as exc:
            # Error strings and response bodies may reflect credentials. Log only
            # class/status, never request headers, key values or exception text.
            state['last_error'] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError): state['last_http_status'] = exc.code
        write_status(status_file, state)
        if attempt < attempts: sleeper(2)
    state.update(state='stop_fallback_attempt', updated_at=clock(),
                 warning='Deletion failed; stop fallback preserves disk and continuing storage charges.')
    write_status(status_file, state)
    try:
        stopped = api(instance_id, key, 'PUT')
    except Exception as exc:
        stopped = False
        state['stop_error'] = type(exc).__name__
    state.update(state='stop_requested' if stopped else 'failed', stop_requested=bool(stopped), updated_at=clock())
    write_status(status_file, state)
    return 2 if stopped else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance-id', required=True, type=int)
    parser.add_argument('--until', required=True)
    parser.add_argument('--status-file', default='.state/remote-deadline.json')
    args = parser.parse_args()
    try:
        return run(args.instance_id, args.until, args.status_file)
    except ValueError as exc:
        # Only our fixed validation descriptions are safe to report. In
        # particular, never print arbitrary parser input or exception text.
        detail = str(exc)
        safe_prefixes = ('Provider CONTAINER_', 'Expected instance ID', 'Deadline must', 'Delete attempts must')
        print('Remote deadline could not arm: ' + (detail if detail.startswith(safe_prefixes)
              else 'Invalid deadline or input'), flush=True)
        return 1
    except OSError:
        print('Remote deadline could not arm: cannot write safe status file', flush=True)
        return 1


if __name__ == '__main__': raise SystemExit(main())

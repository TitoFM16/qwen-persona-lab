import datetime
import json
from pathlib import Path
import tempfile
import unittest
from scripts.remote_deadline import parse_deadline, provider_credentials, run


class Clock:
    def __init__(self): self.now = 1_000_000
    def time(self): return self.now
    def sleep(self, seconds): self.now += seconds
    def until(self, delta=60):
        return datetime.datetime.fromtimestamp(self.now + delta, datetime.timezone.utc).isoformat()


class RemoteDeadlineTests(unittest.TestCase):
    def test_deadline_and_identity_fail_closed(self):
        clock = Clock()
        for until in [clock.until(-1), clock.until(12 * 3600 + 1), '2026-11-13T21:00:00']:
            with self.assertRaises(ValueError): parse_deadline(until, clock.time())
        self.assertEqual(parse_deadline(clock.until(), clock.time()), clock.time() + 60)
        for instance in [0, -1, True]:
            with self.assertRaises(ValueError): provider_credentials(instance, {}, '/absent')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            provider_credentials(123, {'CONTAINER_ID': '456', 'CONTAINER_API_KEY': 'secret'}, '/absent')
        with self.assertRaisesRegex(ValueError, 'no account-key fallback'):
            provider_credentials(123, {'VAST_AI_API_KEY': 'account-secret'}, '/absent')

    def test_narrow_proc_fallback_does_not_use_account_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory) / 'environ'
            proc.write_bytes(b'OTHER_SECRET=unrelated\0CONTAINER_ID=123\0CONTAINER_API_KEY=instance-secret\0VAST_API_KEY=account-secret\0')
            self.assertEqual(provider_credentials(123, {}, proc), 'instance-secret')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                provider_credentials(124, {}, proc)

    def test_wait_then_delete_success_without_exposing_key(self):
        clock, calls = Clock(), []
        def api(instance, key, method):
            calls.append((instance, method, clock.time())); self.assertEqual(key, 'instance-secret'); return True
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / 'status.json'; until = clock.until()
            result = run(123, until, status, environ={'CONTAINER_ID': '123', 'CONTAINER_API_KEY': 'instance-secret'},
                         clock=clock.time, sleeper=clock.sleep, api=api)
            self.assertEqual(result, 0); self.assertEqual(calls, [(123, 'DELETE', 1_000_060)])
            text = status.read_text(); self.assertNotIn('instance-secret', text)
            self.assertEqual(json.loads(text)['state'], 'destroy_requested')
            self.assertEqual(status.stat().st_mode & 0o777, 0o600)

    def test_bounded_delete_retry_and_storage_warning_on_stop_fallback(self):
        clock, calls = Clock(), []
        def api(instance, key, method):
            calls.append(method)
            if method == 'DELETE': raise RuntimeError('instance-secret MUST NOT BE LOGGED')
            return True
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / 'status.json'
            result = run(123, clock.until(), status, environ={'CONTAINER_API_KEY': 'instance-secret'},
                         proc_path='/absent', clock=clock.time, sleeper=clock.sleep, api=api, attempts=3)
            self.assertEqual(result, 2); self.assertEqual(calls, ['DELETE'] * 3 + ['PUT'])
            text = status.read_text(); state = json.loads(text)
            self.assertNotIn('instance-secret', text)
            self.assertEqual(state['state'], 'stop_requested')
            self.assertIn('storage charges', state['warning'])


if __name__ == '__main__': unittest.main()

import importlib.util
import http.client
import json
import os
import subprocess
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    'recipe_assistant', Path(__file__).resolve().parents[1] / 'scripts' / 'recipe_assistant.py')
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

ORIGIN = 'http://127.0.0.1:8791'
GOOD = {'recipe': {'title': 'Soup', 'steps': ['boil']}, 'question': 'Can I halve it?'}


def cli_ok(text='Yes.', model='claude-haiku-5-5'):
    return json.dumps({'type': 'result', 'is_error': False, 'result': text, 'modelUsage': {model: {}}})


class FakeRunner:
    def __init__(self, stdout=None, returncode=0, exc=None, block=None):
        self.stdout = cli_ok() if stdout is None else stdout
        self.returncode, self.exc, self.block = returncode, exc, block
        self.calls = []
        self.entered = threading.Event()

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw, os.path.isdir(kw.get('cwd', ''))))
        self.entered.set()
        if self.block:
            self.block.wait(5)
        if self.exc:
            raise self.exc
        return SimpleNamespace(stdout=self.stdout, stderr='SECRET-STDERR', returncode=self.returncode)


class Base(unittest.TestCase):
    def start(self, runner=None, **kw):
        self.runner = runner or FakeRunner()
        self.now = [0.0]
        bridge = ra.Bridge(port=0, runner=self.runner, claude_bin='/fake/claude',
                           clock=lambda: self.now[0], **kw)
        self.srv = ra.make_server(bridge)
        self.bridge = bridge
        self.port = bridge.port
        t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(lambda: (self.srv.shutdown(), self.srv.server_close()))

    def req(self, method, path, body=None, headers=None, host='default', raw=None):
        h = {'Host': 'default' == host and '127.0.0.1:%d' % self.port or host}
        h.update(headers or {})
        c = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        hs = dict(h)
        if data is not None and 'Content-Length' not in hs and 'Content-Length-None' not in hs:
            hs['Content-Length'] = str(len(data))
        hs.pop('Content-Length-None', None)
        for k, v in hs.items():
            c.putheader(k, v)
        c.endheaders(data)
        r = c.getresponse()
        txt = r.read()
        c.close()
        try:
            js = json.loads(txt) if txt else None
        except ValueError:
            js = None
        return r.status, js, r

    def post(self, body=GOOD, headers=None, **kw):
        h = {'Origin': ORIGIN, 'X-Recipe-Assistant': '1', 'Content-Type': 'application/json'}
        h.update(headers or {})
        return self.req('POST', '/api/ask', body, h, **kw)


class SecurityTests(Base):
    def setUp(self):
        self.start()

    def test_health(self):
        s, js, _ = self.req('GET', '/health')
        self.assertEqual((s, js), (200, {'status': 'ok', 'model': 'claude-haiku-5-5', 'tools_enabled': False}))

    def test_health_localhost_host_and_origin(self):
        s, js, r = self.req('GET', '/health', headers={'Origin': 'http://localhost:8787'}, host='localhost:%d' % self.port)
        self.assertEqual(s, 200)
        self.assertEqual(r.getheader('Access-Control-Allow-Origin'), 'http://localhost:8787')

    def test_health_bad_origin_and_host(self):
        self.assertEqual(self.req('GET', '/health', headers={'Origin': 'http://evil.example'})[0], 403)
        self.assertEqual(self.req('GET', '/health', host='evil.example')[0], 403)
        self.assertEqual(self.req('GET', '/health', host='127.0.0.1:1')[0], 403)

    def test_bad_host_post(self):
        self.assertEqual(self.post(host='evil.example:%d' % self.port)[0], 403)
        self.assertEqual(self.runner.calls, [])

    def test_origin_required_and_exact(self):
        h = {'X-Recipe-Assistant': '1', 'Content-Type': 'application/json'}
        self.assertEqual(self.req('POST', '/api/ask', GOOD, h)[0], 403)
        for bad in ('http://127.0.0.1:8792', 'https://127.0.0.1:8791', 'null', 'http://localhost', 'http://127.0.0.1:8791/', '*'):
            self.assertEqual(self.post(headers={'Origin': bad})[0], 403, bad)
        for good in ('http://127.0.0.1:8791', 'http://localhost:8791', 'http://127.0.0.1:8787', 'http://localhost:8787'):
            s, _, r = self.post(headers={'Origin': good})
            self.assertEqual(s, 200, good)
            self.assertEqual(r.getheader('Access-Control-Allow-Origin'), good)
        self.assertEqual(len(self.runner.calls), 4)

    def test_no_wildcard_cors(self):
        _, _, r = self.post()
        self.assertNotEqual(r.getheader('Access-Control-Allow-Origin'), '*')
        _, _, r = self.req('OPTIONS', '/api/ask', headers={'Origin': ORIGIN})
        self.assertNotEqual(r.getheader('Access-Control-Allow-Origin'), '*')

    def test_options_preflight(self):
        s, _, r = self.req('OPTIONS', '/api/ask', headers={
            'Origin': ORIGIN, 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type, x-recipe-assistant'})
        self.assertEqual(s, 204)
        self.assertEqual(r.getheader('Access-Control-Allow-Origin'), ORIGIN)
        self.assertEqual(r.getheader('Access-Control-Allow-Methods'), 'POST')
        self.assertEqual(r.getheader('Access-Control-Allow-Headers'), 'Content-Type, X-Recipe-Assistant')

    def test_options_rejections(self):
        self.assertEqual(self.req('OPTIONS', '/api/ask')[0], 403)  # no Origin
        self.assertEqual(self.req('OPTIONS', '/api/ask', headers={'Origin': 'http://evil.example'})[0], 403)
        self.assertEqual(self.req('OPTIONS', '/api/ask', headers={'Origin': ORIGIN, 'Access-Control-Request-Method': 'DELETE'})[0], 403)
        self.assertEqual(self.req('OPTIONS', '/api/ask', headers={'Origin': ORIGIN, 'Access-Control-Request-Headers': 'authorization'})[0], 403)
        self.assertEqual(self.req('OPTIONS', '/api/ask', headers={'Origin': ORIGIN}, host='evil:1')[0], 403)

    def test_custom_header_required(self):
        h = {'Origin': ORIGIN, 'Content-Type': 'application/json'}
        self.assertEqual(self.req('POST', '/api/ask', GOOD, h)[0], 403)
        h['X-Recipe-Assistant'] = '0'
        self.assertEqual(self.req('POST', '/api/ask', GOOD, h)[0], 403)
        self.assertEqual(self.runner.calls, [])

    def test_content_type_required(self):
        self.assertEqual(self.post(headers={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.post(headers={'Content-Type': 'application/json; charset=utf-8'})[0], 200)

    def test_length_rules(self):
        self.assertEqual(self.post(headers={'Content-Length-None': '1'})[0], 411)
        s, js, _ = self.post(headers={'Transfer-Encoding': 'chunked', 'Content-Length-None': '1'}, raw=b'0\r\n\r\n')
        self.assertEqual(s, 400)
        self.assertEqual(self.post(headers={'Content-Length': 'abc'}, raw=b'{}')[0], 400)
        big = {'recipe': {'x': 'a' * ra.MAX_BODY}, 'question': 'q'}
        self.assertEqual(self.post(big)[0], 413)
        self.assertEqual(self.runner.calls, [])

    def test_payload_validation(self):
        bad = [
            {'recipe': [], 'question': 'q'},
            {'recipe': 'x', 'question': 'q'},
            {'question': 'q'},
            {'recipe': {}, 'question': ''},
            {'recipe': {}, 'question': 5},
            {'recipe': {}, 'question': 'q' * 4001},
            {'recipe': {}, 'question': 'q', 'history': 'x'},
            {'recipe': {}, 'question': 'q', 'history': [{'role': 'user', 'content': 'a'}] * 11},
            {'recipe': {}, 'question': 'q', 'history': [{'role': 'system', 'content': 'a'}]},
            {'recipe': {}, 'question': 'q', 'history': [{'role': 'user', 'content': 'a' * 4001}]},
            {'recipe': {}, 'question': 'q', 'history': [{'role': 'user'}]},
            {'recipe': {}, 'question': 'q', 'history': ['x']},
            [1, 2],
        ]
        for b in bad:
            s, js, _ = self.post(b)
            self.assertEqual(s, 400, b)
            self.assertIn('error', js)
        self.assertEqual(self.post(raw=b'{not json')[0], 400)
        self.assertEqual(self.runner.calls, [])

    def test_boundary_valid(self):
        body = {'recipe': {}, 'question': 'q' * 4000,
                'history': [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'a' * 100} for i in range(10)]}
        self.assertEqual(self.post(body)[0], 200)

    def test_unknown_path_and_methods(self):
        self.assertEqual(self.req('GET', '/api/ask')[0], 404)
        self.assertEqual(self.req('GET', '/other')[0], 404)
        self.assertEqual(self.req('PUT', '/api/ask')[0], 405)
        self.assertEqual(self.post_path('/nope')[0], 404)

    def post_path(self, path):
        h = {'Origin': ORIGIN, 'X-Recipe-Assistant': '1', 'Content-Type': 'application/json'}
        return self.req('POST', path, GOOD, h)


class ErrorCorsTests(Base):
    def setUp(self):
        self.start()

    def test_allowed_origin_error_has_cors(self):
        s, js, r = self.post(headers={'Content-Type': 'text/plain'})
        self.assertEqual(s, 415)
        self.assertEqual(r.getheader('Access-Control-Allow-Origin'), ORIGIN)
        self.assertEqual(r.getheader('Vary'), 'Origin')
        s, js, r = self.post(body={'recipe': 1})
        self.assertEqual(s, 400)
        self.assertEqual(r.getheader('Access-Control-Allow-Origin'), ORIGIN)
        self.assertTrue(js['error'])

    def test_bad_origin_error_has_no_cors(self):
        s, _, r = self.post(headers={'Origin': 'http://evil.example'})
        self.assertEqual(s, 403)
        self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))

    def test_bad_host_error_has_no_cors(self):
        s, _, r = self.post(host='evil.example')
        self.assertEqual(s, 403)
        self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))
        s, _, r = self.post(host='evil.example', body={'recipe': 1})
        self.assertIsNone(r.getheader('Access-Control-Allow-Origin'))

    def test_slow_body_times_out_408(self):
        old = ra.Handler.io_timeout
        ra.Handler.io_timeout = 0.3
        self.addCleanup(setattr, ra.Handler, 'io_timeout', old)
        import socket
        c = socket.create_connection(('127.0.0.1', self.port), timeout=5)
        self.addCleanup(c.close)
        c.sendall(('POST /api/ask HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nOrigin: %s\r\n'
                   'X-Recipe-Assistant: 1\r\nContent-Type: application/json\r\n'
                   'Content-Length: 100\r\n\r\n{' % (self.port, ORIGIN)).encode())
        data = b''
        while True:
            chunk = c.recv(4096)
            if not chunk:
                break
            data += chunk
        text = data.decode()
        self.assertIn(' 408 ', text.split('\r\n', 1)[0])
        self.assertIn('Connection: close', text)
        self.assertIn('Access-Control-Allow-Origin: ' + ORIGIN, text)
        self.assertEqual(self.runner.calls, [])


class ExecutionTests(Base):
    def test_success_and_contract(self):
        self.start(FakeRunner(cli_ok('Halve everything.', 'claude-haiku-5-5-20260101')))
        body = dict(GOOD, history=[{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}])
        s, js, _ = self.post(body)
        self.assertEqual(s, 200)
        self.assertEqual(js, {'answer': 'Halve everything.', 'model': 'claude-haiku-5-5-20260101', 'tools_enabled': False})
        argv, kw, cwd_existed = self.runner.calls[0]
        sent = json.loads(kw['input'])
        self.assertEqual(sent['recipe'], GOOD['recipe'])
        self.assertEqual(sent['question'], GOOD['question'])
        self.assertEqual(sent['history'], body['history'])
        self.assertIn('UNTRUSTED', sent['notice'])

    def test_model_falls_back_to_configured(self):
        self.start(FakeRunner(json.dumps({'is_error': False, 'result': 'ok'})))
        self.assertEqual(self.post()[1]['model'], 'claude-haiku-5-5')

    def test_fixed_argv_flags_and_no_shell(self):
        self.start()
        self.post()
        argv, kw, _ = self.runner.calls[0]
        self.assertIsInstance(argv, list)
        self.assertIs(kw['shell'], False)
        self.assertEqual(argv[0], '/fake/claude')
        for flag in ('-p', '--safe-mode', '--restricted', '--strict-mcp-config', '--disable-slash-commands',
                     '--no-session-persistence', '--system-prompt', '--max-budget-usd'):
            self.assertIn(flag, argv)
        self.assertNotIn('--bare', argv)
        self.assertEqual(argv[argv.index('--tools') + 1], '')
        self.assertEqual(argv[argv.index('--output-format') + 1], 'json')
        self.assertEqual(argv[argv.index('--model') + 1], 'claude-haiku-5-5')
        self.assertEqual(argv[argv.index('--max-budget-usd') + 1], '0.10')
        self.assertEqual(kw['timeout'], 60)
        # user content never reaches argv
        self.assertFalse(any('Can I halve' in a or 'Soup' in a for a in argv))

    def test_neutral_fresh_cwd(self):
        self.start()
        self.post()
        self.post()
        c1, c2 = self.runner.calls[0][1]['cwd'], self.runner.calls[1][1]['cwd']
        self.assertNotEqual(c1, c2)
        self.assertFalse(os.path.exists(c1))  # cleaned up
        self.assertNotEqual(Path(c1).resolve(), Path.cwd().resolve())
        self.assertTrue(self.runner.calls[0][2])  # existed during run
        self.assertEqual(os.listdir(os.path.dirname(c1)).count(os.path.basename(c1)), 0)

    def test_env_drops_api_keys_keeps_rest(self):
        env = ra.child_environment({'PATH': '/bin', 'HOME': '/h', 'ANTHROPIC_API_KEY': 'k', 'OPENAI_API_KEY': 'k2', 'CLAUDE_CODE_OAUTH_TOKEN': 't'})
        self.assertEqual(env, {'PATH': '/bin', 'HOME': '/h', 'CLAUDE_CODE_OAUTH_TOKEN': 't'})

    def test_busy_429(self):
        gate = threading.Event()
        self.start(FakeRunner(block=gate))
        out = []
        t = threading.Thread(target=lambda: out.append(self.post()))
        t.start()
        self.assertTrue(self.runner.entered.wait(5))
        s, js, _ = self.post()
        self.assertEqual(s, 429)
        self.assertIn('busy', js['error'])
        gate.set()
        t.join(10)
        self.assertEqual(out[0][0], 200)
        self.assertEqual(self.post()[0], 200)  # semaphore released
        self.assertEqual(len(self.runner.calls), 2)

    def test_rate_limit_with_test_clock(self):
        self.start(rate_limit=2, rate_window=60)
        self.assertEqual(self.post()[0], 200)
        self.assertEqual(self.post()[0], 200)
        self.assertEqual(self.post()[0], 429)
        self.now[0] = 61
        self.assertEqual(self.post()[0], 200)

    def test_invalid_requests_do_not_consume_rate(self):
        self.start(rate_limit=1)
        self.post({'recipe': 1, 'question': 'q'})
        self.assertEqual(self.post()[0], 200)

    def test_timeout_504(self):
        self.start(FakeRunner(exc=subprocess.TimeoutExpired('claude', 60)))
        s, js, _ = self.post()
        self.assertEqual(s, 504)
        self.assertEqual(self.post_after_ok(), 200)

    def post_after_ok(self):
        self.runner.exc = None
        return self.post()[0]

    def test_cli_missing_503(self):
        self.start(FakeRunner(exc=FileNotFoundError('claude')))
        s, js, _ = self.post()
        self.assertEqual(s, 503)
        self.assertIn('unavailable', js['error'])

    def test_auth_error_clear_other_errors_generic(self):
        self.start(FakeRunner(json.dumps({'is_error': True, 'result': 'Not logged in. Please log in'})))
        s, js, _ = self.post()
        self.assertEqual(s, 503)
        self.assertIn('authenticated', js['error'])
        self.runner.stdout = json.dumps({'is_error': True, 'result': 'secret internal detail /home/x'})
        s, js, _ = self.post()
        self.assertEqual(s, 502)
        self.assertNotIn('secret', json.dumps(js))

    def test_nonzero_exit_and_garbage_do_not_leak_stderr(self):
        self.start(FakeRunner('', returncode=1))
        s, js, _ = self.post()
        self.assertEqual(s, 502)
        self.assertNotIn('SECRET-STDERR', json.dumps(js))
        self.runner.stdout, self.runner.returncode = 'not json', 0
        self.assertEqual(self.post()[0], 502)
        self.runner.stdout = json.dumps({'is_error': False, 'result': '  '})
        self.assertEqual(self.post()[0], 502)

    def test_tool_use_refused(self):
        out = json.dumps({'is_error': False, 'result': 'x', 'content': [{'type': 'tool_use', 'name': 'Bash'}]})
        self.start(FakeRunner(out))
        s, js, _ = self.post()
        self.assertEqual(s, 502)
        self.assertNotIn('answer', js)

    def test_answer_capped(self):
        self.start(FakeRunner(cli_ok('a' * 50000)))
        self.assertEqual(len(self.post()[1]['answer']), 16000)

    def test_answer_never_executed_and_no_logging(self):
        # request content must not appear in any handler log output
        self.start(FakeRunner(cli_ok('$(touch /tmp/pwned)')))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            self.post(dict(GOOD, question='TOPSECRETQUESTION'))
            self.req('GET', '/nope')
        self.assertNotIn('TOPSECRETQUESTION', buf.getvalue())
        self.assertEqual(buf.getvalue(), '')
        self.assertFalse(os.path.exists('/tmp/pwned'))


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(ra.DEFAULT_MODEL, 'claude-haiku-5-5')
        self.assertEqual((ra.DEFAULT_PORT, ra.DEFAULT_TIMEOUT, ra.MAX_TIMEOUT), (8788, 60, 120))

    def test_timeout_bounded(self):
        for bad in (0, -1, 121):
            with self.assertRaises(ValueError):
                ra.Bridge(timeout=bad)
            with self.assertRaises(SystemExit):
                ra.main(['--timeout', str(bad)])

    def test_main_parses_and_serves(self):
        made = {}

        class FakeSrv:
            server_address = ('127.0.0.1', 9)
            def serve_forever(self): raise KeyboardInterrupt
            def server_close(self): made['closed'] = True

        orig = ra.make_server
        ra.make_server = lambda b, host='127.0.0.1': made.update(bridge=b) or FakeSrv()
        try:
            self.assertEqual(ra.main(['--port', '9', '--model', 'm', '--timeout', '5']), 0)
        finally:
            ra.make_server = orig
        b = made['bridge']
        self.assertEqual((b.port, b.model, b.timeout), (9, 'm', 5))
        self.assertTrue(made['closed'])

    def test_binds_loopback_only(self):
        srv = ra.make_server(ra.Bridge(port=0))
        try:
            self.assertEqual(srv.server_address[0], '127.0.0.1')
        finally:
            srv.server_close()


if __name__ == '__main__':
    unittest.main()

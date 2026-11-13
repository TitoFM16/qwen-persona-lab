import asyncio
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / 'scripts' / 'explorer_safe.py'
OVERLAY = ROOT / 'scripts' / 'explorer_recipe_overlay.js'
PAYLOAD = ROOT / 'lab' / 'static' / 'recipe_payload.js'
UPSTREAM_APP = ROOT / 'vendor/neuron-explorer/explorer/static/app.js'
NODE = shutil.which('node')

_spec = importlib.util.spec_from_file_location('explorer_safe_under_test', SAFE)
safe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safe)  # must not import the GPU/explorer server


def req(path, method='GET'):
    return SimpleNamespace(url=SimpleNamespace(path=path), method=method)


def run(coro):
    return asyncio.run(coro)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest('fastapi not installed')
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / 'app.js').write_text('import {x} from "./x.js";\nconst original = 1;\n')
        self.mw = safe.make_middleware(self.tmp, OVERLAY)
        self.passed = []

        async def call_next(request):
            self.passed.append(request.url.path)
            return SimpleNamespace(status_code=200, body=b'upstream')
        self.call_next = call_next

    def call(self, path, method='GET'):
        return run(self.mw(req(path, method), self.call_next))

    def test_import_does_not_load_explorer_server(self):
        self.assertFalse(hasattr(safe, 'server'))

    def test_every_agent_route_is_forbidden_for_all_methods(self):
        for method in ('GET', 'POST', 'DELETE', 'PUT'):
            for path in ('/api/agent', '/api/agent/send', '/api/agent/events', '/api/agents', '//api/agent/send', '/API/AGENT/chats'):
                response = self.call(path, method)
                self.assertEqual(response.status_code, 403, (method, path))
        self.assertEqual(self.passed, [])

    def test_non_agent_api_passes_through(self):
        self.assertEqual(self.call('/api/index').status_code, 200)
        self.assertEqual(self.passed, ['/api/index'])

    def test_app_js_is_upstream_plus_late_dynamic_import_tail(self):
        response = self.call('/app.js')
        body = response.body.decode()
        original = (self.tmp / 'app.js').read_text()
        self.assertTrue(body.startswith(original))  # upstream source first, untouched
        tail = body[len(original):]
        self.assertIn("import('/explorer_recipe_overlay.js')", tail)
        self.assertNotRegex(tail, r'(?m)^\s*import\s+[\w{*]')  # no hoisted static import
        for name in ('openNeuron', 'openClaude', 'closeEvents', 'map', 'state', 'markAgent'):
            self.assertIn(name, tail)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertIn('javascript', response.media_type)

    def test_overlay_is_served_before_upstream_static_mount(self):
        response = self.call('/explorer_recipe_overlay.js')
        self.assertEqual(response.body, OVERLAY.read_bytes())
        self.assertEqual(self.passed, [])

    def test_only_exact_app_js_get_is_intercepted(self):
        self.call('/app.js', 'POST')
        self.call('/map3d.js')
        self.assertEqual(self.passed, ['/app.js', '/map3d.js'])

    @unittest.skipUnless(NODE and UPSTREAM_APP.is_file(), 'node or vendor explorer missing')
    def test_real_upstream_served_script_parses_and_vendor_file_is_unchanged(self):
        before = UPSTREAM_APP.read_bytes()
        served = safe.build_app_js()
        self.assertEqual(UPSTREAM_APP.read_bytes(), before)
        self.assertTrue(served.startswith(before))
        for name in ('openNeuron', 'openClaude', 'setTab', 'closeEvents', 'agentMarked'):
            self.assertRegex(before.decode(), name)  # lexical names the adapter relies on still exist upstream
        out = self.tmp / 'served.mjs'
        out.write_bytes(served)
        done = subprocess.run([NODE, '--check', str(out)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


HARNESS = r"""
// `node -e SCRIPT a b` -> process.argv is [node, a, b] (script args start at index 1)
const [, payloadPath, overlayPath] = process.argv;
const P = require(payloadPath), O = require(overlayPath);
const out = {}; const run = (k, f) => { try { out[k] = { ok: f() }; } catch (e) { out[k] = { error: e.message }; } };
const comp = (l, n, d) => ({ layer: l, neuron: n, dose: d });
const rec = (model, components, extra) => Object.assign({ id: 'r1', name: 'Calm', model, components, method: { kind: 'x' },
  normalization: 'neuronary', source: 'starter', status: 'pending', provenance: { url: 'u' } }, extra || {});
const four = 'Qwen/Qwen3.5-4B', eight = 'Qwen/Qwen3-8B', big = 'Qwen/Qwen3.8-27B';
const mk = (m, cs) => () => P.buildFragment(rec(m, cs), cs, '3d', 'http://127.0.0.1:8790');

const original = [comp(1, 2, 0.5), comp(3, 4, -1.25), comp(5, 6, 2)];
const edited = [comp(1, 2, 0.5), comp(3, 4, -9), comp(5, 6, 2), comp(7, 8, -0.05)];
run('roundtrip', () => { const f = P.buildFragment(rec(four, original), edited, '3d', 'http://127.0.0.1:8790'); const p = O.parseFragment(f);
  return { len: f.length, comps: p.recipe.components, intent: p.intent, origin: p.labOrigin, name: p.recipe.name, model: p.recipe.model,
    norm: p.recipe.normalization, source: p.recipe.source, prov: p.recipe.provenance, method: p.recipe.method, id: p.recipe.id, marks: O.marksFor(p.recipe), notes: O.notesFor(p.recipe) }; });
run('layer32_4b', mk(four, [comp(32, 0, 1)]));
run('neuron9216_4b', mk(four, [comp(0, 9216, 1)]));
run('edge_4b', mk(four, [comp(31, 9215, 1)]));
run('edge_8b', mk(eight, [comp(35, 12287, -1)]));
run('layer36_8b', mk(eight, [comp(36, 0, 1)]));
run('edge_27b', mk(big, [comp(63, 17407, -1)]));
run('layer64_27b', mk(big, [comp(64, 0, 1)]));
run('neuron17408_27b', mk(big, [comp(0, 17408, 1)]));
run('nan', mk(four, [comp(1, 1, NaN)]));
run('inf', mk(four, [comp(1, 1, Infinity)]));
run('string_dose', mk(four, [comp(1, 1, '1')]));
run('float_layer', mk(four, [comp(1.5, 1, 1)]));
run('unknown_model', mk('Qwen/other', [comp(0, 0, 1)]));
run('empty', mk(four, []));
const many = n => Array.from({ length: n }, (_, i) => comp(i % 32, i, -0.123456789 * (i + 1)));
run('many_400', () => { const cs = many(400); try { const f = P.buildFragment(rec(four, cs), cs, 'claude', 'http://127.0.0.1:8790'); return { len: f.length, n: O.parseFragment(f).recipe.components.length }; } catch (e) { return { rejected: e.message }; } });
run('too_many', mk(four, many(401)));
run('huge_meta_bounded', () => { const cs = many(10); const f = P.buildFragment(rec(four, cs, { provenance: { blob: 'x'.repeat(200000) }, method: { blob: 'y'.repeat(200000) }, source: 'z'.repeat(200000), name: 'n'.repeat(5000) }), cs, '3d', 'http://127.0.0.1:8790');
  const p = O.parseFragment(f); return { len: f.length, n: p.recipe.components.length, name: p.recipe.name.length, source: p.recipe.source.length }; });
run('bad_version', () => O.parseFragment(O.PREFIX + encodeURIComponent(JSON.stringify({ v: 2, intent: '3d', recipe: rec(four, original) }))));
run('bad_intent', () => O.parseFragment(O.PREFIX + encodeURIComponent(JSON.stringify({ v: 1, intent: 'run', recipe: rec(four, original) }))));
run('tamper_bounds', () => O.parseFragment(O.PREFIX + encodeURIComponent(JSON.stringify({ v: 1, intent: '3d', recipe: rec(four, [comp(40, 1, 1)]) }))));
run('not_ours', () => O.parseFragment('#L19/N7176'));
run('foreign_origin', () => O.parseFragment(P.buildFragment(rec(four, original), original, '3d', 'https://evil.example')).labOrigin);
run('non4b_flag', () => [O.isExplorerModel({ model: eight }), O.isExplorerModel({ model: four })]);

// app.js builds an "effective" copy carrying the CURRENT normalization; source comparison keeps the edited flag honest
run('norm_effective', () => {
  const sel = rec(four, original, { normalization: 'neuronary', method: { kind: 'x', normalization: 'neuronary' } });
  const eff = Object.assign({}, sel, { normalization: 'layer_median', method: Object.assign({}, sel.method, { normalization: 'layer_median' }), source_normalization: 'neuronary' });
  const p = O.parseFragment(P.buildFragment(eff, original, '3d', 'http://127.0.0.1:8787')).recipe;
  const same = Object.assign({}, sel, { source_normalization: 'neuronary' });
  const q = O.parseFragment(P.buildFragment(same, original, '3d', 'http://127.0.0.1:8787')).recipe;
  return { norm: p.normalization, mnorm: p.method.normalization, edited: p.provenance.edited_in_lab, srcnorm: p.provenance.source_normalization,
    unchangedEdited: q.provenance.edited_in_lab, unchangedNorm: q.normalization };
});
run('raw_urls', () => {
  const sel = rec(four, original, { provenance: undefined, raw: { source_url: 'https://example.org/s', recipe_url: 'https://example.org/r', url: 'u'.repeat(5000), huge: 'x'.repeat(100000) } });
  const f = P.buildFragment(sel, original, '3d', 'http://127.0.0.1:8787'); const pr = O.parseFragment(f).recipe.provenance;
  return { keys: Object.keys(pr).sort(), s: pr.source_url, r: pr.recipe_url, ulen: pr.url.length, len: f.length };
});

const tag = i => 'é'.repeat(3990) + '#' + i;
const hist10 = Array.from({ length: 10 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: tag(i) }));
const bytes = b => Buffer.byteLength(JSON.stringify(b), 'utf8');
run('body_trim', () => { const r = O.validateRecipe(rec(four, many(100))); const b = O.buildAskBody(r, '€'.repeat(4000), hist10);
  return { bytes: bytes(b), chars: JSON.stringify(b).length, ncomp: b.recipe.components.length, tags: (b.history || []).map(m => m.content.slice(-2)), q: b.question.length }; });
run('body_untrimmed_small', () => { const b = O.buildAskBody(O.validateRecipe(rec(four, original)), 'q', hist10.slice(0, 2)); return { n: b.history.length, bytes: bytes(b) }; });
run('body_history_cap', () => { const h = Array.from({ length: 14 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: 'm' + i })); return O.buildAskBody({ components: [] }, 'q', h).history.map(m => m.content); });
run('body_drop_all', () => { const b = O.buildAskBody({ name: 'x'.repeat(32000), components: [] }, 'q', hist10.slice(0, 2)); return { hasHistory: 'history' in b, bytes: bytes(b) }; });
run('body_recipe_too_big', () => O.buildAskBody({ name: 'x'.repeat(33000), components: [] }, 'q', []));

const calls = [];
const okFetch = async (url, init) => { calls.push({ url, init }); return { ok: true, status: 200, json: async () => ({ answer: 'hi', model: 'claude-haiku-5-5', tools_enabled: false }) }; };
(async () => {
  const r = rec(four, original);
  try { const a = await O.ask(r, ' why? ', [{ role: 'user', content: 'a' }, { role: 'system', content: 'x' }, { role: 'assistant', content: 'b' }], okFetch);
    out.ask = { ok: { a, url: calls[0].url, headers: calls[0].init.headers, method: calls[0].init.method, body: JSON.parse(calls[0].init.body) } }; } catch (e) { out.ask = { error: e.message }; }
  try { await O.ask(r, 'q', [], async () => ({ ok: true, status: 200, json: async () => ({ answer: 'x', model: 'm', tools_enabled: true }) })); out.tools = { ok: 1 }; } catch (e) { out.tools = { error: e.message }; }
  try { await O.ask(r, 'q', [], async () => { throw new TypeError('fetch failed'); }); out.down = { ok: 1 }; } catch (e) { out.down = { error: e.message, unavailable: !!e.unavailable }; }
  try { await O.ask(r, 'q', [], async () => ({ ok: false, status: 429, json: async () => ({ error: 'busy' }) })); } catch (e) { out.err = { error: e.message }; }
  try { await O.ask(r, '   ', [], okFetch); } catch (e) { out.empty_q = { error: e.message }; }
  console.log(JSON.stringify(out));
})();
"""


@unittest.skipUnless(NODE, 'node not installed')
class PureJsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        done = subprocess.run([NODE, '-e', HARNESS, str(PAYLOAD), str(OVERLAY)], capture_output=True, text=True, timeout=60)
        if done.returncode:
            raise AssertionError(done.stderr)
        cls.r = json.loads(done.stdout)

    def ok(self, key):
        self.assertIn('ok', self.r[key], self.r[key])
        return self.r[key]['ok']

    def err(self, key):
        self.assertIn('error', self.r[key], self.r[key])
        return self.r[key]['error']

    def test_roundtrip_uses_edited_components_and_preserves_all_signs(self):
        v = self.ok('roundtrip')
        self.assertEqual([(c['layer'], c['neuron'], c['dose']) for c in v['comps']], [(1, 2, 0.5), (3, 4, -9), (5, 6, 2), (7, 8, -0.05)])
        self.assertLessEqual(v['len'], 32768)
        self.assertEqual((v['intent'], v['origin'], v['name'], v['model'], v['id']), ('3d', 'http://127.0.0.1:8790', 'Calm', 'Qwen/Qwen3.5-4B', 'r1'))
        self.assertEqual((v['norm'], v['source']), ('neuronary', 'starter'))
        self.assertTrue(v['prov']['edited_in_lab'])
        self.assertEqual(v['method'], {'kind': 'x'})
        self.assertEqual([m['sign'] for m in v['marks']], ['+', '-', '+', '-'])
        self.assertEqual(len(v['marks']), 4)
        self.assertEqual(len(v['notes']), 4)
        self.assertTrue(all(0 < m['w'] <= 1 for m in v['marks']))

    def test_model_bounds(self):
        for key in ('layer32_4b', 'neuron9216_4b', 'layer36_8b', 'layer64_27b', 'neuron17408_27b'):
            self.err(key)
        for key in ('edge_4b', 'edge_8b', 'edge_27b'):
            self.ok(key)

    def test_rejects_non_finite_or_malformed_components(self):
        for key in ('nan', 'inf', 'string_dose', 'float_layer', 'unknown_model', 'empty', 'too_many'):
            self.err(key)

    def test_all_components_kept_or_rejected_never_truncated(self):
        v = self.ok('many_400')
        self.assertTrue(('rejected' in v) or (v['n'] == 400 and v['len'] <= 32768))

    def test_huge_metadata_is_bounded_but_components_survive(self):
        v = self.ok('huge_meta_bounded')
        self.assertLessEqual(v['len'], 32768)
        self.assertEqual(v['n'], 10)
        self.assertLessEqual(v['name'], 200)
        self.assertLessEqual(v['source'], 1000)

    def test_parse_rejects_tampered_or_foreign_links(self):
        self.err('bad_version')
        self.err('bad_intent')
        self.assertIn('out of range', self.err('tamper_bounds'))
        self.assertIsNone(self.ok('not_ours'))
        self.assertEqual(self.ok('foreign_origin'), '')

    def test_only_4b_is_explorer_model(self):
        self.assertEqual(self.ok('non4b_flag'), [False, True])

    def test_payload_describes_current_normalization_and_marks_edit(self):
        v = self.ok('norm_effective')
        self.assertEqual((v['norm'], v['mnorm']), ('layer_median', 'layer_median'))
        self.assertTrue(v['edited'])
        self.assertEqual(v['srcnorm'], 'neuronary')
        self.assertFalse(v['unchangedEdited'])
        self.assertEqual(v['unchangedNorm'], 'neuronary')

    def test_raw_source_urls_are_preserved_bounded_without_whole_raw(self):
        v = self.ok('raw_urls')
        self.assertEqual((v['s'], v['r']), ('https://example.org/s', 'https://example.org/r'))
        self.assertLessEqual(v['ulen'], 1000)
        self.assertNotIn('huge', v['keys'])
        self.assertNotIn('raw', v['keys'])
        self.assertLess(v['len'], 32768)

    def test_ask_body_trims_oldest_history_to_32kib_utf8_keeping_recipe_and_question(self):
        v = self.ok('body_trim')
        self.assertLessEqual(v['bytes'], 32768)
        self.assertGreater(v['bytes'], v['chars'])  # multibyte: bytes, not characters, are what is bounded
        self.assertEqual((v['ncomp'], v['q']), (100, 4000))
        self.assertTrue(v['tags'])
        self.assertLess(len(v['tags']), 10)
        self.assertEqual(v['tags'][-1], '#9')  # newest survives, oldest dropped first
        small = self.ok('body_untrimmed_small')
        self.assertEqual(small['n'], 2)
        self.assertEqual(self.ok('body_history_cap'), ['m4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'])
        dropped = self.ok('body_drop_all')
        self.assertFalse(dropped['hasHistory'])
        self.assertLessEqual(dropped['bytes'], 32768)
        self.assertIn('too large', self.err('body_recipe_too_big'))

    def test_ask_contract(self):
        v = self.ok('ask')
        self.assertEqual(v['url'], 'http://127.0.0.1:8788/api/ask')
        self.assertEqual(v['method'], 'POST')
        self.assertEqual(v['headers'], {'Content-Type': 'application/json', 'X-Recipe-Assistant': '1'})
        self.assertEqual(v['body']['question'], 'why?')
        self.assertEqual(v['body']['history'], [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}])
        self.assertEqual(len(v['body']['recipe']['components']), 3)
        self.assertEqual(v['a'], {'answer': 'hi', 'model': 'claude-haiku-5-5'})

    def test_ask_refuses_tools_and_reports_unavailable_bridge(self):
        self.assertIn('tools are disabled', self.err('tools'))
        down = self.r['down']
        self.assertTrue(down['unavailable'])
        self.assertIn('8788', down['error'])
        self.assertEqual(self.err('err'), 'busy')
        self.err('empty_q')


class StaticContractTests(unittest.TestCase):
    def test_lab_page_wiring(self):
        html = (ROOT / 'lab/static/index.html').read_text()
        self.assertLess(html.index('recipe_payload.js'), html.index('/app.js'))
        for needle in ('id="open3d"', 'id="askClaude"', 'applies separately to the baseline reply', 'ignores the seed', '4096'):
            self.assertIn(needle, html)
        app = (ROOT / 'lab/static/app.js').read_text()
        body = app.split('function openRecipeIn')[1].split('updateRecipeActions();')[0]
        self.assertIn("$('#normalization').value", body)
        self.assertIn('source_normalization:selected.normalization', body)
        self.assertIn('LabRecipePayload.buildFragment(effective,componentInputs()', body)
        self.assertNotIn('buildFragment(selected', body)
        self.assertNotIn("api('/api/compare'", body)

    def test_non4b_context_hides_the_4b_map_and_falls_back_to_lab_8787(self):
        js = OVERLAY.read_text()
        for needle in ('#map-view', '#embed-view', '#list', 'data-tab]:not([data-tab="claude"])', 'body.lab-non4b #lab-boundary', "classList.toggle('lab-non4b'"):
            self.assertIn(needle, js)
        self.assertIn('http://127.0.0.1:8787', js)
        self.assertNotIn('8790', js)
        self.assertNotIn('#tab-claude{display:none', js)  # assistant stays reachable
        self.assertIn("boundary.setAttribute('role', 'alert')", js)  # boundary is announced to assistive tech

    def test_assistant_wording_is_accurate(self):
        js = OVERLAY.read_text()
        self.assertNotIn('a local Haiku on your Mac', js)
        for phrase in ('Haiku runs through the Claude CLI on your Mac', 'sent to Anthropic', 'no GPU-host credentials or tools'):
            self.assertIn(phrase, js)

    def test_overlay_never_uses_innerhtml_or_legacy_agent_api(self):
        js = OVERLAY.read_text()
        self.assertNotIn('innerHTML', js)
        self.assertNotIn('/api/agent', js)
        self.assertNotIn('outerHTML', js)


if __name__ == '__main__':
    unittest.main()

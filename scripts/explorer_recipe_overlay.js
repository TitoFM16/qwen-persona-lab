/* Persona Laboratory overlay for the ORIGINAL 3D explorer. Served by scripts/explorer_safe.py and loaded by a
   dynamic import appended to the upstream app.js at response time; the vendor files are never modified.
   Pure helpers (parse/validate) are node-testable; install() touches the DOM. */
(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LabRecipeOverlay = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const BOUNDS = {
    'Qwen/Qwen3.5-4B': { layers: 32, neurons: 9216 },
    'Qwen/Qwen3-8B': { layers: 36, neurons: 12288 },
    'Qwen/Qwen3.8-27B': { layers: 64, neurons: 17408 },
  };
  const EXPLORER_MODEL = 'Qwen/Qwen3.5-4B';
  const PREFIX = '#lab-recipe=';
  const MAX_FRAGMENT = 32768;
  const BRIDGE = 'http://127.0.0.1:8788/api/ask';
  const MAX_BODY_BYTES = 32768;
  const MAX_HISTORY = 10;
  const DEFAULT_LAB = 'http://127.0.0.1:8787';
  const NON4B_CSS = `
body.lab-non4b #map-view,body.lab-non4b #embed-view,body.lab-non4b #list,body.lab-non4b #list-tab,body.lab-non4b #views,body.lab-non4b #layer,body.lab-non4b #layout-mode,body.lab-non4b #web-mode,body.lab-non4b #web-z,body.lab-non4b #jump,body.lab-non4b #q-mode,body.lab-non4b #lens-q,body.lab-non4b #tabs button[data-tab]:not([data-tab="claude"]){display:none!important}
body.lab-non4b #lab-boundary{display:block}
#lab-boundary{display:none;position:fixed;left:16px;top:64px;right:min(480px,45vw);z-index:25;padding:16px 18px;background:rgba(40,28,10,.96);border:2px solid #ffcf8a;border-radius:8px;color:#ffe3b8;font-size:14px;line-height:1.5}
#lab-boundary a{color:#ffcf8a}`;
  const ORIGIN_RE = /^http:\/\/(127\.0\.0\.1|localhost):\d{1,5}$/;

  function validateRecipe(r) {
    if (!r || typeof r !== 'object') throw Error('Recipe payload is missing.');
    const b = BOUNDS[r.model];
    if (!b) throw Error('Unsupported model: ' + String(r.model).slice(0, 60));
    if (!Array.isArray(r.components) || !r.components.length) throw Error('Recipe has no components.');
    if (r.components.length > 400) throw Error('Recipe has too many components.');
    const components = r.components.map((c, i) => {
      if (!c || !Number.isInteger(c.layer) || c.layer < 0 || c.layer >= b.layers) throw Error(`Component ${i + 1}: layer out of range for ${r.model} (0–${b.layers - 1}).`);
      if (!Number.isInteger(c.neuron) || c.neuron < 0 || c.neuron >= b.neurons) throw Error(`Component ${i + 1}: neuron out of range for ${r.model} (0–${b.neurons - 1}).`);
      if (typeof c.dose !== 'number' || !Number.isFinite(c.dose)) throw Error(`Component ${i + 1}: dose must be finite.`);
      return { layer: c.layer, neuron: c.neuron, dose: c.dose };
    });
    const s = (v, n) => String(v === undefined || v === null ? '' : v).slice(0, n);
    const obj = v => (v && typeof v === 'object' && !Array.isArray(v) ? v : {});
    return { id: s(r.id, 200), name: s(r.name || r.id, 200), model: r.model, components,
      method: obj(r.method), normalization: s(r.normalization || 'unspecified', 200), source: s(r.source, 1000), provenance: obj(r.provenance) };
  }
  /* hash string (with leading #) -> {intent, labOrigin, recipe} or null when it is not a lab-recipe link */
  function parseFragment(hash) {
    if (typeof hash !== 'string' || !hash.startsWith(PREFIX)) return null;
    if (hash.length > MAX_FRAGMENT) throw Error('Recipe link is too large.');
    let p;
    try { p = JSON.parse(decodeURIComponent(hash.slice(PREFIX.length))); } catch (e) { throw Error('Recipe link could not be decoded.'); }
    if (!p || p.v !== 1 || !['3d', 'claude'].includes(p.intent)) throw Error('Unsupported recipe link.');
    const labOrigin = typeof p.lab_origin === 'string' && ORIGIN_RE.test(p.lab_origin) ? p.lab_origin : '';
    return { intent: p.intent, labOrigin, recipe: validateRecipe(p.recipe) };
  }
  const isExplorerModel = r => r.model === EXPLORER_MODEL;
  const signed = d => (d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d);
  function marksFor(recipe) {
    return recipe.components.map(c => ({ layer: c.layer, n: c.neuron, sign: c.dose < 0 ? '-' : '+', w: Math.min(1, 0.35 + Math.abs(c.dose) / 4) }));
  }
  function notesFor(recipe) {
    return recipe.components.map((c, i) => ({ layer: c.layer, n: c.neuron, text: `#${i + 1} ${signed(c.dose)}` }));
  }
  function buildAskBody(recipe, question, history) {
    const q = String(question || '').trim();
    if (!q) throw Error('Type a question first.');
    if (q.length > 4000) throw Error('Question must be 4000 characters or fewer.');
    const h = (Array.isArray(history) ? history : []).filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .slice(-MAX_HISTORY).map(m => ({ role: m.role, content: m.content.slice(0, 4000) }));
    const bytes = b => new TextEncoder().encode(JSON.stringify(b)).length;
    // Drop the OLDEST history until the UTF-8 JSON body fits the bridge limit; recipe and question are never truncated.
    const make = () => { const b = { recipe, question: q }; if (h.length) b.history = h.slice(); return b; };
    let body = make();
    while (h.length && bytes(body) > MAX_BODY_BYTES) { h.shift(); body = make(); }
    if (bytes(body) > MAX_BODY_BYTES) throw Error('This recipe plus your question is too large for the assistant (limit ' + MAX_BODY_BYTES + ' bytes). Shorten the question or use a smaller recipe; components are never dropped.');
    return body;
  }
  async function ask(recipe, question, history, fetchImpl, signal) {
    const body = buildAskBody(recipe, question, history);
    let res;
    try {
      res = await (fetchImpl || root.fetch)(BRIDGE, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Recipe-Assistant': '1' }, body: JSON.stringify(body), signal });
    } catch (e) {
      const err = Error('The local recipe assistant is not reachable. Start the read-only Haiku bridge on your Mac (scripts/recipe_assistant.py, listening on 127.0.0.1:8788) and try again. Nothing was sent to the GPU host.');
      err.unavailable = true; throw err;
    }
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) throw Error((data && typeof data.error === 'string' && data.error) || 'Assistant error ' + res.status);
    if (!data || typeof data.answer !== 'string') throw Error('Assistant returned an unexpected response.');
    if (data.tools_enabled !== false) throw Error('Refusing the answer: the assistant did not confirm tools are disabled.');
    return { answer: data.answer, model: typeof data.model === 'string' ? data.model : 'unknown' };
  }

  /* ---------------- DOM (browser only) ---------------- */
  function install(adapter, doc, win) {
    doc = doc || root.document; win = win || root;
    const el = (tag, cls, text) => { const n = doc.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = String(text); return n; };
    const style = el('style', null, `
#tab-claude > :not(#lab-assistant){display:none!important}
#a-agent,#a-past,#a-new{display:none!important}
#lab-assistant{display:flex;flex-direction:column;flex:1;min-height:0;padding:12px 16px;gap:10px;overflow-y:auto;font-size:13px}
#lab-assistant h3{margin:0;font-size:14px}#lab-assistant .lab-note{opacity:.75;font-size:12px;line-height:1.4}
#lab-assistant .lab-meta{font-family:monospace;font-size:12px;line-height:1.5;word-break:break-word}
#lab-thread{display:flex;flex-direction:column;gap:8px;flex:1;min-height:60px;overflow-y:auto}
#lab-thread .lab-msg{white-space:pre-wrap;word-break:break-word;padding:6px 8px;border-radius:6px;background:rgba(255,255,255,.06)}
#lab-thread .lab-msg.user{background:rgba(255,255,255,.12)}#lab-thread .lab-err{color:#ff9d8a;white-space:pre-wrap}
#lab-assistant textarea{width:100%;box-sizing:border-box;min-height:56px}
#lab-assistant .lab-row{display:flex;gap:8px}
#lab-recipe-panel{position:fixed;left:50%;bottom:10px;transform:translateX(-50%);z-index:30;max-width:min(560px,60vw);max-height:42vh;overflow:auto;background:rgba(20,20,24,.94);border:1px solid rgba(255,255,255,.18);border-radius:8px;padding:8px 12px;font-size:12px;line-height:1.45}
#lab-recipe-panel .lab-boundary{color:#ffcf8a}#lab-recipe-panel ol{margin:6px 0;padding-left:0;list-style:none;font-family:monospace}
#lab-recipe-panel li{display:flex;gap:8px;align-items:center}
` + NON4B_CSS);
    doc.head.append(style);
    const tab = doc.querySelector('#tab-claude');
    const state = { parsed: null, history: [], busy: false, ctl: null };

    // ---- disable every legacy agent control; keep the original elements (ids) in place, hidden ----
    try { adapter.closeEvents(); } catch (e) { /* nothing to close */ }
    const stop = ev => { if (ev.target && ev.target.closest && ev.target.closest('#lab-assistant')) return; ev.preventDefault(); ev.stopImmediatePropagation(); };
    for (const type of ['submit', 'click', 'keydown', 'input', 'paste', 'change', 'drop', 'dragover', 'dragenter']) tab.addEventListener(type, stop, true);
    for (const id of ['#a-input', '#a-send', '#a-attach', '#a-file']) { const n = doc.querySelector(id); if (n) n.disabled = true; }
    const agentSel = doc.querySelector('#a-agent'); if (agentSel) agentSel.disabled = true;
    const claudeBtn = doc.querySelector('#tabs button[data-tab="claude"]');
    if (claudeBtn) { claudeBtn.title = 'Read-only recipe assistant (tool-free Haiku via the Claude CLI)'; }
    const boundary = el('div'); boundary.id = 'lab-boundary'; boundary.setAttribute('role', 'alert');
    doc.body.append(boundary);
    function setContext(r) {
      const non4b = !!r && !isExplorerModel(r);
      doc.body.classList.toggle('lab-non4b', non4b);
      if (non4b) {
        try { adapter.markAgent(false); adapter.map.setMarks([]); adapter.map.setNotes([]); } catch (e) { /* map may not be ready */ }
        const a = el('a', null, 'Return to Persona Laboratory ↗'); a.target = '_blank'; a.rel = 'noopener';
        a.href = (state.parsed && state.parsed.labOrigin || DEFAULT_LAB) + '/#atlas=' + encodeURIComponent(r.model);
        boundary.replaceChildren(el('strong', null, 'Model boundary: the original 3D explorer map is hidden.'),
          el('div', null, `This recipe is for ${r.model}. The map contains Qwen3.5-4B neurons only, so nothing is shown, focused or highlighted. The read-only assistant stays available on the right.`), a);
      } else boundary.replaceChildren();
    }

    // ---- assistant UI inside the original Claude tab ----
    const box = el('section'); box.id = 'lab-assistant';
    box.append(el('h3', null, 'Ask Claude about this recipe'));
    box.append(el('p', 'lab-note', 'Read-only and tool-free: Haiku runs through the Claude CLI on your Mac. Only the recipe and your questions are sent to Anthropic; no GPU-host credentials or tools are available to it. It cannot run tools, steer a model, or execute anything it suggests. Your browser sends the request to the local bridge at 127.0.0.1:8788, never to the GPU host.'));
    const meta = el('div', 'lab-meta', 'No recipe loaded. Open one from Persona Laboratory (“Ask Claude about this recipe”).');
    const thread = el('div'); thread.id = 'lab-thread'; thread.setAttribute('role', 'log'); thread.setAttribute('aria-live', 'polite');
    const input = el('textarea'); input.id = 'lab-question'; input.rows = 2; input.maxLength = 4000; input.placeholder = 'Ask about this recipe… (Enter to send, Shift+Enter for a new line)'; input.setAttribute('aria-label', 'Question about this recipe');
    const status = el('p', 'lab-note'); status.setAttribute('role', 'status');
    const send = el('button', null, 'Ask Haiku ↗'); send.type = 'button'; send.id = 'lab-ask';
    const clear = el('button', null, 'Clear conversation'); clear.type = 'button';
    const row = el('div', 'lab-row'); row.append(send, clear);
    box.append(meta, thread, input, row, status);
    tab.append(box);

    function line(cls, text) { const n = el('div', cls, text); thread.append(n); thread.scrollTop = thread.scrollHeight; return n; }
    function refreshAsk() { send.disabled = state.busy || !state.parsed; input.disabled = !state.parsed; }
    async function submit() {
      if (state.busy || !state.parsed) return;
      const question = input.value.trim(); if (!question) return;
      state.busy = true; refreshAsk(); status.textContent = 'Asking read-only Haiku…';
      line('lab-msg user', question); input.value = '';
      state.ctl = typeof AbortController === 'function' ? new AbortController() : null;
      const timer = win.setTimeout(() => state.ctl && state.ctl.abort(), 120000);
      try {
        const r = await ask(state.parsed.recipe, question, state.history, win.fetch && win.fetch.bind(win), state.ctl && state.ctl.signal);
        state.history.push({ role: 'user', content: question }, { role: 'assistant', content: r.answer });
        line('lab-msg assistant', r.answer); status.textContent = `${r.model} · tools disabled · read-only`;
      } catch (e) { line('lab-err', e.message); status.textContent = ''; }
      finally { win.clearTimeout(timer); state.busy = false; refreshAsk(); }
    }
    send.onclick = submit;
    clear.onclick = () => { if (state.ctl) state.ctl.abort(); state.history = []; thread.replaceChildren(); status.textContent = ''; };
    input.addEventListener('keydown', ev => { if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); submit(); } });
    refreshAsk();

    // ---- upstream neuron Chat steers exactly one neuron: say so ----
    const chatTab = doc.querySelector('#tab-chat');
    if (chatTab) { const n = el('p', null, 'Upstream chat steers ONLY the one focused neuron (not the whole recipe). To generate with the full signed combination, use Persona Laboratory.'); n.id = 'lab-chat-note'; n.style.cssText = 'margin:0;padding:8px 12px;font-size:12px;color:#ffcf8a'; chatTab.prepend(n); }

    // ---- recipe panel + 3D behaviour ----
    let panel = null;
    function describe(parsed) {
      const r = parsed.recipe;
      meta.replaceChildren(el('div', null, `${r.name}`), el('div', null, `${r.model} · ${r.normalization} · ${r.components.length} components`),
        el('div', null, `method ${JSON.stringify(r.method)}`), el('div', null, `source: ${r.source || '—'}`),
        el('div', null, r.components.map((c, i) => `#${i + 1} L${c.layer} N${c.neuron} ${signed(c.dose)}`).join('  ')));
      if (!isExplorerModel(r)) meta.append(el('div', 'lab-note', 'Model boundary: the original 3D explorer holds Qwen3.5-4B coordinates only. These neuron indexes are not shown on its map.'));
    }
    function highlightAll(parsed) {
      adapter.markAgent(true);
      adapter.map.setMarks(marksFor(parsed.recipe)); adapter.map.setNotes(notesFor(parsed.recipe));
    }
    function renderPanel(parsed) {
      if (panel) panel.remove();
      const r = parsed.recipe, ok = isExplorerModel(r);
      panel = el('div'); panel.id = 'lab-recipe-panel';
      panel.append(el('strong', null, r.name), el('div', null, `${r.model} · ${r.normalization} · ${r.components.length} components · signed doses`));
      if (ok) {
        panel.append(el('div', null, 'All components are highlighted on the map (− and + shown by sign). The first is focused; pick another to focus it.'));
        const ol = el('ol');
        r.components.forEach((c, i) => {
          const li = el('li'); const b = el('button', null, `#${i + 1} L${c.layer} N${c.neuron} ${signed(c.dose)}`); b.type = 'button';
          b.onclick = async () => { await adapter.openNeuron(c.layer, c.neuron, 'neuron', true, false); highlightAll(parsed); };
          li.append(b); ol.append(li);
        });
        panel.append(ol);
      } else {
        panel.append(el('div', 'lab-boundary', `Model boundary: this recipe is for ${r.model}. The original explorer only contains Qwen3.5-4B coordinates, so nothing is focused or highlighted.`));
        const a = el('a', null, 'Return to Persona Laboratory model atlas ↗'); a.target = '_blank'; a.rel = 'noopener';
        a.href = (parsed.labOrigin || DEFAULT_LAB) + '/#atlas=' + encodeURIComponent(r.model); panel.append(a);
      }
      const ask = el('button', null, 'Ask Claude about this recipe'); ask.type = 'button'; ask.onclick = () => adapter.openClaude();
      panel.append(ask, el('div', 'lab-note', 'Generation with the full combination happens only in Persona Laboratory; nothing here runs the model.'));
      doc.body.append(panel);
    }
    async function apply(parsed) {
      state.parsed = parsed; state.history = []; thread.replaceChildren(); status.textContent = '';
      setContext(parsed.recipe); describe(parsed); renderPanel(parsed); refreshAsk();
      if (!isExplorerModel(parsed.recipe)) adapter.openClaude();
      if (isExplorerModel(parsed.recipe)) {
        const c0 = parsed.recipe.components[0];
        try { await adapter.openNeuron(c0.layer, c0.neuron, 'neuron', true, false); } catch (e) { /* focus is best effort; marks still apply */ }
        highlightAll(parsed);
      }
      if (parsed.intent === 'claude') adapter.openClaude();
    }
    function fromHash() {
      let parsed;
      try { parsed = parseFragment(win.location.hash); } catch (e) { meta.textContent = 'Recipe link rejected: ' + e.message; return; }
      if (!parsed) return;
      try { win.history.replaceState(null, '', win.location.pathname + win.location.search); } catch (e) { /* ignore */ }
      apply(parsed);
    }
    win.addEventListener('hashchange', fromHash);
    fromHash();
    return { state, apply };
  }
  return { BOUNDS, PREFIX, MAX_FRAGMENT, BRIDGE, MAX_BODY_BYTES, NON4B_CSS, DEFAULT_LAB, validateRecipe, parseFragment, marksFor, notesFor, buildAskBody, ask, install, isExplorerModel };
});

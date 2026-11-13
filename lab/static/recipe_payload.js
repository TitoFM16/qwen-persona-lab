/* Pure helper: builds the bounded #lab-recipe=... URL fragment sent to the original 3D explorer.
   No DOM, no network. Loaded by the Lab page and by node tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LabRecipePayload = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const MODEL_BOUNDS = {
    'Qwen/Qwen3.5-4B': { layers: 32, neurons: 9216 },
    'Qwen/Qwen3-8B': { layers: 36, neurons: 12288 },
    'Qwen/Qwen3.8-27B': { layers: 64, neurons: 17408 },
  };
  const MAX_FRAGMENT = 32768;       // characters of the whole "#lab-recipe=..." fragment
  const PREFIX = '#lab-recipe=';
  const INTENTS = ['3d', 'claude'];

  function capText(v, max) {
    const s = typeof v === 'string' ? v : JSON.stringify(v === undefined ? null : v);
    return s.length > max ? s.slice(0, max - 1) + '…' : s;
  }
  function capJSON(v, max) {
    if (v === undefined || v === null) return {};
    const s = JSON.stringify(v);
    if (s.length <= max) return JSON.parse(s);
    return { truncated: true, preview: s.slice(0, Math.floor(max / 2)) };
  }
  function checkComponents(model, components) {
    const b = MODEL_BOUNDS[model];
    if (!b) throw Error('Unsupported model: ' + model);
    if (!Array.isArray(components) || !components.length) throw Error('Recipe has no steering components.');
    if (components.length > 400) throw Error('Recipe has too many components (' + components.length + ') for a link.');
    return components.map((c, i) => {
      const layer = c && c.layer, neuron = c && c.neuron, dose = c && c.dose;
      if (!Number.isInteger(layer) || layer < 0 || layer >= b.layers) throw Error(`Component ${i + 1}: layer must be 0–${b.layers - 1} for ${model}.`);
      if (!Number.isInteger(neuron) || neuron < 0 || neuron >= b.neurons) throw Error(`Component ${i + 1}: neuron must be 0–${b.neurons - 1} for ${model}.`);
      if (typeof dose !== 'number' || !Number.isFinite(dose)) throw Error(`Component ${i + 1}: dose must be a finite signed number.`);
      return { layer, neuron, dose };
    });
  }
  /* selected: catalog record; components: the CURRENT (possibly edited) component inputs. */
  function buildPayload(selected, components, intent, labOrigin) {
    if (!INTENTS.includes(intent)) throw Error('Unknown intent.');
    if (!selected || typeof selected !== 'object') throw Error('Select a recipe first.');
    const comps = checkComponents(selected.model, components);
    const source = capText(selected.source === undefined ? '' : selected.source, 1000);
    let original = [];
    try { original = (selected.components || selected.neurons || []).map(c => [c.layer, c.neuron, c.dose].join(',')); } catch (e) { original = []; }
    const srcNorm = typeof selected.source_normalization === 'string' ? selected.source_normalization : null;
    const normChanged = srcNorm !== null && srcNorm !== (selected.normalization || 'unspecified');
    const edited = normChanged || original.join(';') !== comps.map(c => [c.layer, c.neuron, c.dose].join(',')).join(';');
    const prov = capJSON(selected.provenance, 3000);
    const rawUrls = {};
    const raw = selected.raw && typeof selected.raw === 'object' ? selected.raw : {};
    for (const k of ['source_url', 'recipe_url', 'url']) if (typeof raw[k] === 'string' && raw[k]) rawUrls[k] = capText(raw[k], 1000);
    const provenance = Object.assign({}, rawUrls, typeof prov === 'object' && !Array.isArray(prov) ? prov : { value: prov },
      { parent_recipe_id: selected.parent_recipe_id || null, edited_in_lab: edited, status: capText(selected.status || 'pending', 40) });
    if (normChanged) provenance.source_normalization = capText(srcNorm, 200);
    return {
      v: 1, intent, lab_origin: String(labOrigin || ''),
      recipe: {
        id: capText(selected.id || '', 200), name: capText(selected.name || selected.id || '', 200), model: selected.model,
        components: comps, method: capJSON(selected.method, 4000),
        normalization: capText(selected.normalization || 'unspecified', 200), source, provenance,
      },
    };
  }
  function buildFragment(selected, components, intent, labOrigin) {
    const frag = PREFIX + encodeURIComponent(JSON.stringify(buildPayload(selected, components, intent, labOrigin)));
    if (frag.length > MAX_FRAGMENT) throw Error('Recipe is too large to pass by link (' + frag.length + ' > ' + MAX_FRAGMENT + ' characters). Components are never dropped, so nothing was opened.');
    return frag;
  }
  return { MODEL_BOUNDS, MAX_FRAGMENT, PREFIX, buildPayload, buildFragment, checkComponents };
});

"""Launch the original 3D explorer with every agent route disabled.

The vendor explorer is never edited. At response time this wrapper:
  * answers 403 for every /api/agent* path (all methods);
  * serves the upstream /app.js bytes plus a small appended tail that dynamically imports the overlay AFTER the
    original module has finished executing (a dynamic import is not hoisted like a static one);
  * serves the overlay itself from scripts/explorer_recipe_overlay.js.
Importing this module does not import the explorer server (or torch); only main() does.
"""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_STATIC = ROOT / 'vendor/neuron-explorer/explorer/static'
OVERLAY_PATH = Path(__file__).resolve().parent / 'explorer_recipe_overlay.js'
OVERLAY_URL = '/explorer_recipe_overlay.js'
APP_URL = '/app.js'
FORBIDDEN = 'Agent execution is disabled in Persona Laboratory.'

# Runs inside the upstream module scope (same lexical bindings), after all of its top-level code and awaits.
ADAPTER_TAIL = f"""
;/* Persona Laboratory overlay: appended when served; the file on disk is unchanged. */
{{
  const __labAdapter = Object.freeze({{ openNeuron, openClaude, setTab, closeEvents, map, state, scatter,
    markAgent: v => {{ agentMarked = !!v; }} }});
  import({OVERLAY_URL!r}).then(() => globalThis.LabRecipeOverlay.install(__labAdapter))
    .catch(error => console.error('Persona Laboratory overlay failed to load', error));
}}
"""


def is_agent_path(path: str) -> bool:
    return re.sub(r'/+', '/', path or '').lower().startswith('/api/agent')


def build_app_js(static_dir: Path = UPSTREAM_STATIC) -> bytes:
    return (Path(static_dir) / 'app.js').read_bytes() + ADAPTER_TAIL.encode()


def make_middleware(static_dir: Path = UPSTREAM_STATIC, overlay_path: Path = OVERLAY_PATH):
    from fastapi.responses import PlainTextResponse, Response
    js = 'text/javascript; charset=utf-8'
    no_store = {'Cache-Control': 'no-store'}

    async def middleware(request, call_next):
        path = request.url.path
        if is_agent_path(path):
            return PlainTextResponse(FORBIDDEN, status_code=403)
        if request.method == 'GET' and path == APP_URL:
            return Response(build_app_js(static_dir), media_type=js, headers=no_store)
        if request.method == 'GET' and path == OVERLAY_URL:
            return Response(Path(overlay_path).read_bytes(), media_type=js, headers=no_store)
        return await call_next(request)
    return middleware


def install(app, static_dir: Path = UPSTREAM_STATIC, overlay_path: Path = OVERLAY_PATH):
    """Register before the upstream static mount is reached; middleware always runs first."""
    app.middleware('http')(make_middleware(static_dir, overlay_path))
    return app


def main():
    os.chdir(ROOT / 'vendor/neuron-explorer')
    sys.path.insert(0, str(Path.cwd() / 'explorer'))
    import server
    install(server.app)
    import uvicorn
    uvicorn.run(server.app, host='127.0.0.1', port=8791)


if __name__ == '__main__':
    main()

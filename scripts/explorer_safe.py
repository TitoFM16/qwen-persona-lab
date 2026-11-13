"""Launch the original 3D explorer with every agent route disabled."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / 'vendor/neuron-explorer')
sys.path.insert(0, str(Path.cwd() / 'explorer'))
import server
from fastapi.responses import PlainTextResponse

@server.app.middleware('http')
async def disable_agents(request, call_next):
    if request.url.path.startswith('/api/agent'):
        return PlainTextResponse('Agent execution is disabled in Persona Laboratory.', status_code=403)
    return await call_next(request)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(server.app, host='127.0.0.1', port=8791)

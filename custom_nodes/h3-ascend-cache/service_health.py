"""Expose the native generation thread's liveness; no model work or auto-restart."""
import threading
from aiohttp import web
from server import PromptServer

def worker_alive():
    return any(t.is_alive() and getattr(getattr(t,'_target',None),'__name__',None)=='prompt_worker'
               for t in threading.enumerate())

@PromptServer.instance.routes.get('/h3-ascend/health')
async def generation_worker_health(request):
    alive=worker_alive()
    return web.json_response({'worker_alive':alive},status=200 if alive else 503)

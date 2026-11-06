"""Configure the upstream proxy without modifying its implementation."""
import argparse
from pathlib import Path
from aiohttp import web
from comfy_api_proxy.app import make_app
from comfy_api_proxy.auth import make_bearer_auth_middleware
from comfy_api_proxy.middleware import make_origin_only_middleware

parser = argparse.ArgumentParser()
parser.add_argument("--comfyui", default="http://127.0.0.1:8189")
parser.add_argument("--port", type=int, default=8190)
parser.add_argument("--state-dir", type=Path, required=True)
parser.add_argument("--token-file", type=Path, required=True)
args = parser.parse_args()
token = args.token_file.read_text().strip()
if not token:
    parser.error("Token is empty")
app = make_app(args.comfyui, state_dir=args.state_dir,
               middlewares=[make_origin_only_middleware([]), make_bearer_auth_middleware(token)])
web.run_app(app, host="127.0.0.1", port=args.port)


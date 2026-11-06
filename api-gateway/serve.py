"""Manual start of two owned loopback API processes; no model startup or auto-restart."""
import argparse
import datetime
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=Path, default=Path(__file__).parent / "config/ascend.json")
parser.add_argument("--port", type=int, default=8191)
parser.add_argument("--host", default="127.0.0.1", help="Gateway bind address; Bearer authentication is always required")
parser.add_argument("--proxy-port", type=int, default=8190)
parser.add_argument("--state-dir", type=Path, default=Path(__file__).parent / "runtime")
args = parser.parse_args()
root = Path(__file__).resolve().parent
os.umask(0o077)
state = args.state_dir.resolve()
state.mkdir(mode=0o700, parents=True, exist_ok=True)
os.chmod(state, 0o700)
token_file = state / "token"
if not token_file.exists():
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(40)+"\n")
os.chmod(token_file, 0o600)
for port in (args.port, args.proxy_port):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host if port == args.port else "127.0.0.1", port))
profile = json.loads(args.profile.read_text())
if profile["proxy_url"] != "http://127.0.0.1:"+str(args.proxy_port):
    raise SystemExit("Profile proxy URL and --proxy-port disagree")
children = []
stop = False


def terminate(signum=None, frame=None):
    global stop
    stop = True
    for child in children:
        if child.poll() is None:
            child.terminate()


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
env = os.environ | {"H3_API_PROFILE": str(args.profile.resolve()), "H3_API_STATE": str(state), "H3_API_TOKEN_FILE": str(token_file), "PYTHONUNBUFFERED": "1"}
commands = [
    [sys.executable, str(root / "run_proxy.py"), "--comfyui", profile["comfy_url"], "--port", str(args.proxy_port),
     "--state-dir", str(state / "official-proxy"), "--token-file", str(token_file)],
    [sys.executable, "-m", "uvicorn", "h3_api.app:configured_app", "--factory", "--host", args.host, "--port", str(args.port)],
]
try:
    for name, command in zip(["proxy", "api"], commands):
        with (state / (name+".log")).open("ab") as log:
            children.append(subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
    receipt = {"pid": os.getpid(), "children": [p.pid for p in children], "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "api": "http://127.0.0.1:"+str(args.port), "proxy": profile["proxy_url"], "token_file": str(token_file),
               "generation_started": False, "automatic_restart": False, "network_configuration_modified": False}
    (state / "service-launch.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    while not stop and all(p.poll() is None for p in children):
        time.sleep(.5)
finally:
    terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    (state / "service-stop.json").write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_codes": [p.returncode for p in children]}))


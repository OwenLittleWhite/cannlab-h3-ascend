#!/usr/bin/env python3
"""Start this ComfyUI instance once, without touching unrelated processes."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path('/mnt/workspace/h3-ascend')
PIDFILE=ROOT/'logs/comfyui.pid'

def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8188/system_stats',timeout=2) as r:
            return json.load(r)
    except Exception:return None

if PIDFILE.exists():
    pid=int(PIDFILE.read_text().strip())
    try:
        if Path(f'/proc/{pid}/cwd').resolve()==ROOT/'ComfyUI':
            current=health()
            if current:
                print('ComfyUI is already running; PID',pid)
                raise SystemExit(0)
            raise SystemExit('Recorded ComfyUI process is still running but not ready. Inspect logs; no process was stopped.')
    except FileNotFoundError:pass
if health():
    raise SystemExit('Port 8188 already serves another process; no process was changed.')
with (ROOT/'logs/comfyui.log').open('ab') as log:
    process=subprocess.Popen(['bash',str(ROOT/'scripts/start.sh')],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
PIDFILE.write_text(str(process.pid)+'\n')
for _ in range(90):
    if process.poll() is not None:raise SystemExit('ComfyUI exited; inspect '+str(ROOT/'logs/comfyui.log'))
    result=health()
    if result:
        print('ComfyUI ready at remote 127.0.0.1:8188; PID',process.pid,
              'NPU devices',len(result.get('devices',[])))
        (ROOT/'evidence').mkdir(exist_ok=True)
        (ROOT/'evidence/server-stats.json').write_text(json.dumps(result,indent=2)+'\n')
        break
    time.sleep(1)
else:raise SystemExit('Startup reached 90 seconds; inspect the existing process before retrying.')

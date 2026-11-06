#!/usr/bin/env python3
"""Explicit idempotent start. No boot service, no backend restart, no retry daemon."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'runtime'
receipt=STATE/'service-launch.json'
if receipt.exists():
    item=json.loads(receipt.read_text());proc=Path('/proc')/str(item['pid'])
    try:
        owned=proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==ROOT and b'serve.py' in (proc/'cmdline').read_bytes()
    except OSError: owned=False
    if owned:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8191/health',timeout=3) as response:
                assert response.status==200
        except Exception as error:
            raise SystemExit('Owned API process exists but is not healthy; inspect logs before taking action') from error
        print('API_ALREADY_RUNNING',item['api']);raise SystemExit(0)
STATE.mkdir(parents=True,exist_ok=True,mode=0o700)
with (ROOT/'supervisor.log').open('ab') as log:
    child=subprocess.Popen(['bash',str(ROOT/'scripts/start.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(40):
    if child.poll() is not None: raise SystemExit('API failed to start; inspect supervisor.log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8191/health',timeout=1) as response:
            if response.status==200:
                print('API_READY http://127.0.0.1:8191; token file:',str(STATE/'token'));break
    except OSError: pass
    time.sleep(.5)
else: raise SystemExit('API not ready after 20 seconds; process was left for inspection')

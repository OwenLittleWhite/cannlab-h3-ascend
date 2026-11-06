"""Start an owned development HTTPS tunnel to the authenticated gateway only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'runtime/https-process.json'
if receipt.exists():
    old=json.loads(receipt.read_text());proc=Path('/proc')/str(old['pid'])
    try:
        same=proc.stat().st_uid==os.getuid() and (proc/'exe').resolve()==Path(old['exe']) and (proc/'stat').read_text().split()[21]==old['start_ticks']
    except OSError: same=False
    if same:
        print('HTTPS_TUNNEL_ALREADY_RUNNING',old['pid']);raise SystemExit(0)
BINARY=ROOT/'tools/cloudflared'
URL='https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/cloudflared-linux-arm64'
SHA='3df43aedc9ff291d1735b5cb576cb97e0b157488796db2e32b28aecabb445775'
BINARY.parent.mkdir(exist_ok=True)
if not BINARY.exists():
    temp=BINARY.with_suffix('.download')
    started=time.monotonic()
    with urllib.request.urlopen(URL,timeout=30) as response, temp.open('wb') as out:
        while chunk:=response.read(1024*1024):
            out.write(chunk)
            if time.monotonic()-started>180: raise TimeoutError('Bounded tunnel binary download expired')
    assert hashlib.sha256(temp.read_bytes()).hexdigest()==SHA,'Official release SHA256 mismatch'
    temp.replace(BINARY);BINARY.chmod(0o700)
assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==SHA
runtime=ROOT/'runtime'
with (runtime/'https.log').open('ab') as log:
    p=subprocess.Popen([str(BINARY),'tunnel','--no-autoupdate','--url','http://127.0.0.1:8191','--protocol','http2'],
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(runtime/'https-process.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],
    'exe':str(BINARY),'sha256':SHA,'target':'http://127.0.0.1:8191','kind':'temporary-development-https','automatic_restart':False},indent=2))
print('HTTPS_TUNNEL_PID',p.pid,flush=True)

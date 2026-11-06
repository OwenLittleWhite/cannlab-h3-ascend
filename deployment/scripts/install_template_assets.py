#!/usr/bin/env python3
"""Complete the pinned template UI assets using the existing aria2 downloader."""
import json
import os
from pathlib import Path
import subprocess

r=Path('/mnt/workspace/h3-ascend')
files=json.loads((r/'template-assets-manifest.json').read_text())
wheels=r/'cache/template-wheels';wheels.mkdir(exist_ok=True)
lines=[]
for f in files:
    lines.extend([f['url'],f'  dir={wheels}',f"  out={f['filename']}",f"  checksum=sha-256={f['sha256']}"])
inputs=r/'cache/template-wheels.aria2.txt';inputs.write_text('\n'.join(lines)+'\n')
env=dict(os.environ)
env['LD_LIBRARY_PATH']=str(r/'tools/aria2/usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
subprocess.run([str(r/'tools/aria2/usr/bin/aria2c'),'-i',str(inputs),'-c','-j','3','-x','8','-s','8',
    '--min-split-size=4M','--check-integrity=true','--auto-file-renaming=false','--max-tries=4',
    '--retry-wait=3','--connect-timeout=15','--timeout=45','--file-allocation=none',
    '--console-log-level=warn','--enable-color=false','--summary-interval=30'],env=env,check=True)
subprocess.run([str(r/'.venv/bin/python'),'-m','pip','install','--no-deps','--no-index',
    *[str(wheels/f['filename']) for f in files]],check=True)
(r/'logs/templates-all.done').touch()
print('ALL_PINNED_TEMPLATE_ASSETS_INSTALLED',flush=True)

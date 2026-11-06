#!/usr/bin/env python3
"""Download the pinned official checkpoints using aria2's resume/checksum support."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/workspace/h3-ascend')
manifest = json.loads((ROOT / 'model-manifest.json').read_text())
model_root = ROOT / 'ComfyUI/models'
if shutil.disk_usage(ROOT).free < sum(f['size'] for f in manifest) + 5 * 1024**3:
    # aria2 allocates the target files in place; restarting a partial download
    # must account for their existing allocated blocks rather than count twice.
    allocated = sum((model_root / f['destination']).stat().st_blocks * 512
                    for f in manifest if (model_root / f['destination']).exists())
    if shutil.disk_usage(ROOT).free + allocated < sum(f['size'] for f in manifest) + 5 * 1024**3:
        raise SystemExit('Not enough persistent disk space for the selected manifest')
entries=[]
for f in manifest:
    dest = model_root / f['destination']
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries.extend([
        f"https://hf-mirror.com/{f['repo']}/resolve/{f['revision']}/{f['path']}",
        f'  dir={dest.parent}',
        f'  out={dest.name}',
        f"  checksum=sha-256={f['sha256']}",
    ])
input_path = ROOT / 'models.aria2.txt'
input_path.write_text('\n'.join(entries)+'\n')
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = str(ROOT / 'tools/aria2/usr/lib/aarch64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH','')
args=[str(ROOT / 'tools/aria2/usr/bin/aria2c'), '--input-file='+str(input_path),
      '--continue=true', '--always-resume=true', '--auto-file-renaming=false',
      '--check-integrity=true', '--file-allocation=none', '--max-concurrent-downloads=2',
      '--max-connection-per-server=12', '--split=12', '--min-split-size=8M',
      '--max-tries=5', '--retry-wait=3', '--connect-timeout=20', '--timeout=90',
      '--summary-interval=30', '--download-result=full', '--console-log-level=notice',
      '--enable-color=false', '--check-certificate=true']
subprocess.run(args, env=env, check=True)
for f in manifest:
    dest = model_root / f['destination']
    assert dest.stat().st_size == f['size'], dest
(ROOT / 'logs/models.done').write_text('aria2 sha-256 validation passed for all selected files\n')
print('ALL_MODELS_VERIFIED',flush=True)

#!/usr/bin/env python3
import json
from pathlib import Path

r = Path('/mnt/workspace/h3-ascend')
print('=== INSTALL ===')
print('\n'.join((r/'logs/bootstrap.log').read_text(errors='replace').splitlines()[-7:]))
print('=== DOWNLOAD ===')
lines=(r/'logs/download-models.log').read_text(errors='replace').splitlines()
selected=[x for x in lines if any(k in x for k in ['[#','Download complete','ERROR','Exception','ALL_MODELS','Download Results'])]
for line in selected[-3:]:
    print(line.split('https://')[0][:250])
for m in json.loads((r/'model-manifest.json').read_text()):
    p=r/'ComfyUI/models'/m['destination']
    print(p.name, 'allocated_GiB', round(p.stat().st_blocks*512/1024**3,2) if p.exists() else 0)
print('BOOTSTRAP_DONE',(r/'logs/bootstrap.done').exists(),'MODELS_DONE',(r/'logs/models.done').exists())

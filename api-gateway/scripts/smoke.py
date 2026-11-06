"""One short real NPU generation through the public contract; resumable evidence."""
import hashlib
import json
from pathlib import Path
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
client=httpx.Client(base_url='http://127.0.0.1:8191',headers={'Authorization':'Bearer '+(ROOT/'runtime/token').read_text().strip()},timeout=45,trust_env=False)
report={}

def save(): (OUT/'smoke.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))

def checked(response):
    response.raise_for_status()
    return response.json()

with httpx.Client(trust_env=False) as raw:
    assert raw.get('http://127.0.0.1:8191/v1/capabilities').status_code==401
    queue=raw.get('http://127.0.0.1:8188/queue').json()
    assert not queue['queue_running'] and not queue['queue_pending'],queue
report['unauthenticated_status']=401
report['ready']=checked(client.get('/ready'))
image=Path('/mnt/workspace/h3-ascend/ComfyUI/input/xianxia-15s/hero.png')
with image.open('rb') as f:
    asset=checked(client.post('/v1/assets',files={'file':('hero.png',f,'image/png')}))
report['asset']=asset
body={'prompt':'<Picture 1> is the identity reference for the adult man. Cinematic medium shot of the same adult man in blue traditional Chinese robes, standing in a sunny garden. He smiles gently. Natural wind ambience. No subtitles or text.',
      'reference_images':[asset['id']],'duration_seconds':1,'resolution':'480p','steps':8,'seed':2026110402}
report['request']=body
preview=checked(client.post('/v1/preview',json=body));report['applied_parameters']=preview['applied_parameters']
validated=checked(client.post('/v1/validate',json=body));assert validated['valid'] and not validated['generation_submitted']
report['validated']=True
headers={'Idempotency-Key':'cannlab-api-smoke-20261104-01'}
job=checked(client.post('/v1/jobs',json=body,headers=headers)); report['job']=job;save()
assert job['status']!='rejected',job
again=checked(client.post('/v1/jobs',json=body,headers=headers));assert again['id']==job['id']
report['duplicate_same_job']=True
assert client.post('/v1/jobs',json=body|{'seed':2026110403},headers=headers).status_code==409
report['changed_payload_status']=409;save()
start=time.monotonic()
while time.monotonic()-start<900:
    status=checked(client.get('/v1/jobs/'+job['id']));report['latest']=status;report['poll_elapsed_seconds']=round(time.monotonic()-start,3);save()
    if status['status'] in {'completed','succeeded','failed','cancelled','canceled','error','rejected'}:
        print(json.dumps({'id':job['id'],'status':status['status'],'elapsed':report['poll_elapsed_seconds']}),flush=True)
        break
    time.sleep(5)
else: raise RuntimeError('Smoke job still pending; inspect existing ID, do not resubmit')

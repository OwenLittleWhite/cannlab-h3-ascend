#!/usr/bin/env python3
"""Standard-library H3 API client. Never prints the token."""
import argparse
import json
import mimetypes
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid
import time

p=argparse.ArgumentParser()
p.add_argument('--connection',type=Path,default=Path(__file__).with_name('connection.json'))
sub=p.add_subparsers(dest='command',required=True)
for name in ['ready','capabilities','models']: sub.add_parser(name)
s=sub.add_parser('upload');s.add_argument('file',type=Path)
s=sub.add_parser('preview');s.add_argument('request',type=Path)
s=sub.add_parser('validate');s.add_argument('request',type=Path)
s=sub.add_parser('submit');s.add_argument('request',type=Path);s.add_argument('--key',required=True)
for name in ['status','audit','workflow']:
    s=sub.add_parser(name);s.add_argument('job_id')
s=sub.add_parser('download');s.add_argument('job_id');s.add_argument('output',type=Path)
a=p.parse_args()
c=json.loads(a.connection.read_text());base=c['base_url'].rstrip('/')
token=(a.connection.parent/c['token_file']).read_text().strip()

def request(path,body=None,headers=None,raw=False):
    url=path if path.startswith(('https://','http://')) else base+path
    if urllib.parse.urlsplit(url)[:2] != urllib.parse.urlsplit(base)[:2]:
        raise ValueError('Refusing to send credentials to a different origin')
    h={'Authorization':'Bearer '+token}|(headers or {})
    if isinstance(body,dict): body=json.dumps(body,ensure_ascii=False).encode();h['Content-Type']='application/json'
    req=urllib.request.Request(url,data=body,headers=h)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=90) as response:
                data=response.read()
                return data if raw else json.loads(data)
        except urllib.error.HTTPError as error:
            raise SystemExit(f'HTTP {error.code}: {error.read(2000).decode(errors="replace")}') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if body is None and attempt<2:
                time.sleep(attempt+1)
                continue
            raise SystemExit(f'Connection failed: {error}. For an uncertain submission, poll or retry with the original idempotency key.') from None

if a.command in ['ready','capabilities','models']:
    result=request('/ready' if a.command=='ready' else '/v1/'+a.command)
elif a.command=='upload':
    boundary=uuid.uuid4().hex;name=a.file.name.replace('"','_').replace('\r','_').replace('\n','_')
    kind=mimetypes.guess_type(name)[0] or 'application/octet-stream'
    data=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: {kind}\r\n\r\n').encode()+a.file.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    result=request('/v1/assets',data,{'Content-Type':'multipart/form-data; boundary='+boundary})
elif a.command in ['preview','validate','submit']:
    result=request('/v1/jobs' if a.command=='submit' else '/v1/'+a.command,json.loads(a.request.read_text()),{'Idempotency-Key':a.key} if a.command=='submit' else {})
else:
    result=request('/v1/jobs/'+urllib.parse.quote(a.job_id,safe='')+('/'+a.command if a.command in ['audit','workflow'] else ''))
    if a.command=='download':
        if result['status']!='succeeded': raise SystemExit('Task status: '+result['status'])
        output=next(x for x in result['execution']['outputs'] if x['content_type'].startswith('video/'))
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(request(output['url'],raw=True))
        result={'downloaded':str(a.output.resolve()),'bytes':a.output.stat().st_size}
print(json.dumps(result,ensure_ascii=False,indent=2))

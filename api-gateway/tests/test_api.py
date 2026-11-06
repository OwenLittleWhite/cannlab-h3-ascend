import json
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from h3_api.app import Settings,make_app
from h3_api.compiler import compile_request
from h3_api.schema import GenerateRequest

PROFILE=json.loads((Path(__file__).parents[1]/'config/ascend.json').read_text())
AUTH={'Authorization':'Bearer test-only-token'}
class Guard:
    def check(self):return {'policy':'test'}

def test_graph_preserves_requested_output_geometry_and_routes():
    for payload in [{'prompt':'text'},{'prompt':'ref','reference_images':['image-id']},{'prompt':'first','first_image':'image-id'}]:
        result=compile_request(GenerateRequest(**payload),PROFILE);w=result['workflow']
        assert w['10']['inputs']['steps']==8
        assert (w['20']['inputs']['width'],w['20']['inputs']['height'])==(1280,720)
        assert result['applied_parameters']['persistent_conditioning_cache']
        assert w['7']['class_type']=='H3AscendAPIConditioning'
        assert not any(n['class_type'].startswith('Ray') for n in w.values())
        if payload.get('reference_images'):assert w['31']['inputs']['image']['info']['id']=='image-id'

def test_auth_and_durable_submission(tmp_path):
    posts=[]
    def backend(request):
        if request.url.path=='/object_info':
            classes={n['class_type']:{} for n in compile_request(GenerateRequest(prompt='x'),PROFILE)['workflow'].values()}
            return httpx.Response(200,json=classes)
        if request.url.path=='/api/v2/jobs' and request.method=='POST':
            posts.append(json.loads(request.content));return httpx.Response(201,json={'id':'upstream-1'})
        raise AssertionError(str(request.url))
    def app():return make_app(Settings(PROFILE,tmp_path,'test-only-token'),guard=Guard(),client=httpx.AsyncClient(transport=httpx.MockTransport(backend)))
    with TestClient(app()) as c:
        assert c.get('/v1/capabilities').status_code==401
        assert c.post('/v1/preview',headers=AUTH,json={'prompt':'x','drive_audio':'a'}).status_code==200
        assert c.post('/v1/preview',headers=AUTH,json={'prompt':'x','final_audio':'a'}).status_code==422
        h=AUTH|{'Idempotency-Key':'stable-key'}
        one=c.post('/v1/jobs',headers=h,json={'prompt':'x'});assert one.status_code==202,one.text
        two=c.post('/v1/jobs',headers=h,json={'prompt':'x'});assert two.json()['id']==one.json()['id']
        assert c.post('/v1/jobs',headers=h,json={'prompt':'different'}).status_code==409
    with TestClient(app()) as c:assert c.post('/v1/jobs',headers=h,json={'prompt':'x'}).json()['id']==one.json()['id']
    assert len(posts)==1


def test_comfy_v3_combo_inventory_and_validation(tmp_path):
    info={n['class_type']:{} for n in compile_request(GenerateRequest(prompt='x'),PROFILE)['workflow'].values()}
    info.update({'SaveVideo':{'input':{'required':{'codec':['COMFY_DYNAMICCOMBO_V3',{'options':[{'key':'auto','inputs':{'required':{}}},{'key':'h264','inputs':{'required':{}}}]}]}}},'UNETLoader':{'input':{'required':{'unet_name':[[PROFILE['model'],PROFILE['text_model']],{}]}}},
        'BasicScheduler':{'input':{'required':{'scheduler':['COMBO',{'options':['simple','karras']}]}}},
        'CLIPLoader':{'input':{'required':{'clip_name':['COMBO',{'options':[PROFILE['clip']]}]}}},
        'MiniMaxH3TurboLoRA':{'input':{'required':{'lora_name':['COMBO',{'options':[PROFILE['loras'][0]['name']]}]}}}})
    def backend(request):
        assert request.url.path=='/object_info'
        return httpx.Response(200,json=info)
    app=make_app(Settings(PROFILE,tmp_path,'test-only-token'),guard=Guard(),client=httpx.AsyncClient(transport=httpx.MockTransport(backend)))
    with TestClient(app) as c:
        assert c.get('/v1/models',headers=AUTH).json()['schedulers']==['simple','karras']
        assert c.post('/v1/validate',headers=AUTH,json={'prompt':'x','scheduler':'karras'}).status_code==200
        assert c.post('/v1/validate',headers=AUTH,json={'prompt':'x','scheduler':'unknown'}).status_code==422

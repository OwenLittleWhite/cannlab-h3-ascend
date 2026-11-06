#!/usr/bin/env python3
"""Create small validation workflows from the upstream Ascend-SACT graph."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'upstream/MiniMax-H3-Comfy-NPU'
PROMPT='A locked-off cinematic medium shot of an adult Chinese man wearing a blue shirt, seated in a bright study. He looks at the camera, gently smiles and says in clear Mandarin Chinese: "你好，这是一次视频测试。" Natural synchronized speech and lip movement, quiet room ambience, no background music, no subtitles, no on-screen text. One continuous shot.'
MODEL='minimax_h3_fl2va_int8_convrot.safetensors'
TE='qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
LORA='minimax_h3_turbo_v4_step600_ema.safetensors'

def node(kind,**inputs):
    return {'class_type':kind,'inputs':inputs}

api={
 '1':node('UNETLoader',unet_name=MODEL,weight_dtype='default'),
 '2':node('CLIPLoader',clip_name=TE,type='minimax',device='default'),
 '3':node('VAELoader',vae_name='minimax_h3_video_vae_fp16.safetensors'),
 '4':node('VAELoader',vae_name='minimax_h3_audio_vae_fp32.safetensors'),
 '5':node('MiniMaxH3TurboLoRA',model=['1',0],lora_name=LORA,strength=1.0,low_vram=False),
 '6':node('MultiNPUParallelConfig',model=['5',0],clip=['2',0],video_vae=['3',0],audio_vae=['4',0],npu_count='2'),
 '7':node('MiniMaxH3ImageToVideo',clip=['6',1],vae=['6',2],prompt=PROMPT,width=736,height=416,length=124),
 '8':node('RandomNoise',noise_seed=20261104),
 '9':node('MiniMaxH3TurboSampler'),
 '10':node('BasicScheduler',model=['6',0],scheduler='simple',steps=6,denoise=1.0),
 '11':node('BasicGuider',model=['6',0],conditioning=['7',0]),
 '12':node('SamplerCustomAdvanced',noise=['8',0],guider=['11',0],sampler=['9',0],sigmas=['10',0],latent_image=['7',1]),
 '13':node('MiniMaxH3VAEDecodeParallel',samples=['12',0],vae=['6',2],model=['6',0]),
 '14':node('VAEDecodeAudio',samples=['12',0],vae=['6',3]),
 '15':node('CreateVideo',images=['13',0],fps=24.0,audio=['14',0],bit_depth=8),
 '16':node('SaveVideo',video=['15',0],filename_prefix='video/cannlab_h3_larry_v4_6step_smoke',format='auto',codec='auto'),
}
ui=json.loads((UP/'additional_files/workflows/minimax-h3/fl2va_8steps_lora.json').read_text())
ui['nodes']=[n for n in ui['nodes'] if n['id'] not in {137,139}]
ui['links']=[l for l in ui['links'] if l[0] not in {304,305}]
valid_links={l[0] for l in ui['links']}
for n in ui['nodes']:
    for inp in n.get('inputs',[]):
        if inp.get('link') is not None and inp['link'] not in valid_links:inp['link']=None
    for out in n.get('outputs',[]):
        if out.get('links') is not None:out['links']=[i for i in out['links'] if i in valid_links]
    kind=n['type']
    if kind=='UNETLoader':n['widgets_values']=[MODEL,'default']
    elif kind=='CLIPLoader':n['widgets_values']=[TE,'minimax','default']
    elif kind=='MultiNPUParallelConfig':n['widgets_values']=['2']
    elif kind=='MiniMaxH3TurboLoRA':n['widgets_values']=[LORA,1.0,False]
    elif kind=='BasicScheduler':n['widgets_values']=['simple',6,1.0]
    elif kind=='ResolutionSelector':n['widgets_values']=['16:9 (Widescreen)',0.3,32]
    elif kind=='PrimitiveFloat':n['widgets_values']=[5]
    elif kind=='PrimitiveStringMultiline':n['widgets_values']=[PROMPT]
    elif kind=='RandomNoise':n['widgets_values']=[20261104,'fixed']
    elif kind=='MiniMaxH3ImageToVideo':n['widgets_values']=['',736,416,124]
    elif kind=='SaveVideo':n['widgets_values']=['video/cannlab_h3_larry_v4_6step','auto','auto']
ui['extra']['cannlab_deployment']={'source':'Ascend-SACT/MiniMax-H3-Comfy-NPU','npu_count':2,'steps':6,'lora':LORA}
for name,data in [('h3_larry_v4_6step_2npu.api.json',api),('h3_larry_v4_6step_2npu.ui.json',ui)]:
    p=ROOT/'workflows'/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(p)

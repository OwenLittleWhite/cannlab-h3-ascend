"""PRO6000-compatible H3 request mapping to native Ascend workflows."""
import hashlib
import json
import math
import re
import secrets
import uuid
from .schema import GenerateRequest


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def compile_request(body:GenerateRequest,profile:dict,job_id=None):
    job_id=job_id or str(uuid.uuid4())
    if body.width:width,height=body.width,body.height
    else:
        short=int(body.resolution[:-1]);a,b=map(int,body.aspect_ratio.split(':'))
        width=round(short*a/min(a,b)/2)*2;height=round(short*b/min(a,b)/2)*2
    cw,ch=math.ceil(width/32)*32,math.ceil(height/32)*32
    requested=max(5,math.ceil(body.duration_seconds*24-1e-9));frames=requested+(5-requested)%17
    mode=body.resolved_audio_mode()
    loras=profile['loras'] if body.loras is None else [x.model_dump() for x in body.loras]
    has_refs=bool(body.reference_images or body.voice_references or body.drive_audio)
    model=body.model or (profile['model'] if has_refs else profile['text_model'])
    seed=body.seed if body.seed is not None else secrets.randbits(32)
    prompt=body.prompt
    if body.voice_references:
        offset=int(bool(body.drive_audio))
        prompt+='\n\nvoice_references:\n'+'\n'.join(f'{v.role}: use the voice in <Audio {i+offset}> as a voice reference.' for i,v in enumerate(body.voice_references,1))
    if body.avoid:prompt+='\n\navoid:\n'+body.avoid
    if mode=='lock_source' and '<Audio 1>' not in prompt:prompt+='\n\nsource_audio:\n<Audio 1> is the source soundtrack for this scene.'
    warnings=['Generated voice identity, dialogue ownership and lip sync require content review.']
    if frames<124:warnings.append('Shorter than the approximately 5-second trained range; suitable for protocol smoke tests.')
    if frames>362:warnings.append('Longer than the tested 362-frame context; runtime, memory and quality need workload validation.')
    if cw*ch>1280*736:warnings.append('Above the previously validated 720p canvas; increased NPU memory and runtime.')
    if body.steps!=8 or body.sampler!='h3_turbo' or loras!=profile['loras']:
        warnings.append('Sampling/LoRA choices are preserved and differ from the default Turbo8 preset.')
    if frames/24!=body.duration_seconds:warnings.append('Duration rounded up to the native 17*k+5 frame grid.')
    graph={};inputs=[];counter=30
    def node(key,kind,**values):
        graph[str(key)]={'class_type':kind,'inputs':values};return [str(key),0]
    def load(asset,kind,meaning):
        nonlocal counter
        if not asset.strip():raise ValueError('Asset IDs must not be blank')
        counter+=1;inputs.append({'asset_id':asset,'kind':'image' if kind=='LoadImage' else 'audio','meaning':meaning})
        return node(counter,kind,**{'image' if kind=='LoadImage' else 'audio':{'__type':'core/ASSET','info':{'id':asset}}})
    model_link=node(1,'UNETLoader',unet_name=model,weight_dtype='default')
    vv=node(3,'VAELoader',vae_name=profile['video_vae']);av=node(4,'VAELoader',vae_name=profile['audio_vae'])
    if loras:
        model_link=node(100,'H3AscendLoRAStack',model=model_link,loras_json=canonical(loras))
    node(6,'H3AscendSamplingConfig',model=model_link,video_vae=vv,audio_vae=av,npu_count=profile['npu_count'])
    cond={'vae':vv,'audio_vae':av,'prompt':prompt,'width':cw,'height':ch,'length':frames,
          'clip_name':profile['clip'],'npu_count':profile['npu_count'],'audio_mode':mode,
          'audio_denoise_strength':body.audio_denoise_strength,'cache_mode':'auto'}
    pictures=[];audios=[];keyframes=[]
    if body.first_image:
        cond['first_frame']=load(body.first_image,'LoadImage','first_frame');pictures.append('first_frame@0')
    for i,k in enumerate(sorted(body.keyframes,key=lambda k:k.time_seconds)):
        frame=math.floor(k.time_seconds*24+.5)
        if not 0<frame<frames-1:raise ValueError('Middle keyframes must lie strictly between first and last frames')
        if any(x['frame']==frame for x in keyframes):raise ValueError('Two keyframes resolve to the same 24fps frame')
        keyframes.append({'slot':i,'frame':frame,'time_seconds':frame/24})
        cond[f'guides.guide_{i}']=load(k.asset_id,'LoadImage',f'keyframe@{frame}');pictures.append(f'keyframe@{frame}')
    if body.last_image:
        cond['last_frame']=load(body.last_image,'LoadImage','last_frame');pictures.append(f'last_frame@{frames-1}')
    for i,asset in enumerate(body.reference_images):
        cond[f'ref_images.ref_image_{i}']=load(asset,'LoadImage',f'reference_image_{i+1}');pictures.append(f'reference_image_{i+1}')
    drive=None;final=None
    if body.drive_audio:
        drive=load(body.drive_audio,'LoadAudio','drive_audio');cond['drive_audio']=drive;audios.append('drive_audio')
    if body.final_audio:final=load(body.final_audio,'LoadAudio','final_audio')
    for i,voice in enumerate(body.voice_references):
        cond[f'ref_audios.ref_audio_{i}']=load(voice.asset_id,'LoadAudio',voice.role);audios.append(voice.role)
    counts={'Picture':len(pictures),'Audio':len(audios),'Video':0}
    for tag in re.finditer(r'<(Picture|Audio|Video)\s+(\d+)>',prompt):
        if not 1<=int(tag[2])<=counts[tag[1]]:raise ValueError('Prompt references unavailable media: '+tag[0])
    media_map={'pictures':{str(i):v for i,v in enumerate(pictures,1)},'audios':{str(i):v for i,v in enumerate(audios,1)},'videos':{},'source_audio_ordinal':1 if drive else None}
    cond['keyframes_json']=canonical(keyframes);cond['media_map_json']=canonical(media_map)
    node(7,'H3AscendAPIConditioning',**cond)
    node(8,'RandomNoise',noise_seed=seed)
    if body.sampler=='h3_turbo':node(9,'MiniMaxH3TurboSampler')
    else:node(9,'KSamplerSelect',sampler_name='euler')
    node(10,'BasicScheduler',model=['6',0],scheduler=body.scheduler,steps=body.steps,denoise=body.denoise)
    node(11,'BasicGuider',model=['6',0],conditioning=['7',0])
    node(12,'SamplerCustomAdvanced',noise=['8',0],guider=['11',0],sampler=['9',0],sigmas=['10',0],latent_image=['7',1])
    image=node(13,'MiniMaxH3VAEDecodeParallel',samples=['12',0],vae=['6',1],model=['6',0])
    output={'samples':['12',0],'audio_vae':['6',2],'conditioning_report':['7',2],'job_id':job_id}
    if drive:output['drive_audio']=drive
    if final:output['final_audio']=final
    audio=node(14,'H3AscendAPIOutput',**output)
    if (width,height)!=(cw,ch):image=node(20,'ImageCrop',image=image,width=width,height=height,x=(cw-width)//2,y=(ch-height)//2)
    video=node(15,'CreateVideo',images=image,audio=audio,fps=24.,bit_depth=8)
    node(16,'SaveVideo',video=video,filename_prefix='h3-api/'+job_id+'/video',format='auto',codec='auto')
    applied=body.model_dump();applied.update(seed=seed,model=model,loras=loras,audio_mode=mode,
        backend_audio_mode='native' if mode=='reference_only' and not drive else mode,
        fps=24,frames=frames,frame_count=frames,actual_duration_seconds=frames/24,duration_seconds=frames/24,
        duration_seconds_requested=body.duration_seconds,output_width=width,output_height=height,width=width,height=height,
        canvas_width=cw,canvas_height=ch,resolved_keyframes=keyframes,npu_count=int(profile['npu_count']),
        mode='hybrid' if pictures and has_refs and (body.first_image or body.last_image) else 'reference_to_video' if has_refs else 'first_last_frame_to_video' if pictures else 'text_to_video',
        persistent_conditioning_cache=True,final_audio=body.final_audio or (body.drive_audio if mode=='lock_source' else None))
    return {'workflow':graph,'applied_parameters':applied,'final_prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'workflow_sha256':digest(graph),'inputs':inputs,'warnings':warnings,'media_map':media_map,
        'frame_processing':'center crop of padded native canvas' if (width,height)!=(cw,ch) else 'none',
        'negative_conditioning_channel':False,'profile':profile['name']}

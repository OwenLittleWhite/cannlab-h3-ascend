"""Ascend H3 API conditioning/audio mapping. GPL-3.0-or-later.

Native ComfyUI encoders and samplers are retained. Audio fitting/masks follow
T8mars/comfyui-minimax-h3-audio-T8 core.py and conditioning.py contracts.
"""
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

import torch
import torchaudio
import folder_paths
import nodes
import node_helpers
import comfy.multigpu as mg
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize
from comfy_extras.nodes_multigpu import _place_minimax_clip_tp

from . import ROOT, read_bundle, write_bundle, flatten_tensors, tensor_digest, file_identity


def ordered(values):
    return [v for k,v in sorted((values or {}).items(),key=lambda item:int(item[0].rsplit('_',1)[-1])) if v is not None]


def encode_audio(vae, audio):
    wave=audio['waveform']
    if wave.ndim!=3 or wave.shape[0]!=1 or not torch.isfinite(wave).all():
        raise ValueError('Audio must be a finite [1,channels,samples] waveform')
    rate=int(getattr(vae,'audio_sample_rate',32000))
    if audio['sample_rate']!=rate:
        wave=torchaudio.functional.resample(wave,audio['sample_rate'],rate)
    if type(vae.first_stage_model).__name__!='MiniMaxH3AudioVAE':
        raise ValueError('The API audio input requires the installed H3 audio VAE')
    vae.crop_input=False
    return vae.encode(wave.movedim(1,-1))


def fit_audio(encoded, template):
    if encoded.shape[1:-1]!=template.shape[1:-1]:
        raise ValueError('Source audio latent layout differs from native H3 audio latent')
    target=template.shape[-1]
    if encoded.shape[-1]<target:
        encoded=torch.nn.functional.pad(encoded,(0,target-encoded.shape[-1]))
    return encoded[...,:target].to(template)


def pack(conditioning, latent, details):
    saved=dict(latent)
    for key in ['samples','noise_mask']:
        if key in saved and isinstance(saved[key],NestedTensor):
            saved[key]={'h3_av_tensors':list(saved[key].tensors)}
    return {'conditioning':conditioning,'latent':saved,'details':details}


def unpack(bundle):
    latent=dict(bundle['latent'])
    for key in ['samples','noise_mask']:
        if key in latent and isinstance(latent[key],dict) and 'h3_av_tensors' in latent[key]:
            latent[key]=NestedTensor(latent[key]['h3_av_tensors'])
    return bundle['conditioning'],latent,dict(bundle['details'])


class H3AscendAPIConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs=[io.Vae.Input('vae'),io.Vae.Input('audio_vae'),
            io.String.Input('prompt',multiline=True),io.Int.Input('width',default=1280,min=32,step=32),
            io.Int.Input('height',default=736,min=32,step=32),io.Int.Input('length',default=124,min=5,step=17),
            io.Combo.Input('clip_name',options=folder_paths.get_filename_list('text_encoders')),
            io.Combo.Input('npu_count',options=['1','2','4'],default='2'),
            io.Combo.Input('audio_mode',options=['native','reference_only','lock_source','remix_source'],default='native'),
            io.Float.Input('audio_denoise_strength',default=.35,min=0,max=1),
            io.String.Input('keyframes_json',default='[]'),io.String.Input('media_map_json',default='{}'),
            io.Combo.Input('cache_mode',options=['auto','cache_only','refresh'],default='auto'),
            io.Image.Input('first_frame',optional=True),io.Image.Input('last_frame',optional=True),
            io.Audio.Input('drive_audio',optional=True),
            io.Autogrow.Input('ref_images',optional=True,template=io.Autogrow.TemplatePrefix(input=io.Image.Input('ref_image'),prefix='ref_image_',min=0,max=9)),
            io.Autogrow.Input('ref_audios',optional=True,template=io.Autogrow.TemplatePrefix(input=io.Audio.Input('ref_audio'),prefix='ref_audio_',min=0,max=3)),
            io.Autogrow.Input('guides',optional=True,template=io.Autogrow.TemplatePrefix(input=io.Image.Input('guide'),prefix='guide_',min=0,max=7))]
        return io.Schema(node_id='H3AscendAPIConditioning',category='minimax/ascend/api',inputs=inputs,
            outputs=[io.Conditioning.Output(),io.Latent.Output(),io.String.Output()])

    @classmethod
    def execute(cls,vae,audio_vae,prompt,width,height,length,clip_name,npu_count,audio_mode,
                audio_denoise_strength,keyframes_json,media_map_json,cache_mode,
                first_frame=None,last_frame=None,drive_audio=None,ref_images=None,ref_audios=None,guides=None):
        started=time.perf_counter()
        refs=[first_frame,last_frame,drive_audio,ref_images,ref_audios,guides]
        skeleton,tensors=flatten_tensors(refs)
        identity={'format':'ascend-api-v3','size':[width,height,length],'prompt':prompt,'audio_mode':audio_mode,
            'audio_denoise_strength':audio_denoise_strength,'keyframes':keyframes_json,'media_map':media_map_json,
            'text_tp_transport':os.getenv('COMFYUI_TEXT_TP_BACKEND','peer'),'npu_count':npu_count,'encoder':file_identity(folder_paths.get_full_path_or_raise('text_encoders',clip_name)),
            'vaes':[file_identity(v.patcher.cached_patcher_init[1][0]) for v in [vae,audio_vae]],
            'code':[hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'ComfyUI/comfy/model_base.py',ROOT/'ComfyUI/comfy/ldm/minimax/model.py',ROOT/'ComfyUI/comfy/multidevice.py',ROOT/'ComfyUI/comfy/text_encoders/minimax.py']],
            'runtime':torch.__version__,'refs':skeleton,'ref_tensors':{k:[str(v.dtype),list(v.shape),tensor_digest(v)] for k,v in tensors.items()}}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        bundle=None if cache_mode=='refresh' else read_bundle(key)
        hit=bundle is not None
        if not hit:
            if cache_mode=='cache_only':raise RuntimeError('No matching persistent H3 API conditioning: '+key)
            latent,frames=_empty_av_latent(width,height,length)
            keyframes=[];pictures=[];items=[];blocks=[]
            specs=[]
            if first_frame is not None:specs.append((0,first_frame,'disabled'))
            plan=json.loads(keyframes_json)
            guide_values=ordered(guides)
            if len(plan)!=len(guide_values):raise ValueError('Guide image count differs from keyframe plan')
            for point,image in zip(plan,guide_values):specs.append((point['frame'],image,'center'))
            if last_frame is not None:specs.append((frames-1,last_frame,'center'))
            for frame,image,crop in sorted(specs,key=lambda row:row[0]):
                image=_resize(image[:1],width,height,crop)
                pictures.append(image)
                keyframes.append({'resolved_frame_index':frame,'latent':vae.encode(image)})
            for image in ordered(ref_images):
                h,w=image.shape[1:3];scale=min(1.,math.sqrt(width*height/(w*h)))
                rw=max(32,round(w*scale/32)*32);rh=max(32,round(h*scale/32)*32)
                resized=_resize(image[:1],rw,rh,'disabled')
                items.append({'type':'image','data':resized})
                blocks.append({'kind':'image','latent_h':rh//16,'latent_w':rw//16,'latent':vae.encode(resized)})
            source=None
            audios=ordered(ref_audios)
            if drive_audio is not None:
                source=fit_audio(encode_audio(audio_vae,drive_audio),latent['samples'].unbind()[1])
                blocks.append({'kind':'audio','ref_audio_t':source.shape[-1],'audio_latent':source})
                items.append({'type':'audio'})
            for audio in audios:
                encoded=encode_audio(audio_vae,audio)
                blocks.append({'kind':'audio','ref_audio_t':encoded.shape[-1],'audio_latent':encoded})
                items.append({'type':'audio'})
            clip=nodes.CLIPLoader().load_clip(clip_name,'minimax','default')[0]
            devices=tuple(mm.get_all_torch_devices(exclude_current=False))[:int(npu_count)]
            clip=_place_minimax_clip_tp(clip,devices) if len(devices)>1 else mg.place_clip(clip,devices[0])
            mg.prepare_stage_devices(clip,stage_name='H3 API Qwen conditioning')
            if blocks:
                tokens=clip.tokenize(prompt,minimax_ref_items=[{'type':'image','data':i} for i in pictures]+items)
            else:
                tokens=clip.tokenize(prompt,images=pictures)
            conditioning=clip.encode_from_tokens_scheduled(tokens)
            mg.offload_components(clip,stage_name='H3 API Qwen conditioning')
            values={}
            if keyframes:values.update(minimax_keyframes=keyframes,minimax_frame_count=frames)
            if blocks:values['minimax_refs']=blocks
            if values:conditioning=node_helpers.conditioning_set_values(conditioning,values)
            if audio_mode in ['lock_source','remix_source']:
                if source is None:raise ValueError('Source audio is required for this audio mode')
                video,template=latent['samples'].unbind()
                strength=0. if audio_mode=='lock_source' else audio_denoise_strength
                latent['samples']=NestedTensor((video,source))
                latent['noise_mask']=NestedTensor((torch.ones_like(video),torch.full_like(source,strength)))
            details={'frames':frames,'audio_mode':audio_mode,'audio_denoise_strength':audio_denoise_strength,
                'source_audio_latent_sha256':tensor_digest(source) if source is not None else None,
                'source_audio_latent_shape':list(source.shape) if source is not None else None,
                'source_audio_mask':0. if audio_mode=='lock_source' else audio_denoise_strength if audio_mode=='remix_source' else None,
                'media_map':json.loads(media_map_json),'keyframe_positions':[x['resolved_frame_index'] for x in keyframes],
                'reference_blocks':[x['kind'] for x in blocks]}
            bundle=pack(conditioning,latent,details);write_bundle(key,bundle)
        conditioning,latent,details=unpack(bundle)
        details.update(text_tp_transport=os.getenv('COMFYUI_TEXT_TP_BACKEND','peer'),cache_status='HIT' if hit else 'MISS',cache_key=key,conditioning_seconds=time.perf_counter()-started)
        logging.info('H3 API conditioning %s key=%s seconds=%.3f',details['cache_status'],key,details['conditioning_seconds'])
        return io.NodeOutput(conditioning,latent,json.dumps(details,ensure_ascii=False))


class H3AscendAPIOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3AscendAPIOutput',category='minimax/ascend/api',is_output_node=True,
            inputs=[io.Latent.Input('samples'),io.Vae.Input('audio_vae'),
                io.String.Input('conditioning_report',force_input=True),io.String.Input('job_id'),
                io.Audio.Input('drive_audio',optional=True),io.Audio.Input('final_audio',optional=True)],
            outputs=[io.Audio.Output(),io.String.Output()])

    @classmethod
    def execute(cls,samples,audio_vae,conditioning_report,job_id,drive_audio=None,final_audio=None):
        report=json.loads(conditioning_report)
        root=ROOT/'api-gateway/runtime/audio-audits';root.mkdir(parents=True,exist_ok=True)
        if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in job_id):
            raise ValueError('Invalid API job identity')
        audio_latent=samples['samples'].unbind()[1]
        if not torch.isfinite(audio_latent).all():raise RuntimeError('Generated audio latent is not finite')
        if report['audio_mode']=='lock_source':
            bundle=read_bundle(report['cache_key'])
            if bundle is None:raise RuntimeError('Locked audio verification requires its original conditioning bundle')
            original=unpack(bundle)[1]['samples'].unbind()[1].detach().cpu().float()
            actual=audio_latent.detach().cpu().float()
            delta=float((actual-original).abs().max()) if actual.shape==original.shape else float('inf')
            report.update(audio_latent_max_abs_delta=delta,audio_latent_tolerance=1e-5,
                audio_latent_bit_exact=tensor_digest(audio_latent)==report['source_audio_latent_sha256'])
            if delta>1e-5:
                report.update(job_id=job_id,audio_latent_preserved=False,audit_error='Locked audio latent exceeded PRO6000 tolerance')
                (root/(job_id+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
                raise RuntimeError('Sampler changed locked audio latent: max_abs_delta='+str(delta))
            audio=final_audio if final_audio is not None else drive_audio
            if audio is None:raise RuntimeError('Locked audio output is missing its source waveform')
            report.update(audio_latent_preserved=True,source_waveform_preserved=True,
                mux_audio_sha256=tensor_digest(audio['waveform']),audio_source='final_audio' if final_audio is not None else 'drive_audio')
        else:
            audio=vae_decode_audio(audio_vae,samples)
            report.update(audio_latent_preserved=False,source_waveform_preserved=False,audio_source='jointly_generated')
        report.update(job_id=job_id,completed_unix=time.time(),sampled_audio_sha256=tensor_digest(audio_latent),
            output_audio_sample_rate=audio['sample_rate'],output_audio_samples=audio['waveform'].shape[-1])
        (root/(job_id+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
        return io.NodeOutput(audio,json.dumps(report,ensure_ascii=False),ui={'text':[json.dumps(report,ensure_ascii=False)]})

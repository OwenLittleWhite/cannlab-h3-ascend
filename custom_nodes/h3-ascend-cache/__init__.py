"""H3 conditioning persistence for the pinned Ascend adapter. GPL-3.0-or-later.

Serialization is reused from Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached (MIT),
commit 507c1ecd7c92439134e097e0fac34381340ab8c9; see _vendor/LICENSE.
NPU sampling placement follows ComfyUI nodes_multigpu.py and uses its helpers.
"""
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import folder_paths
import nodes
import comfy.model_management as mm
import comfy.multigpu as mg
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io, ComfyExtension
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from comfy_extras.nodes_multigpu import MultiNPUParallelConfig, _place_minimax_clip_tp
from ._vendor.serialize import flatten_tensors, unflatten_tensors
from .diagnostics import H3AscendAttention

ROOT = Path('/mnt/workspace/h3-ascend')
CACHE = ROOT / 'cache/h3-conditioning-v1'
EVENTS = ROOT / 'experiments/cache-optimization-20261104/evidence/cache-events.jsonl'


def file_identity(path):
    path = Path(path).resolve()
    st = path.stat()
    return [str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def packed_bundle(conditioning, latent):
    packed = dict(latent)
    if isinstance(packed['samples'], NestedTensor):
        packed['samples'] = {'h3_av_tensors': list(packed['samples'].tensors)}
    return {'conditioning': conditioning, 'latent': packed}


def unpack_bundle(bundle):
    latent = dict(bundle['latent'])
    samples = latent['samples']
    if isinstance(samples, dict) and 'h3_av_tensors' in samples:
        latent['samples'] = NestedTensor(samples['h3_av_tensors'])
    return bundle['conditioning'], latent


def bundle_digest(bundle):
    skeleton, tensors = flatten_tensors(bundle)
    identities = {k: [str(v.dtype), list(v.shape), tensor_digest(v)] for k, v in tensors.items()}
    return hashlib.sha256(json.dumps([skeleton, identities], sort_keys=True).encode()).hexdigest()


def read_bundle(key):
    meta_path = CACHE / (key + '.json')
    data_path = CACHE / (key + '.safetensors')
    if not meta_path.exists() and not data_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    with safe_open(data_path, framework='pt', device='cpu') as f:
        assert f.metadata()['generation'] == meta['generation'], 'Torn cache write'
    with data_path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['file_sha256'], 'Cache checksum mismatch'
    bundle = unflatten_tensors(meta['skeleton'], load_file(data_path, device='cpu'))
    assert bundle_digest(bundle) == meta['bundle_sha256'], 'Cache tensor/metadata mismatch'
    return bundle


def write_bundle(key, bundle):
    CACHE.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    skeleton, tensors = flatten_tensors(bundle)
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
    temp_data = CACHE / (key + '.' + generation + '.tmp.safetensors')
    temp_meta = CACHE / (key + '.' + generation + '.tmp.json')
    save_file(tensors, str(temp_data), metadata={'generation': generation})
    with temp_data.open('rb') as f:
        file_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = {'generation': generation, 'skeleton': skeleton, 'file_sha256': file_sha256,
            'bundle_sha256': bundle_digest(bundle)}
    temp_meta.write_text(json.dumps(meta))
    os.replace(temp_data, CACHE / (key + '.safetensors'))
    os.replace(temp_meta, CACHE / (key + '.json'))
    assert bundle_digest(read_bundle(key)) == meta['bundle_sha256'], 'Round trip failed'
    return meta['bundle_sha256']


class H3AscendSamplingConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3AscendSamplingConfig', category='minimax/ascend',
            inputs=[io.Model.Input('model'), io.Vae.Input('video_vae'), io.Vae.Input('audio_vae'),
                    io.Combo.Input('npu_count', options=['1', '2', '4'], default='2')],
            outputs=[io.Model.Output(), io.Vae.Output(), io.Vae.Output()])

    @classmethod
    def execute(cls, model, video_vae, audio_vae, npu_count):
        count = int(npu_count)
        devices = tuple(mm.get_all_torch_devices(exclude_current=False))[:count]
        if len(devices) != count or any(d.type != 'npu' for d in devices):
            raise RuntimeError('Requested NPU devices are not available')
        roles = MultiNPUParallelConfig.ROLE_INDICES[count].copy()
        if model.load_device in devices:
            roles['dit'] = devices.index(model.load_device)
        model = mg.clone_patcher_to_device(model, devices[roles['dit']])
        model.load_device = devices[roles['dit']]
        if count > 1:
            model = mg.create_multigpu_deepclones(model, count, reuse_loaded=True,
                reload_model_options={'initial_device': torch.device('meta'), 'assign_state_dict': True})
            assert 1 + len(model.get_additional_models_with_key('multigpu')) == count
        else:
            model.remove_additional_models('multigpu')
        video_vae = mg.place_vae(video_vae, devices[roles['video_vae']])
        audio_vae = mg.place_vae(audio_vae, devices[roles['audio_vae']])
        config = {'device_count': count, 'devices': devices,
                  'roles': {k: devices[v] for k, v in roles.items()}, 'text_parallel': 'cached'}
        model.model_options['sequence_parallel'] = count > 1
        model.model_options['multi_npu_parallel_config'] = config
        model.attachments['pipeline_offload_before_sampling'] = (video_vae, audio_vae)
        video_vae.multi_device_config = config
        video_vae.previous_stage_components = (model, audio_vae)
        return io.NodeOutput(model, video_vae, audio_vae)


class H3AscendCachedRef2VA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = copy.deepcopy(MiniMaxH3ReferenceToVideo.define_schema())
        schema.node_id = 'H3AscendCachedRef2VA'
        schema.display_name = 'H3 Ascend Persistent Ref2VA Conditioning'
        schema.category = 'minimax/ascend'
        schema.inputs = [x for x in schema.inputs if x.id != 'clip']
        schema.inputs.extend([
            io.Combo.Input('clip_name', options=folder_paths.get_filename_list('text_encoders')),
            io.Combo.Input('npu_count', options=['1', '2', '4'], default='2'),
            io.Combo.Input('cache_mode', options=['auto', 'cache_only', 'refresh'], default='auto')])
        schema.outputs.append(io.String.Output())
        return schema

    @classmethod
    def execute(cls, vae, audio_vae, prompt, width, height, length, ref_image_size,
                clip_name, npu_count, cache_mode, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None):
        started = time.perf_counter()
        refs = [ref_images, ref_videos, ref_video_audios, ref_audios]
        skeleton, tensors = flatten_tensors(refs)
        code_files = [__file__, ROOT/'ComfyUI/comfy_extras/nodes_minimax_h3.py',
                      ROOT/'ComfyUI/comfy_extras/nodes_multigpu.py',
                      ROOT/'ComfyUI/comfy/text_encoders/minimax.py',
                      ROOT/'ComfyUI/comfy/ldm/minimax/vae.py']
        identity = {'format': 1, 'prompt': prompt, 'size': [width, height, length],
            'reference_size': ref_image_size, 'npu_count': npu_count,
            'encoder': file_identity(folder_paths.get_full_path_or_raise('text_encoders', clip_name)),
            'vaes': [file_identity(v.patcher.cached_patcher_init[1][0]) for v in (vae, audio_vae)],
            'runtime': [str(torch.__version__), str(torch_npu.__version__)],
            'code': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files],
            'refs': skeleton, 'ref_tensors': {k: [str(v.dtype), list(v.shape), tensor_digest(v)] for k,v in tensors.items()}}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        bundle = None if cache_mode == 'refresh' else read_bundle(key)
        hit = bundle is not None
        if not hit:
            if cache_mode == 'cache_only':
                raise RuntimeError('No matching persistent H3 conditioning: ' + key)
            clip = nodes.CLIPLoader().load_clip(clip_name, 'minimax', 'default')[0]
            devices = tuple(mm.get_all_torch_devices(exclude_current=False))[:int(npu_count)]
            clip = _place_minimax_clip_tp(clip, devices) if len(devices)>1 else mg.place_clip(clip, devices[0])
            result = MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt,
                width, height, length, ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios)
            conditioning, latent = result[0], result[1]
            bundle = packed_bundle(conditioning, latent)
            write_bundle(key, bundle)
        conditioning, latent = unpack_bundle(bundle)
        event = {'status': 'HIT' if hit else 'MISS', 'key': key,
            'seconds': time.perf_counter()-started, 'encoder_loaded': not hit,
            'bundle_sha256': bundle_digest(bundle), 'round_trip_exact': True}
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open('a') as f: f.write(json.dumps(event)+'\n')
        logging.info('H3 persistent conditioning %s', json.dumps(event))
        return io.NodeOutput(conditioning, latent, json.dumps(event))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [H3AscendSamplingConfig, H3AscendCachedRef2VA, H3AscendAttention, H3AscendAPIConditioning, H3AscendAPIOutput, H3AscendLoRAStack]


def comfy_entrypoint():
    return Extension()

from .api_nodes import H3AscendAPIConditioning, H3AscendAPIOutput

from .lora_stack import H3AscendLoRAStack

from . import service_health

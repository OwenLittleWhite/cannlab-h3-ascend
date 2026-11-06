"""Exercise real H3/Turbo loading and reinjection, without sampling or prompts."""
from pathlib import Path
import collections,importlib.util,json,sys,threading,time
import torch,torch_npu

R=Path('/mnt/workspace/h3-ascend');D=R/'diagnostics/sdma-repair-20261105-2208'
sys.path.insert(0,str(R/'ComfyUI'))
import nodes
import comfy.model_management as mm
import comfy.multigpu as mg
import comfy.weight_adapter.bypass as bypass

torch.set_num_threads(4);started=time.monotonic()
profile=json.loads((R/'api-gateway/config/ascend.json').read_text())
source=R/'ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py'
spec=importlib.util.spec_from_file_location('h3_turbo_component_test',source)
turbo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=turbo;spec.loader.exec_module(turbo)
counts=collections.Counter();lock=threading.Lock();original_move=bypass.move_adapter_tensor
def observed_move(weight,device,dtype=None):
    moved=original_move(weight,device,dtype)
    with lock:counts[(str(weight.device),str(device))]+=1
    return moved
bypass.move_adapter_tensor=observed_move

print('LOAD_REAL_H3_CHECKPOINT',profile['model'],flush=True)
base=nodes.UNETLoader().load_unet(profile['model'],'default')[0]
model=turbo.MiniMaxH3TurboLoRA().apply_lora(base,profile['loras'][0]['name'],1.0,low_vram=False)[0]
devices=tuple(mm.get_all_torch_devices(exclude_current=False))[:2]
assert len(devices)==2
print('WARM_PRIMARY_WITH_REAL_TURBO',flush=True)
mm.load_models_gpu([model],force_full_load=True)
assert model.is_injected
print('CLONE_RESIDENT_LORA_TO_SECOND_NPU',flush=True)
model=mg.create_multigpu_deepclones(model,2,reuse_loaded=True,
    reload_model_options={'initial_device':torch.device('meta'),'assign_state_dict':True})
patchers=list(mg.component_patchers(model));assert len(patchers)==2
rounds=[]
for iteration in range(3):
    phase=time.monotonic();print('DUAL_NPU_LOAD_BEGIN',iteration,flush=True)
    mm.load_models_gpu(patchers,force_full_load=True)
    assert all(p.is_injected for p in patchers)
    for device in devices:
        with mm.cuda_device_context(device):mm.synchronize()
    memory=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
    rounds.append({'round':iteration,'seconds':time.monotonic()-phase,'cgroup_gib':memory/2**30,
        'both_models_injected':True})
    print('DUAL_NPU_LOAD_PASS',rounds[-1],flush=True)
    mg.offload_components(model,stage_name='SDMA repair H3 loader verification')

peer_moves=sum(v for (source,target),v in counts.items() if source.startswith('npu:') and target.startswith('npu:') and source!=target)
assert peer_moves>0,'The resident cross-device adapter reinjection case was not exercised'
proof={'passed':True,'checkpoint':profile['model'],'lora':profile['loras'][0],
    'low_vram_merge_enabled':False,'rounds':rounds,
    'weight_moves':[{'source':s,'target':t,'count':v} for (s,t),v in sorted(counts.items())],
    'cross_device_host_staged_moves':peer_moves,'elapsed_seconds':time.monotonic()-started,
    'generation_submitted':False,'sampling_executed':False,'videos_generated':0}
(D/'h3-loader-results.json').write_text(json.dumps(proof,indent=2))
print('REAL_H3_LOADER_PASS',json.dumps(proof),flush=True)

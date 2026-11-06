"""Compose H3 LoRA deltas without overwriting an earlier bypass injection."""
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
import folder_paths
import nodes
import comfy.utils
from comfy_api.latest import io


def concatenate_loras(entries, pruned=False):
    """Represent sum(strength_i * B_i @ A_i) as one pair of rank-concatenated matrices."""
    by_module={}
    for weights,strength in entries:
        modules=[k[:-len('.lora_A.weight')] for k in weights if k.endswith('.lora_A.weight')]
        if not modules:raise ValueError('LoRA has no native H3 lora_A/lora_B pairs')
        for module in modules:
            a=weights[module+'.lora_A.weight'].float();b=weights[module+'.lora_B.weight'].float()
            alpha=weights.get(module+'.alpha')
            factor=strength
            if alpha is not None and not (pruned and 'adaln_proj.linear' in module):factor*=float(alpha)/a.shape[0]
            by_module.setdefault(module,[]).append((a,b*factor))
    result={}
    for module,parts in by_module.items():
        result[module+'.lora_A.weight']=torch.cat([a for a,b in parts],dim=0).contiguous()
        result[module+'.lora_B.weight']=torch.cat([b for a,b in parts],dim=1).contiguous()
    return result


class H3AscendLoRAStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3AscendLoRAStack',category='minimax/ascend/api',
            inputs=[io.Model.Input('model'),io.String.Input('loras_json',default='[]')],outputs=[io.Model.Output()])

    @classmethod
    def execute(cls,model,loras_json):
        requested=json.loads(loras_json);merged={};identity=[]
        for item in requested:
            path=Path(folder_paths.get_full_path_or_raise('loras',item['name']))
            st=path.stat();strength=float(item['strength'])
            identity.append([str(path.resolve()),st.st_size,st.st_mtime_ns,strength])
            merged[item['name']]=merged.get(item['name'],0.)+strength
        active=[(name,value) for name,value in merged.items() if value!=0.]
        loader=nodes.NODE_CLASS_MAPPINGS['MiniMaxH3TurboLoRA']()
        if not active:return io.NodeOutput(model)
        if len(active)==1:
            name,strength=active[0]
        else:
            key=hashlib.sha256(json.dumps([identity,bool(getattr(model.model.diffusion_model,'use_adaln_curves',False))],sort_keys=True).encode()).hexdigest()
            root=Path(folder_paths.get_folder_paths('loras')[0])/'h3-api-stacks';root.mkdir(parents=True,exist_ok=True)
            file=root/(key+'.safetensors')
            if not file.exists():
                entries=[(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise('loras',name),safe_load=True),strength) for name,strength in active]
                combined=concatenate_loras(entries,pruned=bool(getattr(model.model.diffusion_model,"use_adaln_curves",False)));temp=file.with_suffix('.tmp.safetensors')
                save_file(combined,str(temp),metadata={'h3_api_stack':json.dumps(requested,ensure_ascii=False)})
                os.replace(temp,file)
            name=str(file.relative_to(root.parent));strength=1.
        logging.info('H3 API LoRA stack: requested=%s effective=%s strength=%s',requested,name,strength)
        patched=loader.apply_lora(model,name,strength,low_vram=False)[0]
        patched.attachments['h3_api_lora_stack']=requested
        return io.NodeOutput(patched)

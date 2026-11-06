"""Real two-NPU LoRA tests; no prompt submission or video generation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,json,sys,time
import torch,torch_npu

R=Path('/mnt/workspace/h3-ascend');D=R/'diagnostics/sdma-repair-20261105-2208'
sys.path.insert(0,str(R/'ComfyUI'))
import comfy.model_management as mm
from comfy.weight_adapter.bypass import move_adapter_tensor,BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

started=time.monotonic();torch.set_num_threads(4)
assert torch.npu.device_count()==2
results=[]
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for shape in [(5376,64),(64,5376)]:
        source=(torch.arange(shape[0]*shape[1],dtype=torch.float32).remainder(1021)-510).div(32).reshape(shape).to(dtype)
        value=source
        for iteration in range(16):
            target=torch.device('npu',iteration%2)
            value=move_adapter_tensor(value,target)
            with mm.cuda_device_context(target):
                mm.synchronize();observed=value.cpu();mm.synchronize()
            assert observed.dtype==source.dtype and torch.equal(observed,source)
        results.append({'dtype':str(dtype),'shape':shape,'transfers':16,'bit_exact':True})
        print('TRANSFER_PASS',results[-1],flush=True)

# Also cover conversion on an already resident adapter.
for dtype in [torch.float16,torch.bfloat16]:
    source=torch.linspace(-5,5,5376*64).reshape(5376,64)
    value=move_adapter_tensor(source,torch.device('npu:0'))
    value=move_adapter_tensor(value,torch.device('npu:0'),dtype)
    with mm.cuda_device_context(torch.device('npu:0')):observed=value.cpu();mm.synchronize()
    assert torch.equal(observed,source.to(dtype))
    results.append({'conversion':str(dtype),'bit_exact':True})

up=(torch.arange(5376*64,dtype=torch.float32).remainder(7)-3).div(128).reshape(5376,64).to(torch.bfloat16)
down=up.T.contiguous()
adapter=LoRAAdapter([], (up,down,None,None,None,None))
seed=torch.nn.Linear(5376,5376,bias=False,dtype=torch.bfloat16)
seed_hook=BypassForwardHook(seed,adapter)
seed_hook._move_adapter_weights_to_device(torch.device('npu:0'))

def forward_test(rank):
    device=torch.device('npu',rank);other=torch.device('npu',1-rank)
    with mm.cuda_device_context(device):
        layer=torch.nn.Linear(5376,5376,bias=False,dtype=torch.bfloat16).to(device)
        layer.weight.detach().zero_()
        local_adapter=copy.copy(adapter)
        hook=BypassForwardHook(layer,local_adapter)
        x=torch.full((1,5376),1/32,dtype=torch.bfloat16,device=device)
        expected=torch.nn.functional.linear(torch.nn.functional.linear(x,down.to(device)),up.to(device))
        mm.synchronize()
        for iteration in range(8):
            # Model replicas may inherit adapter tensors resident on the other NPU.
            hook._move_adapter_weights_to_device(other)
            hook.inject()
            actual=layer(x)
            mm.synchronize()
            assert torch.equal(actual.detach().cpu(),expected.detach().cpu())
            hook.eject()
        return {'rank':rank,'forward_reinjections':8,'bit_exact':True}

with ThreadPoolExecutor(max_workers=2) as pool:forward_results=list(pool.map(forward_test,[0,1]))
proof={'passed':True,'transfers':results,'forward_results':forward_results,
    'elapsed_seconds':time.monotonic()-started,'generation_submitted':False}
(D/'npu-component-results.json').write_text(json.dumps(proof,indent=2))
print('NPU_COMPONENT_PASS',json.dumps(proof),flush=True)

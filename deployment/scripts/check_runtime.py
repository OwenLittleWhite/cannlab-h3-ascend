#!/usr/bin/env python3
"""Check NPU INT8 execution and media codecs without loading H3 weights."""
import importlib.metadata as metadata
import json
from pathlib import Path
import av
import torch
import torch_npu

assert torch.npu.is_available()
torch.manual_seed(20261104)
with torch.inference_mode():
    x=torch.randn(16,256,device='npu:0',dtype=torch.bfloat16)
    w=torch.randint(-32,32,(128,256),device='npu:0',dtype=torch.int8)
    xq,xscale=torch_npu.npu_dynamic_quant(x)
    wscale=torch.ones(128,device='npu:0',dtype=torch.float32)
    result=torch_npu.npu_quant_matmul(xq.contiguous(),w.T.contiguous(),wscale,
        pertoken_scale=xscale.reshape(-1).float().contiguous(),output_dtype=torch.bfloat16)
    torch.npu.synchronize()
    reference=(xq.float().cpu()@w.float().cpu().T)*xscale.float().cpu().reshape(-1,1)
    max_error=(result.float().cpu()-reference.to(torch.bfloat16).float()).abs().max().item()
    assert result.device.type=='npu' and torch.isfinite(result).all().item()
    assert max_error <= 2.0, max_error
report={'torch':torch.__version__,'torch_npu':torch_npu.__version__,
    'torchaudio':metadata.version('torchaudio'),'device_count':torch.npu.device_count(),
    'int8_output_device':str(result.device),'int8_shape':list(result.shape),
    'int8_max_abs_error_vs_quantized_reference':max_error,
    'h264_encoder_available':'libx264' in av.codecs_available,
    'aac_encoder_available':'aac' in av.codecs_available}
assert report['h264_encoder_available'] and report['aac_encoder_available'], report
p=Path('/mnt/workspace/h3-ascend/evidence/runtime-check.json')
p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

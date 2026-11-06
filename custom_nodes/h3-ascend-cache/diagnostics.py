"""Request-scoped measurement and exact dense attention alternatives. GPL-3.0-or-later."""
from collections import defaultdict
import json
from pathlib import Path
import threading
import time

import torch
import torch_npu
from safetensors.torch import save_file
import comfy.patcher_extension as pe
from comfy_api.latest import io

OUT = Path('/mnt/workspace/h3-ascend/experiments/cache-optimization-20261104/evidence')


def dense_attention(mode, q, k, v, scale):
    if mode == 'fusion':
        return torch_npu.npu_fusion_attention(q, k, v, head_num=q.shape[1],
            input_layout='BNSD', scale=scale, keep_prob=1.0, sparse_mode=0)[0]
    if mode == 'prompt':
        return torch_npu.npu_prompt_flash_attention(q, k, v, num_heads=q.shape[1],
            input_layout='BNSD', scale_value=scale, sparse_mode=0,
            pre_tokens=2147483647, next_tokens=2147483647)
    raise ValueError(mode)


class Trace:
    def __init__(self, label):
        self.label = label
        self.events = []
        self.lock = threading.Lock()

    def add(self, kind, device, start, end, wall, shapes):
        with self.lock:
            self.events.append((kind, str(device), start, end, wall, shapes))

    def finish(self, seconds, result):
        totals = defaultdict(lambda: {'calls': 0, 'device_ms': 0, 'host_seconds': 0})
        records = []
        for kind, device, start, end, wall, shapes in self.events:
            ms = start.elapsed_time(end)
            key = kind + ':' + device
            totals[key]['calls'] += 1
            totals[key]['device_ms'] += ms
            totals[key]['host_seconds'] += wall
            records.append({'kind': kind, 'device': device, 'ms': ms, 'host_seconds': wall, 'shapes': shapes})
        report = {'label': self.label, 'sampling_wall_seconds': seconds,
                  'totals': dict(totals), 'records': records,
                  'note': 'Per-device stream intervals overlap across ranks; do not sum rank totals as wall time.'}
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = str(time.time_ns())
        save_file({'packed_av_latent': result.detach().cpu().contiguous()}, str(OUT / ('profile-latent-' + self.label + '-' + stamp + '.safetensors')))
        (OUT / ('profile-' + self.label + '-' + stamp + '.json')).write_text(json.dumps(report, indent=2))
        self.events.clear()


class TimedCoordinator:
    def __init__(self, base, trace, packed=False):
        self.base = base
        self.trace = trace
        self.packed = packed

    def exchange(self, rank, tensors, cat_dims):
        if self.trace:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
        if self.packed and len(tensors)==2 and tuple(cat_dims)==(2,2) and tensors[0].shape==tensors[1].shape:
            batch = tensors[0].shape[0]
            combined, = self.base.exchange(rank, (torch.cat(tensors, dim=0),), (2,))
            result = tuple(combined.split(batch, dim=0))
        else:
            result = self.base.exchange(rank, tensors, cat_dims)
        if self.trace:
            end.record()
            self.trace.add('exchange-' + self.base.backend + ('-packed' if self.packed else ''), tensors[0].device, start, end,
                           time.perf_counter()-wall, [list(t.shape) for t in tensors])
        return result


class AttentionOverride:
    def __init__(self, mode, trace):
        self.mode = mode
        self.trace = trace

    def __call__(self, original, q, k, v, heads, mask=None, **kwargs):
        if self.trace:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
        if self.mode == 'baseline' or not kwargs.get('skip_reshape') or mask is not None:
            result = original(q, k, v, heads, mask=mask, **kwargs)
        else:
            result = dense_attention(self.mode, q, k, v, kwargs.get('scale', q.shape[-1]**-0.5))
            if not kwargs.get('skip_output_reshape', False):
                result = result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        if self.trace:
            end.record()
            self.trace.add('attention-' + self.mode, q.device, start, end,
                           time.perf_counter()-wall, [list(q.shape), list(k.shape)])
        return result


class DiffusionTrace:
    def __init__(self, trace, packed=False):
        self.trace = trace
        self.packed = packed

    def __call__(self, executor, *args, **kwargs):
        # MiniMaxH3Model.forward supplies x,timestep,context,transformer_options.
        options = dict(args[3])
        context = options.get('sequence_parallel_context')
        if context is not None:
            context = dict(context)
            context['coordinator'] = TimedCoordinator(context['coordinator'], self.trace, self.packed)
            options['sequence_parallel_context'] = context
        return executor(*args[:3], options, *args[4:], **kwargs)


class OuterTrace:
    def __init__(self, trace):
        self.trace = trace

    def __call__(self, executor, *args, **kwargs):
        self.trace.events.clear()
        started = time.perf_counter()
        result = executor(*args, **kwargs)
        for i in range(torch.npu.device_count()):
            torch.npu.synchronize(i)
        self.trace.finish(time.perf_counter()-started, result)
        return result


class H3AscendAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3AscendAttention', category='minimax/ascend',
            inputs=[io.Model.Input('model'),
                    io.Combo.Input('backend', options=['baseline', 'fusion', 'prompt'], default='baseline'),
                    io.Combo.Input('communication', options=['default','packed_kv','peer'], default='default'),
                    io.Boolean.Input('measure', default=False)], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model, backend, communication, measure):
        model = model.clone()
        trace = Trace(backend + '-' + communication) if measure else None
        if communication == 'peer':
            model.model_options['sequence_parallel_backend'] = 'peer'
        options = model.model_options['transformer_options'] = model.model_options.get('transformer_options', {}).copy()
        if backend != 'baseline' or measure:
            options['optimized_attention_override'] = AttentionOverride(backend, trace)
        if measure or communication == 'packed_kv':
            model.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, 'h3-timing', DiffusionTrace(trace, communication == 'packed_kv'))
        if measure:
            model.add_wrapper_with_key(pe.WrappersMP.OUTER_SAMPLE, 'h3-timing', OuterTrace(trace))
        return io.NodeOutput(model)

"""Deterministic interleaving regression; CPU only and no live process changes."""
import ast
import concurrent.futures
from contextlib import contextmanager
from pathlib import Path
import logging
from types import SimpleNamespace
import threading
import time
import unittest

ROOT=Path(__file__).parent
class TestCleanupOrder(unittest.TestCase):
    def exercise(self,path,fail=False):
        tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['prepare_stage_devices','empty_device_caches']]
        ns={'time':time,'concurrent':concurrent,'logging':logging}
        # Hashable devices match torch.device keys.
        class Device:
            type='npu'
            def __init__(self,index):self.index=index
            def __repr__(self):return f'npu:{self.index}'
        d0,d1=Device(0),Device(1);entered=threading.Event();peer_done=threading.Event();pending={d0};lock=threading.Lock();events=[];local=threading.local()
        class Model:
            def model_stage_unload(self):
                entered.set()
                if fail:raise RuntimeError('unload failed')
                assert peer_done.wait(2), 'peer device did not complete its offload scope'
                with lock:pending.clear();events.append('offload_done')
        @contextmanager
        def context(device):
            local.device=device
            if device==d1:assert entered.wait(2)
            try:
                yield
            finally:
                if device==d1:peer_done.set()
        def flush():
            with lock:
                if pending:
                    raise RuntimeError('global cache flush overlaps parameter release')
                events.append('flush')
        mm=SimpleNamespace(claim_device_models_except=lambda groups:{d0:[Model()],d1:[]},cuda_device_context=context,synchronize=lambda:None,soft_empty_cache=flush)
        ns.update(comfy=SimpleNamespace(model_management=mm),component_patchers=lambda *args:args)
        exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
        ns['prepare_stage_devices'](SimpleNamespace(load_device=d0),SimpleNamespace(load_device=d1))
        return events
    def test_original_code_exposes_real_overlap(self):
        with self.assertRaisesRegex(RuntimeError,'overlaps parameter release'):self.exercise(ROOT/'multigpu-before.py')
    def test_proposed_barrier_defers_all_flushes(self):
        self.assertEqual(self.exercise(ROOT/'multigpu-proposed.py'),['offload_done','flush','flush'])
    def test_offload_error_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError,'unload failed'):self.exercise(ROOT/'multigpu-proposed.py',fail=True)
if __name__=='__main__':unittest.main()

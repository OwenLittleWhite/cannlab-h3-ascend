import ast,logging
from pathlib import Path
from types import SimpleNamespace
import unittest
R=Path(__file__).parent
class Patcher:
    def __init__(self,device):self.load_device=device;self.clone_base_uuid='same-family';self.model=object();self.extra=[];self.is_multigpu_base_clone=True
    def clone(self):
        n=Patcher(self.load_device);n.extra=list(self.extra);return n
    def get_additional_models_with_key(self,key):return self.extra
    def set_additional_models(self,key,models):self.extra=models
    def match_multigpu_clones(self):pass
class Options:
    def register(self,model):pass
class TestExpiredPatchers(unittest.TestCase):
    def run_clone(self,path,loaded):
        n=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_multigpu_deepclones')
        generated=[]
        def clones(model,devices,**kw):generated.extend(devices);return [Patcher(d) for d in devices]
        ns={'ModelPatcher':Patcher,'GPUOptionsGroup':Options,'logging':logging,'comfy':SimpleNamespace(model_management=SimpleNamespace(get_all_torch_devices=lambda **kw:[0,1],loaded_models=lambda:loaded)),'deepclone_patcher_to_devices':clones}
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),ns)
        return ns['create_multigpu_deepclones'](Patcher(0),2,reuse_loaded=True),generated
    def test_original_crashes_on_expired_weak_reference(self):
        with self.assertRaises(AttributeError):self.run_clone(R/'multigpu-proposed.py',[None])
    def test_fix_skips_expired_and_reuses_live_clone(self):
        model,created=self.run_clone(R/'multigpu-recovery-v2.py',[None,SimpleNamespace(model=None),Patcher(1)])
        self.assertEqual([m.load_device for m in model.extra],[1]);self.assertEqual(created,[])
    def test_fix_creates_missing_clone_instead_of_reducing_device_count(self):
        model,created=self.run_clone(R/'multigpu-recovery-v2.py',[None,SimpleNamespace(model=None)])
        self.assertEqual([m.load_device for m in model.extra],[1]);self.assertEqual(created,[1])
if __name__=='__main__':unittest.main()

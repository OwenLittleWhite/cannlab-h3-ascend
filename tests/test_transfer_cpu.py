import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

P=Path(__file__).parent/'proposed/ComfyUI/comfy/weight_adapter/bypass.py'
tree=ast.parse(P.read_text())
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='move_adapter_tensor')
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BypassForwardHook')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_move_adapter_weights_to_device')

class Device:
    def __init__(self,name):self.name=name;self.type=name.split(':')[0]
    def __eq__(self,other):return isinstance(other,Device) and self.name==other.name

class Tensor:
    events=[]
    fail=False
    def __init__(self,device,dtype='bf16',value=7):self.device=Device(device);self.dtype=dtype;self.value=value
    def to(self,device,dtype=None,non_blocking=None):
        target=Device(device) if isinstance(device,str) else device
        if self.device.type==target.type=='npu' and self.device!=target:raise AssertionError('peer copy')
        Tensor.events.append(('copy',self.device.name,target.name,dtype,non_blocking))
        if Tensor.fail:raise RuntimeError('copy failed')
        return Tensor(target.name,dtype or self.dtype,self.value)

class Module:
    def to(self,**kwargs):self.called=kwargs

@contextmanager
def context(device):
    Tensor.events.append(('enter',device.name));yield;Tensor.events.append(('exit',device.name))

namespace={'comfy':SimpleNamespace(model_management=SimpleNamespace(cuda_device_context=context,
    synchronize=lambda:Tensor.events.append(('sync',)))), 'torch':SimpleNamespace(Tensor=Tensor),
    'nn':SimpleNamespace(Module=Module),'logging':SimpleNamespace(debug=lambda *a:None)}
exec(compile(ast.Module(body=[helper,method],type_ignores=[]),str(P),'exec'),namespace)
move=namespace['move_adapter_tensor'];move_weights=namespace['_move_adapter_weights_to_device']

class TransferTests(unittest.TestCase):
    def setUp(self):Tensor.events=[];Tensor.fail=False
    def test_peer_is_host_staged_and_synchronized(self):
        t=move(Tensor('npu:0'),Device('npu:1'))
        copies=[e for e in Tensor.events if e[0]=='copy']
        self.assertEqual([(e[1],e[2],e[4]) for e in copies],[('npu:0','cpu',False),('cpu','npu:1',False)])
        self.assertEqual(sum(e[0]=='sync' for e in Tensor.events),3);self.assertEqual(t.value,7)
    def test_same_device_has_no_copy(self):
        w=Tensor('npu:1');self.assertIs(move(w,Device('npu:1')),w)
        self.assertFalse(any(e[0]=='copy' for e in Tensor.events))
    def test_same_device_dtype_change_uses_host(self):
        t=move(Tensor('npu:0'),Device('npu:0'),'fp16')
        self.assertEqual(t.dtype,'fp16');self.assertEqual([e[2] for e in Tensor.events if e[0]=='copy'],['cpu','npu:0'])
    def test_cpu_upload_has_destination_fence(self):
        t=move(Tensor('cpu'),Device('npu:1'));self.assertEqual(t.dtype,'bf16')
        self.assertEqual(Tensor.events[-2],('sync',))
    def test_non_npu_keeps_original_transfer(self):
        move(Tensor('cpu'),Device('cuda:0'),'fp32')
        self.assertEqual(Tensor.events,[('copy','cpu','cuda:0','fp32',None)])
    def test_metadata_and_container_preserved(self):
        for container in [tuple,list]:
            a=SimpleNamespace(weights=container([Tensor('npu:0'),None,0.75,(2,3)]));move_weights(SimpleNamespace(adapter=a),Device('npu:1'))
            self.assertIsInstance(a.weights,container);self.assertEqual(a.weights[1:],[None,0.75,(2,3)] if container is list else (None,0.75,(2,3)))
    def test_failed_copy_keeps_previous_weights(self):
        old=(Tensor('cpu'),);a=SimpleNamespace(weights=old);Tensor.fail=True
        with self.assertRaises(RuntimeError):move_weights(SimpleNamespace(adapter=a),Device('npu:0'))
        self.assertIs(a.weights,old)
    def test_training_module_path_unchanged(self):
        a=Module();move_weights(SimpleNamespace(adapter=a),Device('npu:0'),'fp16')
        self.assertEqual(a.called['device'].name,'npu:0');self.assertNotIn('dtype',a.called)

if __name__=='__main__':unittest.main()

from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from h3_api.compiler import compile_request
from h3_api.schema import GenerateRequest
P=json.loads((Path(__file__).parents[1]/'config/ascend.json').read_text())
def build(**kw):return compile_request(GenerateRequest(prompt='测试场景',**kw),P)

@pytest.mark.parametrize('mode',['auto','native','reference_only','lock_source','remix_source'])
def test_all_audio_modes_have_real_graph_inputs(mode):
    d=build(drive_audio='source',audio_mode=mode,voice_references=[{'role':'甲','asset_id':'voice'}],audio_denoise_strength=.2)
    c=d['workflow']['7']['inputs'];o=d['workflow']['14']['inputs']
    assert c['audio_mode']==('lock_source' if mode=='auto' else mode)
    assert c['audio_denoise_strength']==.2 and c['drive_audio']==o['drive_audio']
    assert c['ref_audios.ref_audio_0']
    assert '甲: use the voice in <Audio 2>' in d['final_prompt']

def test_voice_reference_does_not_lock_or_replace_generated_audio():
    d=build(voice_references=[{'role':'甲','asset_id':'v1'},{'role':'乙','asset_id':'v2'}])
    assert d['applied_parameters']['audio_mode']=='reference_only'
    assert 'drive_audio' not in d['workflow']['7']['inputs']
    assert 'final_audio' not in d['workflow']['14']['inputs']
    assert d['media_map']['audios']=={'1':'甲','2':'乙'}

def test_separate_drive_and_final_audio_are_not_conflated():
    d=build(drive_audio='source',final_audio='clean')
    assert d['workflow']['14']['inputs']['drive_audio']!=d['workflow']['14']['inputs']['final_audio']
    assert len([x for x in d['inputs'] if x['kind']=='audio'])==2
    assert d['applied_parameters']['final_audio']=='clean'

def test_hybrid_keyframes_references_and_voice_bindings():
    d=build(first_image='f',last_image='l',keyframes=[{'asset_id':'middle2','time_seconds':2.5},{'asset_id':'middle1','time_seconds':1.25}],reference_images=['r'],drive_audio='a',voice_references=[{'role':'甲','asset_id':'v'}])
    assert [x['frame'] for x in d['applied_parameters']['resolved_keyframes']]==[30,60]
    assert d['media_map']['pictures']=={'1':'first_frame@0','2':'keyframe@30','3':'keyframe@60','4':'last_frame@242','5':'reference_image_1'}
    assert d['media_map']['audios']=={'1':'drive_audio','2':'甲'}
    assert 'guides.guide_1' in d['workflow']['7']['inputs']

@pytest.mark.parametrize('changes',[{'resolution':'1080p'},{'width':1278,'height':722},{'sampler':'euler','steps':20,'loras':[]},{'steps':4,'denoise':.8,'scheduler':'karras'},{'model':P['text_model'],'loras':[{'name':P['loras'][0]['name'],'strength':.7},{'name':P['loras'][0]['name'],'strength':.2}]}])
def test_previous_pro6000_parameter_choices_are_preserved(changes):
    d=build(**changes);a=d['applied_parameters']
    for k,v in changes.items():
        if k not in ['width','height','loras']:assert a[k]==v
    if changes.get('loras')==[]:assert not any(n['class_type']=='H3AscendLoRAStack' for n in d['workflow'].values())
    if changes.get('sampler')=='euler':assert d['workflow']['9']['class_type']=='KSamplerSelect'

@pytest.mark.parametrize('changes',[{'audio_mode':'lock_source'},{'final_audio':'x'},{'audio_mode':'reference_only'},{'keyframes':[{'asset_id':'x','time_seconds':1}]},{'voice_references':[{'role':'x','asset_id':'a'},{'role':'x','asset_id':'b'}]}])
def test_invalid_cross_field_combinations_remain_errors(changes):
    with pytest.raises(ValidationError):GenerateRequest(prompt='x',**changes)

@pytest.mark.parametrize('changes',[{'first_image':'a','last_image':'b','keyframes':[{'asset_id':'c','time_seconds':20}]},{'first_image':'a','last_image':'b','keyframes':[{'asset_id':'c','time_seconds':1},{'asset_id':'d','time_seconds':1.001}]}])
def test_invalid_keyframe_positions_are_not_silently_ignored(changes):
    with pytest.raises(ValueError):build(**changes)

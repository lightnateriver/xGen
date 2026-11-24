"""Recommended NPU startup JSON reaches the shared inference parser unchanged."""
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
import video_server
from x_diffusers.adaptor.h3_npu_options import H3_FUSION_FEATURES, validate


@pytest.mark.parametrize('task',['t2va','fl2va','ref2va'])
@pytest.mark.parametrize('disable_rope',[False,True])
def test_npu_vc_startup_recipe_and_individual_override(tmp_path,monkeypatch,task,disable_rope):
    root=Path(video_server.__file__).resolve().parents[2]
    name='h3_npu_ref2va' if task=='ref2va' else 'h3_npu_base'
    config=root/'example/diffusers/configs'/(name+'.example.json')
    image=tmp_path/'reference.png'
    Image.new('RGB',(512,512),(80,90,100)).save(image)
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    if disable_rope:service.default_config['h3_opt_rope']=False
    payload=dict(task_type=task,prompt='NPU startup contract',duration=5)
    if task=='ref2va':payload['ref_path']=[str(image)]
    if task=='fl2va':payload['first_frame']=str(image)
    # The explicit startup list must win over environment-only defaults.
    monkeypatch.setenv('H3_OPT_FEATURES','none')
    args=service._build_args(payload,save_path='unused.mp4')
    assert args.atten=='vc' and args.h3_opts=='all' and args.num_inference_steps==20
    expected=tuple(name for name in H3_FUSION_FEATURES if not (disable_rope and name=='rope'))
    assert validate(args,'npu')==expected

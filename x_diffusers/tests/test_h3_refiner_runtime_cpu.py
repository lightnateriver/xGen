"""Real Torch-CPU/FFmpeg checks; no model loading and no NPU allocations."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None,'requires Torch CPU')
class RuntimeTests(unittest.TestCase):
    def test_vae_conv_accepts_bf16_after_loader_kept_conv_in_fp32(self):
        source=(ROOT/'x_diffusers/x_diffusers/framework/pipeline/h3_regenerate/pipeline.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_vae_precision')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'vae-precision','exec'),namespace)
        vae=torch.nn.Module()
        vae.post_quant_conv=torch.nn.Conv3d(2,2,1).float()
        vae.decoder=torch.nn.Sequential(torch.nn.Conv3d(2,2,1),torch.nn.GroupNorm(1,2))
        namespace['prepare_vae_precision'](vae)
        self.assertEqual(vae.post_quant_conv.weight.dtype,torch.bfloat16)
        self.assertEqual(vae.post_quant_conv.bias.dtype,torch.bfloat16)
        self.assertEqual(vae.decoder[1].weight.dtype,torch.float32)
        output=vae.decoder(vae.post_quant_conv(torch.ones(1,2,2,2,2,dtype=torch.bfloat16)))
        self.assertTrue(torch.isfinite(output).all())

        namespace['prepare_vae_precision'](vae, backend='cuda')
        self.assertEqual(vae.decoder[1].weight.dtype,torch.bfloat16)
        output=vae.decoder(vae.post_quant_conv(torch.ones(1,2,2,2,2,dtype=torch.bfloat16)))
        self.assertTrue(torch.isfinite(output).all())

    def test_legacy_and_callback_decode_are_numerically_identical(self):
        path='x_diffusers/x_diffusers/framework/vae/minimax_h3.py'
        original=subprocess.check_output(['git','show','origin/v0.1:'+path],cwd=ROOT).decode()
        current=(ROOT/path).read_text(encoding='utf-8')
        def function(source):
            node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_decode_chunks_streaming')
            capture=[];namespace={'torch':torch,'_stream_frames_to_av':lambda stream,chunk:capture.append(chunk.clone())}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'decoder','exec'),namespace)
            return namespace['_decode_chunks_streaming'],capture
        old,old_frames=function(original);new,new_frames=function(current)
        vae=SimpleNamespace(tokens_chunk_size=2,config=SimpleNamespace(token_drop=0,clip_length=4),
            temporal_compression_ratio=2,token_overlap=0,frame_pre_padding=0,frame_overlap=0)
        vae._decode_clip=lambda z:z.repeat_interleave(2,dim=2)
        vae._blend=lambda a,b,*unused,**kwargs:b
        z=torch.arange(1*3*4*2*2,dtype=torch.float32).reshape(1,3,4,2,2)
        expected=old(vae,z,None,object());actual=new(vae,z,None,object())
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(torch.cat(old_frames,2),torch.cat(new_frames,2)))
        callback=[]
        callback_result=new(vae,z,None,object(),frame_callback=lambda value:callback.append(value.clone()))
        self.assertTrue(torch.equal(expected,callback_result))
        self.assertTrue(torch.equal(torch.cat(old_frames,2),torch.cat(callback,2)))

    def test_vendor_import_shims_are_restored(self):
        path=ROOT/'x_diffusers/x_diffusers/framework/pipeline/h3_regenerate/vendor/upscaler.py'
        mod=load('_upscaler_test',path)
        names=('folder_paths','comfy.model_management','comfy_api.latest')
        sentinel=object();previous={name:sys.modules.get(name,sentinel) for name in names}
        mod.load('/unused/weights.safetensors')
        self.assertEqual(previous,{name:sys.modules.get(name,sentinel) for name in names})

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'requires existing FFmpeg/ffprobe')
    def test_real_av_codec_success_failure_and_existing_legacy_mode(self):
        folder=ROOT/'x_base/x_base/vae_parallelism'
        package=ModuleType('_av_test');package.__path__=[str(folder)];sys.modules['_av_test']=package
        mod=load('_av_test.save_av_stream',folder/'save_av_stream.py')
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            for strict in (False,True):
                output=Path(directory)/f'{strict}.mp4'
                stream=mod.SaveAVStream(str(output),None,24,32,32,strict=strict)
                for _ in range(3):stream.put_frame(np.zeros((32,32,3),dtype=np.uint8))
                stream.close()
                result=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)]))
                self.assertEqual(result['streams'][0]['nb_frames'],'3')
                self.assertEqual(stream.proc.exitcode,0)
            stream=mod.SaveAVStream(str(Path(directory)/'bad.mp4'),None,24,32,32,codec='missing_codec',strict=True)
            try:
                with self.assertRaises((RuntimeError,BrokenPipeError)):
                    stream.put_frame(np.zeros((32,32,3),dtype=np.uint8));stream.close()
            finally:
                stream.abort()
            self.assertFalse(stream.proc.is_alive())


if __name__=='__main__':unittest.main(verbosity=2)

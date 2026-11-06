"""Compare the real native temporal loop against bounded CPU storage."""
import importlib.util
from pathlib import Path
from types import MethodType,SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

path=Path(__file__).resolve().parents[1]/'x_diffusers/framework/vae/h3_cuda_chunks.py'
spec=importlib.util.spec_from_file_location('h3_cuda_chunks',path)
chunks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunks)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',(torch.float16,torch.float32))
@pytest.mark.parametrize('frames',(3,7,13,37))
def test_native_decode_output_is_bit_exact(dtype,frames):
    vae=SimpleNamespace(tokens_chunk_size=5,temporal_compression_ratio=4,token_overlap=3,
                        frame_pre_padding=3,frame_overlap=12,config=SimpleNamespace(token_drop=3,clip_length=17))
    vae._blend=MethodType(AutoencoderKLMiniMaxH3._blend,vae)
    def clip(z):
        value=z.sum(dim=1,keepdim=True).repeat(1,3,4,1,1)
        # Exercise spatial blending on CUDA before a completed temporal clip
        # moves to CPU; this must not route the tile stitch through the CPU.
        output=vae._blend(value,value+0.125,2,dim=-1)
        assert output.is_cuda
        return output
    vae._decode_clip=clip
    vae._decode=MethodType(AutoencoderKLMiniMaxH3._decode,vae)
    latent=torch.randn(1,2,frames,4,8,device='cuda',dtype=dtype)
    with torch.no_grad():
        reference=vae._decode(latent).cpu()
        original_clip,original_blend=vae._decode_clip,vae._blend
        chunks.install_cpu_decode_chunks(vae)
        result=vae._decode(latent)
    assert result.device.type=='cpu' and result.dtype==reference.dtype
    assert torch.equal(result,reference)
    assert vae._decode_clip is original_clip and vae._blend is original_blend
    assert vae._h3_cpu_decode_stats['clips']>0


def test_failed_native_decode_restores_methods():
    def clip(z): raise RuntimeError('injected clip failure')
    def blend(*values): raise AssertionError('blend is not reached')
    vae=SimpleNamespace(_decode_clip=clip,_blend=blend)
    vae._decode=lambda z:vae._decode_clip(z)
    chunks.install_cpu_decode_chunks(vae)
    with pytest.raises(RuntimeError,match='injected clip failure'): vae._decode(torch.zeros(1))
    assert vae._decode_clip is clip and vae._blend is blend


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_staged_decoder_reuses_pinned_cpu_masters_exactly():
    class VAE:
        def __init__(self):
            self.decoder=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,32)).eval()
            self.post_quant_conv=torch.nn.Identity()
        def decode(self,x):return self.decoder(self.post_quant_conv(x))
    vae=VAE()
    x=torch.randn(17,32,device='cuda')
    vae.decoder.cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):reference=vae.decode(x)
    vae.decoder.cpu()
    chunks.install_staged_decoder(vae,torch.device('cuda',torch.cuda.current_device()),lambda m,d:m.to(d))
    masters=vae._h3_decoder_group.cpu_param_dict
    identities={id(p):v.data_ptr() for p,v in masters.items()}
    for _ in range(2):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):actual=vae.decode(x)
        assert torch.equal(actual,reference)
        assert all(p.device.type=='cpu' and p.is_pinned() for p in vae.decoder.parameters())
        assert {id(p):v.data_ptr() for p,v in masters.items()}==identities


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_rgb8_matches_native_pil_including_round_ties():
    from diffusers.video_processor import VideoProcessor
    processor=VideoProcessor(do_normalize=False)
    samples=[torch.rand(2,3,19,4,8,device='cuda'),
             ((torch.arange(256,device='cuda',dtype=torch.float32)+0.5)/255).clamp(0,1).view(1,1,1,16,16).expand(1,3,1,16,16)]
    for rgb in samples:
        reference=processor.postprocess_video(rgb.cpu(),output_type='pil')
        actual=chunks.rgb8_to_pil(rgb)
        assert [[f.tobytes() for f in batch] for batch in actual]==[[f.tobytes() for f in batch] for batch in reference]

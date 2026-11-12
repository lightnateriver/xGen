"""Independent CPU distributed checks for sharding, tile order and bounded FFN."""
import importlib.util,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image

module_path=Path(__file__).resolve().parents[1]/'x_diffusers/x_diffusers/framework/transformer/h3_image_optimizations.py'
spec=importlib.util.spec_from_file_location('image_optim',module_path);optim=importlib.util.module_from_spec(spec);spec.loader.exec_module(optim)
dist.init_process_group('gloo');torch.set_num_threads(2);torch.manual_seed(421)
try:
    model=MiniMaxH3ImageTransformer2DModel(num_attention_heads=4,attention_head_dim=8,hidden_size=32,num_layers=2,num_refiner_layers=1,ffn_dim=64,text_dim=16,time_embed_hidden_dim=32,time_embed_dim=16,freq_dim=16,rope_freq_dim=1,in_channels=2,num_pdd_heads=6).eval()
    data=dict(hidden_states=torch.randn(1,12,8),encoder_hidden_states=torch.randn(1,5,16),timestep=torch.tensor([0.4,0.999]),timestep_indices=torch.arange(17)%2,token_tags=torch.tensor([1]*5+[0]*12),position_ids=torch.randn(17,3),image_indices=torch.arange(5,17),text_indices=torch.arange(5),pdd_step_index=2,return_dict=False)
    with torch.inference_mode():
        expected=model(**data)[0]
        optim.install_image_dit_tp(model,dist.group.WORLD)
        actual=model(**data)[0]
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        optim.install_image_ff_chunk(model,chunk_size=3)
        torch.testing.assert_close(model(**data)[0],expected,atol=2e-6,rtol=2e-5)

    class Encoder(nn.Module):
        def forward(self,x):return F.avg_pool2d(x[:,:,0],2)[:,:,None]
    class Decoder(nn.Module):
        def forward(self,x):return F.interpolate(x,scale_factor=2,mode='nearest')
    class TileVAE(nn.Module):
        _split_tiles=AutoencoderKLMiniMaxH3Image._split_tiles
        _stitch_tiles=AutoencoderKLMiniMaxH3Image._stitch_tiles
        _blend=AutoencoderKLMiniMaxH3Image._blend
        _encode=AutoencoderKLMiniMaxH3Image._encode
        _decode=AutoencoderKLMiniMaxH3Image._decode
        def __init__(self):
            super().__init__();self.use_tiling=True;self.spatial_compression_ratio=2
            self.tile_sample_min_height=self.tile_sample_min_width=16
            self.tile_sample_min_overlap_height=self.tile_sample_min_overlap_width=4
            self.encoder=Encoder();self.decoder=Decoder();self.quant_conv=self.post_quant_conv=nn.Identity()
    vae=TileVAE();optim.install_image_vae_parallel(vae,dist.group.WORLD)
    with torch.inference_mode():
        for h,w in [(8,8),(32,32),(32,48),(8,64)]:
            x=torch.arange(h*w,dtype=torch.float32).reshape(1,1,h,w)
            torch.testing.assert_close(vae._encode(x),F.avg_pool2d(x,2),atol=1e-4,rtol=1e-6)
            torch.testing.assert_close(vae._decode(x),F.interpolate(x,scale_factor=2,mode='nearest'),atol=1e-4,rtol=1e-6)
    if dist.get_rank()==0:print('PASS: full Transformer TP, chunked FFN, original VAE blend with distributed encode/decode tiles',flush=True)
finally:dist.destroy_process_group()

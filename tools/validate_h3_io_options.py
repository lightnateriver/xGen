"""CPU/Gloo native H3 single-block gate for all four input/output switches."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace
import torch
import torch.distributed as dist
import x_diffusers
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from x_base.sequence_parallelism.mesh import ParallelManager
from x_base import set_pad

torch.set_num_threads(1)
dist.init_process_group('gloo',timeout=timedelta(seconds=90))
rank=dist.get_rank();world=dist.get_world_size();assert world==8
torch.manual_seed(42)
reference=MiniMaxH3Transformer3DModel(hidden_size=128,num_attention_heads=8,attention_head_dim=32,
    num_layers=1,num_refiner_layers=1,ffn_dim=256,in_channels=4,audio_in_channels=8,
    patch_size=(1,2,2),text_dim=64,freq_dim=32,time_embed_hidden_dim=128,time_embed_dim=32,rope_freq_dim=4).eval().requires_grad_(False)
manager=ParallelManager(1,1,8,1,register_tp_global=False)
checks=[]
with torch.inference_mode():
    for total in (3,17,32,35):
        torch.manual_seed(500+total)
        order=torch.randperm(total);ti,vi,ai=order[:1],order[1:-1],order[-1:]
        tags=torch.empty(total,dtype=torch.int64);tags[ti],tags[vi],tags[ai]=1,0,2
        values=dict(hidden_states=torch.randn(1,len(vi),16),audio_hidden_states=torch.randn(1,len(ai),8),
            encoder_hidden_states=torch.randn(1,len(ti),64),timestep=torch.tensor([0.6,0.0]),
            timestep_indices=torch.arange(total)%2,token_tags=tags,position_ids=torch.arange(total)[:,None].expand(-1,3).clone(),
            video_indices=vi,audio_indices=ai,text_indices=ti,return_dict=False)
        expected=reference(**values)
        for local_input in (False,True):
            for local_output in (False,True):
                model=deepcopy(reference);model.set_parallel_manager(manager)
                model._h3_cuda_local_input=local_input;model._h3_cuda_local_output=local_output
                if local_input:
                    for b in model.transformer_blocks:b.attn._h3_tp_cp=True
                # CPU uses ordinary Ulysses, whose legacy adapter consumes
                # 'pad'; production PHAA consumes 'mmh3_seq' directly.
                set_pad('pad',total,dist.group.WORLD)
                actual=model(**values)
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
                checks.append(dict(total=total,local_input=local_input,local_output=local_output,
                                   max_abs=max((a-b).abs().max().item() for a,b in zip(actual,expected))))
print(json.dumps(dict(rank=rank,status='passed',cases=checks)),flush=True)
dist.destroy_process_group()

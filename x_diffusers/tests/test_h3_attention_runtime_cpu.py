"""Real CPU tensors test layout transport and PHAA global-head mapping."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from x_diffusers.framework.transformer import h3_attention


class Recorder:
    def span(self,*args,**kwargs):return nullcontext()
    def host_span(self,*args,**kwargs):return nullcontext()
    def reset(self):pass


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(heads=56)) for _ in range(50)]
    def forward(self,position_ids,text_indices,audio_indices,video_indices):return position_ids


class Program:
    def __init__(self,layout):self.layout=layout
    def __call__(self,q,k,v,**kwargs):return q+2*k+3*v


class TransportTests(unittest.TestCase):
    def test_permutation_padding_and_global_last_head_shards(self):
        args=SimpleNamespace(h3_attention='spark',h3_veda_predictor=None,save_path='unused.mp4')
        model=FakeModel()
        with patch.object(h3_attention,'Telemetry',Recorder),patch('h3_ascend.backends.spark.SparkAttention',Program):
            session=h3_attention.BaseAttentionSession(model,args)
            session.request=args;session.operator_calls=0;session.routed_heads=set()
            n=37*24*42
            # Deliberately start native video before audio; the sparse layout must
            # gather text/audio first and scatter results to original token IDs.
            text=torch.arange(4,device='cpu')
            video=torch.arange(4,n+4,device='cpu')
            audio=torch.arange(n+4,n+7,device='cpu')
            model(torch.zeros(n+7,3,device='cpu'),text,audio,video)
            for offset,count in ((0,4),(4,3)):
                attn=model.transformer_blocks[24].attn
                attn.parallel_manager=SimpleNamespace(sp_group=None,sp_size=8)
                values=[torch.randn(1,count,n+8,128,device='cpu',dtype=torch.bfloat16) for _ in range(3)]
                with patch.object(h3_attention.dist,'get_rank',return_value=7):
                    output=attn._h3_attention_backend(*values,head_offset=offset)
                expected=values[0]+2*values[1]+3*values[2]
                expected[:,:,n+7:]=0
                torch.testing.assert_close(output,expected,rtol=0,atol=0)
            self.assertEqual(session.routed_heads,set(range(49,56)))
            self.assertEqual(set(session.programs),{(49,4),(53,3)})

    def test_reject_incomplete_real_token_layout(self):
        model=FakeModel();args=SimpleNamespace(h3_attention='spark',h3_veda_predictor=None)
        with patch.object(h3_attention,'Telemetry',Recorder):
            session=h3_attention.BaseAttentionSession(model,args);session.request=args
            n=37*24*42
            with self.assertRaisesRegex(ValueError,'exactly once'):
                model(torch.zeros(n+2,3,device='cpu'),torch.tensor([0],device='cpu'),
                      torch.tensor([0],device='cpu'),torch.arange(2,n+2,device='cpu'))
            self.assertTrue(session.failed)


if __name__=='__main__':unittest.main(verbosity=2)

"""Check model-local Sage dispatch and the existing masked SDPA fallback."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from x_base.operator import attention
from x_base.operator.h3_sage_cuda import H3SageCUDA


def test_sage_dispatch_is_model_local_and_counted(monkeypatch):
    kernel=Mock(return_value='sage-output')
    monkeypatch.setitem(sys.modules,'sageattention',SimpleNamespace(sageattn=kernel))
    previous=attention.attention_manager.cuda_attention
    backend=H3SageCUDA()
    q=SimpleNamespace(is_cuda=True,shape=(1,4,513,128),dtype=torch.bfloat16)
    assert backend(q,q,q)=='sage-output'
    assert backend.calls==backend.kernel_calls==1
    assert kernel.call_args.kwargs==dict(tensor_layout='HND',is_causal=False,sm_scale=None)
    assert attention.attention_manager.cuda_attention is previous


def test_masked_call_preserves_sdpa_fallback(monkeypatch):
    kernel=Mock()
    fallback=Mock(return_value='masked-output')
    monkeypatch.setitem(sys.modules,'sageattention',SimpleNamespace(sageattn=kernel))
    monkeypatch.setattr(attention,'base_attention_infer',fallback)
    backend=H3SageCUDA()
    q=SimpleNamespace(is_cuda=True,shape=(1,4,513,128),dtype=torch.bfloat16)
    mask=object()
    assert backend(q,q,q,attn_mask=mask)=='masked-output'
    assert backend.calls==1 and backend.kernel_calls==0
    assert fallback.call_args.args[3] is mask
    kernel.assert_not_called()

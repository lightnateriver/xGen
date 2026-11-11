"""Align each peer, retaining the exact FP8 bytes/scales and decoded tensors."""
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_non_cuda_transport_is_rejected_before_collective(monkeypatch, device):
    pytest.importorskip('triton')
    from x_base.phaa import fp8_transport as transport

    monkeypatch.setenv('XGEN_PHAA_FP8_ALIGN16', '1')
    tensor = torch.empty((2, 128), device=device)
    with patch.object(dist, 'get_world_size') as world_size, \
         patch.object(dist, 'all_to_all_single') as exchange:
        with pytest.raises(ValueError, match='requires CUDA tensors'):
            transport.fp8_all_to_all(torch.empty_like(tensor), tensor, 128, None)
    world_size.assert_not_called()
    exchange.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize('streams', [1, 3])
@pytest.mark.parametrize('seq', [17, 20])
@pytest.mark.parametrize('batch', [1, 2])
def test_peer_padding_preserves_transport(monkeypatch, streams, seq, batch):
    pytest.importorskip('triton')
    from x_base.phaa import fp8_transport as transport

    peers, heads, width = 2, 3, 128
    generator=torch.Generator(device='cuda').manual_seed(42)
    tensors=tuple(torch.randn((peers,seq,batch,heads,width),device='cuda',dtype=torch.bfloat16,
                             generator=generator) for _ in range(streams))
    wires=[]
    def exchange(output,tensor,**kwargs):
        wires.append(tensor.clone())
        output.copy_(tensor)
    outputs=[]
    with patch.object(dist,'get_world_size',return_value=peers), \
         patch.object(dist,'all_to_all_single',side_effect=exchange):
        # An absent switch must preserve the original row-shaped wire buffer.
        monkeypatch.delenv('XGEN_PHAA_FP8_ALIGN16',raising=False)
        for enabled in (False,True):
            if enabled:monkeypatch.setenv('XGEN_PHAA_FP8_ALIGN16','1')
            result=tuple(torch.empty_like(t) for t in tensors)
            transport.fp8_all_to_all_many(result,tensors,width,None)
            outputs.append(result)
    valid=seq*batch*heads*(width+4)*streams
    old,new=wires[0].reshape(peers,-1),wires[1].reshape(peers,-1)
    assert wires[0].shape==(streams*peers*seq*batch*heads,width+4)
    assert new.shape[1]%16==0
    assert torch.equal(old,new[:,:valid])
    assert bool(torch.all(new[:,valid:]==0))
    assert all(torch.equal(a,b) for a,b in zip(*outputs))

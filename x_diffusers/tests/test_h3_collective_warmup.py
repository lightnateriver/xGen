"""Warmup wiring on CPU; the distributed CUDA gate verifies actual collectives."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from x_diffusers.adaptor import h3_collective_warmup as module


def test_disabled_warmup_never_touches_cuda(monkeypatch):
    current=Mock(side_effect=AssertionError('disabled warmup touched CUDA'))
    monkeypatch.setattr(torch.cuda,'current_device',current)
    assert module.warmup_groups(None,SimpleNamespace(warmup_mib=0)) is None
    current.assert_not_called()


@pytest.mark.parametrize('tp,cp,wire',[(1,8,'fp8'),(4,2,'fp8'),(2,4,'bf16')])
def test_explicit_groups_dtypes_and_no_tensor_cache(monkeypatch,tp,cp,wire):
    zeros,empty=torch.zeros,torch.empty
    monkeypatch.setattr(torch,'zeros',lambda n,device,dtype:zeros(n,dtype=dtype))
    monkeypatch.setattr(torch,'empty',lambda n,device,dtype:empty(n,dtype=dtype))
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(2**30,2**31))
    synchronize=Mock();monkeypatch.setattr(torch.cuda,'synchronize',synchronize)
    calls=[]
    def reduce(value,**kwargs):calls.append(('reduce',value.dtype,kwargs))
    def gather(receive,send,**kwargs):calls.append(('gather',send.dtype,kwargs));receive.copy_(send.repeat(tp))
    def exchange(receive,send,**kwargs):calls.append(('exchange',send.dtype,kwargs));receive.copy_(send)
    monkeypatch.setattr(module.dist,'all_reduce',reduce)
    monkeypatch.setattr(module.dist,'all_gather_into_tensor',gather)
    monkeypatch.setattr(module.dist,'all_to_all_single',exchange)
    manager=SimpleNamespace(tp_size=tp,sp_size=cp,tp_group=object(),sp_group=object())
    stats=module.warmup_groups(manager,SimpleNamespace(warmup_mib=1,tp_comm=wire,cp_comm=wire))
    assert stats['completed'] and stats['calls']==1 and stats['cp_buffer_bytes']==1024**2
    assert calls[-1][0]=='exchange' and calls[-1][2]['group'] is manager.sp_group
    assert calls[-1][1]==(torch.uint8 if wire=='fp8' else torch.bfloat16)
    assert all(call[2]['group'] is manager.tp_group for call in calls[:-1])
    assert [call[0] for call in calls[:-1]]==([] if tp==1 else ['reduce','gather'] if wire=='fp8' else ['reduce'])
    assert synchronize.call_count==(1+(tp>1))
    assert not any(isinstance(value,torch.Tensor) for value in stats.values())


def test_collective_error_is_propagated_once(monkeypatch):
    zeros=torch.zeros
    monkeypatch.setattr(torch,'zeros',lambda n,device,dtype:zeros(n,dtype=dtype))
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(2**30,2**31))
    exchange=Mock(side_effect=RuntimeError('NCCL failed'))
    monkeypatch.setattr(module.dist,'all_to_all_single',exchange)
    with pytest.raises(RuntimeError,match='NCCL failed'):
        module.warmup_groups(SimpleNamespace(tp_size=1,sp_size=8,sp_group=object()),
                             SimpleNamespace(warmup_mib=1,cp_comm='fp8'))
    exchange.assert_called_once()


@pytest.mark.parametrize('mode,tp,cp',[('tp_sp',2,4),('tp_cp_shared',8,1)])
@pytest.mark.parametrize('wire',['bf16','fp8'])
def test_shared_sequence_bridges_warm_actual_collectives_without_unused_cp_group(monkeypatch,mode,tp,cp,wire):
    zeros,empty=torch.zeros,torch.empty
    monkeypatch.setattr(torch,'zeros',lambda n,device,dtype:zeros(n,dtype=dtype))
    monkeypatch.setattr(torch,'empty',lambda n,device,dtype:empty(n,dtype=dtype))
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(2**30,2**31))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    calls=[]
    manager=SimpleNamespace(tp_size=tp,sp_size=cp,tp_group=object(),sp_group=object())
    for name in ('all_reduce','all_gather_into_tensor','all_to_all_single','reduce_scatter_tensor'):
        def record(*values,_name=name,**kwargs):
            assert kwargs['group'] is manager.tp_group
            calls.append((_name,values[-1].dtype))
        monkeypatch.setattr(module.dist,name,record)
    stats=module.warmup_groups(manager,SimpleNamespace(mode=mode,warmup_mib=1,tp_comm=wire,cp_comm=wire))
    assert stats['cp_buffer_bytes']==0
    assert [n for n,d in calls]==['all_reduce','all_gather_into_tensor',
        'all_to_all_single' if wire=='fp8' else 'reduce_scatter_tensor']
    assert calls[-1][1]==(torch.uint8 if wire=='fp8' else torch.bfloat16)

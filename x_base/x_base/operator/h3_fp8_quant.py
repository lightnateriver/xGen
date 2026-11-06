# SPDX-License-Identifier: Apache-2.0
"""Exact per-tensor FP8 packing without full FP32 activation temporaries."""
import torch
import triton
import triton.language as tl
from .h3_abs_peak import activation_abs_peak


@triton.jit
def _pack(x,out,scale,N:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(x+offset,offset<N,other=0).to(tl.float32)
    value=tl.div_rn(value,tl.load(scale))
    value=tl.minimum(tl.maximum(value,-448.0),448.0)
    tl.store(out+offset,value.to(out.dtype.element_ty),offset<N)


def quantize_activation(x):
    scale=(activation_abs_peak(x)/448.0).clamp_min(torch.finfo(torch.float32).tiny)
    output=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    _pack[(triton.cdiv(x.numel(),4096),)](x,output,scale,x.numel(),4096)
    return output,scale

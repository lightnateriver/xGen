"""Opt-in PHAA7 SP4 BF16 layout kernels; callers own outputs and stream ordering."""
import os
from pathlib import Path
import torch
from .loader import load_library
_loaded=None
def library_path():return _loaded or Path(os.environ.get('H3_PHAA_LAYOUT_LIBRARY',Path(__file__).parent/'f4/build/libh3_phaa_layout.so')).resolve()
def _load(*xs):
    global _loaded
    if any(x.requires_grad for x in xs):raise ValueError('Inference only')
    if _loaded is None:
        path=library_path();load_library(str(path));_loaded=path
def pack_qkv(q,k,v,q_send,k_send,v_send,*,W=4,P=7,C=2,tile=64,blocks=40):
    if (W,P,C)!=(4,7,2):raise ValueError('Only SP4/PHAA7/2 heads supported')
    _load(q,k,v,q_send,k_send,v_send)
    torch.ops.h3_phaa_layout.pack(q,k,v,q_send,k_send,v_send,tile,blocks)
def unpack_qkv(q_recv,k_recv,v_recv,*,L):
    result=[]
    for x in [q_recv,k_recv,v_recv]:
        if x.ndim!=5 or x.shape[0]!=4 or tuple(x.shape[2:])!=(1,2,128) or not x.is_contiguous() or x.dtype!=torch.bfloat16:raise ValueError('contiguous BF16 [4,S,1,2,128] required')
        if not 0<L<=4*x.shape[1]:raise ValueError('Invalid effective length')
        result.append(x.flatten(0,1).permute(1,2,0,3).narrow(2,0,L))
    return tuple(result)
def pack_back(attn,back_send,*,L,S,W=4,tile=64,blocks=40):
    if W!=4:raise ValueError('Only SP4 supported')
    _load(attn,back_send);torch.ops.h3_phaa_layout.back(attn,back_send,S,L,tile,blocks)
def merge_back(back_recv,merged,*,W=4,P=7,C=2,tile=64,blocks=40):
    if (W,P,C)!=(4,7,2):raise ValueError('Only SP4/PHAA7/2 heads supported')
    _load(back_recv,merged);torch.ops.h3_phaa_layout.merge(back_recv,merged,tile,blocks)

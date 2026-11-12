"""Experimental merge/DQ only; original HCCL join and output MatMul stay outside."""
import os
from pathlib import Path
import torch
from .loader import load_library
_loaded=None
def library_path():
    return _loaded or Path(os.environ.get('H3_MERGE_QUANT_LIBRARY',Path(__file__).parent/'f4_merge_quant/build/libh3_merge_quant.so')).resolve()
def merge_quant(back_recv,*,blocks=40):
    global _loaded
    if back_recv.requires_grad:raise ValueError('Inference only; no backward')
    if _loaded is None:
        path=library_path();load_library(str(path));_loaded=path
    return torch.ops.h3_phaa_merge_quant.fused(back_recv,blocks)

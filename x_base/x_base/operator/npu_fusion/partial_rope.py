"""Opt-in inference-only Partial RoPE, FP32 frequencies and complete BF16 output."""
import os
from pathlib import Path
import torch
from .loader import load_library
_loaded_path=None
def library_path():
    return _loaded_path or Path(os.environ.get('H3_PARTIAL_ROPE_LIBRARY',Path(__file__).parent/'f3/build/libh3_partial_rope.so')).resolve()
def partial_rope_pair(q,k,cos,sin,*,rotary_dim=96,tile_heads=32,blocks=40):
    global _loaded_path
    if rotary_dim!=96:raise ValueError('Only rotary_dim=96 supported')
    if any(t.requires_grad for t in (q,k,cos,sin)):raise ValueError('Inference only; no backward')
    if _loaded_path is None:
        path=library_path();load_library(str(path));_loaded_path=path
    return torch.ops.h3_phaa_partial_rope.partial_rope_pair(q,k,cos,sin,tile_heads,blocks)

"""Indexed BF16 gate*branch + residual, with explicit multiply rounding and queued lifetime handling."""
import os
from pathlib import Path
import torch
from .loader import load_library
_loaded_path=None
def library_path():
    return _loaded_path or Path(os.environ.get('H3_GATE_RESIDUAL_LIBRARY',Path(__file__).parent/'f2a/build/libh3_gate_residual.so')).resolve()
def gate_residual(residual,branch,gate_table,indices,*,tile=5376,blocks=40):
    global _loaded_path
    if any(x.requires_grad for x in [residual,branch,gate_table]):raise ValueError('Experimental inference-only gate_residual has no backward')
    if _loaded_path is None:
        path=library_path();load_library(str(path));_loaded_path=path
    return torch.ops.h3_phaa_gate_residual.gate_residual(residual,branch,gate_table,indices,tile,blocks)

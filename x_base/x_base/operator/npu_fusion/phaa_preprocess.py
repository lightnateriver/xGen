"""Fused exact Laser FP16 input preparation, fixed H3 two-head BF16 path."""
import os
from pathlib import Path
import torch
from .loader import load_library
_loaded=None
def library_path():return _loaded or Path(os.environ.get('H3_PHAA_PREPROCESS_LIBRARY',Path(__file__).parent/'f4_preprocess/build/libh3_phaa_preprocess.so')).resolve()
def preprocess(q,k,v,*,tile=64,blocks=40):
    global _loaded
    if any(x.requires_grad for x in [q,k,v]):raise ValueError('Inference only')
    if _loaded is None:
        path=library_path();load_library(str(path));_loaded=path
    return torch.ops.h3_phaa_preprocess.run(q,k,v,tile,blocks)

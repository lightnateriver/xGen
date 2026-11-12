from pathlib import Path
import os
import torch
from .loader import load_library
_loaded=False
_vector_loaded=False
_reuse_loaded=False
_loaded_paths={}


def library_path(implementation='scalar'):
    if implementation in _loaded_paths: return _loaded_paths[implementation]
    names={'scalar':('H3_SWIGLU_QUANT_LIBRARY','libh3_swiglu_quant.so'),
           'vector':('H3_SWIGLU_QUANT_VECTOR_LIBRARY','libh3_swiglu_quant_vector.so'),
           'reuse':('H3_SWIGLU_QUANT_REUSE_LIBRARY','libh3_swiglu_quant_reuse.so')}
    env,name=names[implementation]
    return Path(os.environ.get(env,Path(__file__).parent/'f1b/build'/name)).resolve()


def swiglu_quantize(u,*,tile=2048,blocks=40,scale_mode=1):
    global _loaded
    if not _loaded:
        path=library_path('scalar')
        load_library(str(path)); _loaded=True; _loaded_paths['scalar']=path
    if u.requires_grad: raise ValueError('Experimental inference-only F1b has no backward')
    return torch.ops.h3_phaa_fusion_quant.swiglu_quantize(u,tile,blocks,scale_mode)


def swiglu_quantize_vector(u,*,tile=2048,blocks=40,scale_mode=1):
    global _vector_loaded
    if not _vector_loaded:
        path=library_path('vector')
        load_library(str(path)); _vector_loaded=True; _loaded_paths['vector']=path
    if u.requires_grad: raise ValueError('Experimental inference-only F1b has no backward')
    return torch.ops.h3_phaa_fusion_quant_vector.swiglu_quantize(u,tile,blocks,scale_mode)


def swiglu_quantize_reuse(u,*,tile=4096,blocks=40,scale_mode=1):
    global _reuse_loaded
    if not _reuse_loaded:
        path=library_path('reuse')
        load_library(str(path)); _reuse_loaded=True; _loaded_paths['reuse']=path
    if u.requires_grad: raise ValueError('Experimental inference-only F1b has no backward')
    return torch.ops.h3_phaa_fusion_quant_reuse.swiglu_quantize(u,tile,blocks,scale_mode)

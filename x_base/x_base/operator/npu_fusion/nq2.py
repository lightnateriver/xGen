"""Experimental two-output NQ adapters, separate from the seven-output debug API."""
import os
from pathlib import Path
import torch
from .loader import load_library

_paths = {}


def library_path(cached=False):
    if cached in _paths:
        return _paths[cached]
    stem = 'h3_norm_affine_quant2' + ('_cached' if cached else '')
    env = 'H3_NQ2_CACHED_LIBRARY' if cached else 'H3_NQ2_LIBRARY'
    return Path(os.environ.get(env, Path(__file__).parent/'f2_nq2/build'/('lib'+stem+'.so'))).resolve()


def norm_affine_quant(x, weight, scale, shift, indices, *, eps=1e-5, blocks=40, cached=False):
    if any(v.requires_grad for v in [x, weight, scale, shift]):
        raise ValueError('Experimental inference only; detach parameters explicitly')
    if cached not in _paths:
        path = library_path(cached)
        load_library(str(path))
        _paths[cached] = path
    ns = torch.ops.h3_phaa_nq2_cached if cached else torch.ops.h3_phaa_nq2
    return ns.fused(x, weight, scale, shift, indices, eps, blocks)

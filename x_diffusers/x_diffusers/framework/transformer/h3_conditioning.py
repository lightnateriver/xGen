"""Request-scoped H3 condition projection; no cross-request tensors on models."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

import torch

logger = logging.getLogger(__name__)
_CURRENT = ContextVar('h3_request_conditioning', default=None)


def _version(value):
    try:
        return value._version
    except RuntimeError:  # Inference-mode tensors have no version counter.
        return None


@contextmanager
def request_conditioning(model):
    if _CURRENT.get() is not None:
        raise RuntimeError('Nested H3 condition precomputation is not supported')
    metrics = dict(projection_calls=0, computations=0, reuses=0,
                   peak_output_bytes=0, released_before_vae=False, cleared=False)
    state = dict(model=model, entries=[], metrics=metrics)
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        state['entries'].clear()
        metrics['cleared'] = True
        model._h3_conditioning_metrics = dict(metrics)  # Scalars only.
        _CURRENT.reset(token)
        logger.info('H3 request condition precompute: %s', metrics)


def project_conditioning(model, source):
    state = _CURRENT.get()
    if state is None or state['model'] is not model:
        raise RuntimeError('H3 condition precompute requires its managed request scope')
    if torch.is_grad_enabled():
        raise ValueError('H3 condition precompute is inference-only')
    metrics = state['metrics']
    metrics['projection_calls'] += 1
    version = _version(source)
    for index, entry in enumerate(state['entries']):
        if entry['source'] is not source:
            continue
        unchanged = (version == entry['version'] if version is not None
                     else torch.equal(source, entry['snapshot']))
        if unchanged:
            metrics['reuses'] += 1
            return entry['output']
        state['entries'].pop(index)
        break
    output = model.token_refiner(model.context_embedder(source.to(model.context_embedder.weight.dtype)))
    # Retain at most positive/negative condition branches. Changed conditions
    # are recomputed; holding the source prevents pointer/id reuse collisions.
    if len(state['entries']) == 2:
        state['entries'].pop(0)
    state['entries'].append(dict(source=source, version=version,
        snapshot=source.clone() if version is None else None, output=output))
    metrics['computations'] += 1
    metrics['peak_output_bytes'] = max(metrics['peak_output_bytes'],
        sum(e['output'].numel()*e['output'].element_size() for e in state['entries']))
    return output


def release_conditioning(model):
    """Release finished DiT conditions before the existing VAE memory trim."""
    state = _CURRENT.get()
    if state is not None and state['model'] is model:
        state['entries'].clear()
        state['metrics']['released_before_vae'] = True

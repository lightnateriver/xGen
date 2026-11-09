"""Optional native FeedForward chunking for H3's replicated input refiner."""
from functools import wraps
import logging

import torch

logger = logging.getLogger(__name__)


def install_ff_chunking(model, chunk_size):
    """Bound token-wise FFN temporaries without changing attention or weights."""
    from diffusers.models.attention import _chunked_feed_forward

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('H3 input refiner FFN chunk size must be a positive integer')
    if hasattr(model, '_h3_refiner_ff_chunking'):
        raise ValueError('H3 input refiner FFN chunking is already installed')
    metrics = dict(chunk_size=chunk_size, modules=len(model.token_refiner.refiner_blocks),
                   calls=0, completed=0, chunks_completed=0, max_input_tokens=0)

    def wrap(original):
        @wraps(original)
        def forward(hidden_states, *args, **kwargs):
            if torch.is_grad_enabled():
                raise ValueError('Experimental H3 refiner FFN chunking is inference-only')
            if hidden_states.ndim != 3:
                raise ValueError('H3 input refiner FFN expects batch/token/channel input')
            tokens = hidden_states.shape[1]
            metrics['calls'] += 1
            metrics['max_input_tokens'] = max(tokens, metrics['max_input_tokens'])
            if tokens <= chunk_size:
                output = original(hidden_states, *args, **kwargs)
                metrics['chunks_completed'] += 1
            else:
                prefix = tokens // chunk_size * chunk_size
                def ff(value):
                    result = original(value, *args, **kwargs)
                    metrics['chunks_completed'] += 1
                    return result
                output = _chunked_feed_forward(ff, hidden_states[:, :prefix], 1, chunk_size)
                if prefix != tokens:
                    tail = ff(hidden_states[:, prefix:])
                    output = torch.cat((output, tail), dim=1)
            metrics['completed'] += 1
            return output
        return forward

    for block in model.token_refiner.refiner_blocks:
        block.ff.forward = wrap(block.ff.forward)
    model._h3_refiner_ff_chunking = metrics
    logger.info('Experimental H3 input refiner FFN chunking=%d tokens, blocks=%d; attention/weights unchanged',
                chunk_size, metrics['modules'])

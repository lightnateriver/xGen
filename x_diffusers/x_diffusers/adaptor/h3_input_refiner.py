"""Opt-in H3 input Token Refiner Ulysses/cuDNN, reusing xGen SP and FP8 ops."""
from functools import wraps
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)
PAD_KEY = 'h3_input_refiner'
PROFILES = ('none', 'cudnn-sp', 'cudnn-sp-fp8')


class RefinerSPAttention:
    """BF16 full-context attention with heads distributed over the SP group."""
    def __init__(self, group, backend, metrics):
        self.group, self.backend, self.metrics = group, backend, metrics

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from x_base import all_to_all_before_attn, all_to_all_after_attn

        if rotary_emb is not None or attention_mask is not None or attn.fused_projections:
            raise ValueError('H3 input refiner SP expects unfused QKV, no RoPE and no mask')
        query, key, value = (layer(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
                             for layer in (attn.to_q, attn.to_k, attn.to_v))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
            raise ValueError('H3 input refiner cuDNN attention requires BF16 QKV')
        self.metrics['attention_calls'] += 1
        if self.group is not None:
            self.metrics['logical_sp_payload_bytes'] += 4 * query.numel() * query.element_size()
        # Native helper removes tail padding BEFORE attention, so padded tokens
        # never enter softmax. Its inverse restores padding before the exchange.
        if self.group is not None:
            query, key, value = (all_to_all_before_attn(tensor, self.group, pad_key=PAD_KEY)
                                 for tensor in (query, key, value))
        query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
        output = (self.backend(query, key, value) if self.backend is not None
                  else F.scaled_dot_product_attention(query, key, value))
        output = output.transpose(1, 2).contiguous()
        if self.group is not None:
            output = all_to_all_after_attn(output, self.group, pad_key=PAD_KEY)
        output = output.flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](output))


def convert_ffn(refiner, group=None):
    """Keep BF16 module interfaces; use global activation amax across SP ranks."""
    from x_base.operator.h3_fp8 import H3FP8Linear

    converted = 0
    for block in refiner.refiner_blocks:
        for parent, name in ((block.ff.net[0], 'proj'), (block.ff.net, '2')):
            original = getattr(parent, name)
            if not isinstance(original, torch.nn.Linear):
                raise ValueError('H3 input refiner FP8 expects original FFN Linear modules')
            packed = H3FP8Linear(original.weight, original.bias)
            packed.activation_group = group
            setattr(parent, name, packed)
            converted += 1
    return converted


def install_input_refiner(model, profile, *, sp=None, attention=None, ffn=None):
    from x_base import gather_sequence, set_pad, split_sequence
    from x_base.operator.h3_dense_cuda import H3DenseCUDA

    if profile not in (*PROFILES[1:], 'custom'):
        raise ValueError('Unknown H3 input refiner profile')
    sp = True if sp is None else sp
    attention = 'cudnn' if attention is None else attention
    ffn = ('fp8' if profile == 'cudnn-sp-fp8' else 'bf16') if ffn is None else ffn
    refiner = model.token_refiner
    if hasattr(refiner, '_h3_sp_metrics'):
        raise ValueError('H3 input refiner SP is already installed')
    # CPU weight preparation precedes enable_parallel() in the normal loader.
    # Bind the existing SP group on the first forward, after xGen creates it.
    world = dist.get_world_size()
    if world != 8 or (sp and any(block.attn.heads % world for block in refiner.refiner_blocks)):
        raise ValueError('H3 input refiner SP requires eight ranks and divisible attention heads')
    if any(p.dtype != torch.bfloat16 for p in refiner.parameters()):
        raise ValueError('H3 input refiner SP expects original BF16 parameters')
    metrics = dict(profile=profile, sp_size=world if sp else 1, communication_dtype='bfloat16' if sp else 'none',
        attention_backend=attention, ffn_precision=ffn,
        calls=0, completed=0, attention_calls=0, max_input_tokens=0, local_tokens=0,
        last_input_tokens=0, last_padding=0, fp8_ffn_projections=0, logical_sp_payload_bytes=0)
    if ffn == 'fp8':
        metrics['fp8_ffn_projections'] = convert_ffn(refiner)
    backend = H3DenseCUDA() if attention == 'cudnn' else None
    processors = []
    for block in refiner.refiner_blocks:
        processor = RefinerSPAttention(None, backend, metrics)
        block.attn.set_processor(processor)
        processors.append(processor)
    original = refiner.forward
    group = None
    bound = False

    @wraps(original)
    def forward(hidden_states):
        nonlocal group, bound
        if torch.is_grad_enabled():
            raise ValueError('H3 input refiner SP is inference-only')
        if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 3:
            raise ValueError('H3 input refiner SP expects CUDA BF16 batch/token/channel inputs')
        tokens = hidden_states.shape[1]
        if tokens < 1:
            raise ValueError('H3 input refiner cannot have an empty context')
        if sp and not bound:
            manager = getattr(model, '_h3_condition_parallel_manager', getattr(model, 'parallel_manager', None))
            if (manager is None or manager.sp_group is None
                    or dist.get_world_size(manager.sp_group) != world
                    or getattr(manager, 'tp_size', 1) != 1):
                raise ValueError('H3 input refiner requires the initialized native SP8 group')
            group = manager.sp_group
            for processor in processors:
                processor.group = group
            if ffn == 'fp8':
                for block in refiner.refiner_blocks:
                    block.ff.net[0].proj.activation_group = group
                    block.ff.net[2].activation_group = group
            bound = True
        pad = set_pad(PAD_KEY, tokens, group) if sp else 0
        metrics['calls'] += 1
        metrics['last_input_tokens'] = tokens
        metrics['last_padding'] = pad
        metrics['max_input_tokens'] = max(tokens, metrics['max_input_tokens'])
        local = split_sequence(hidden_states, group, dim=1, pad=pad) if sp else hidden_states
        metrics['local_tokens'] = local.shape[1]
        local = original(local)
        result = gather_sequence(local, group, dim=1, pad=pad) if sp else local
        metrics['completed'] += 1
        return result

    refiner.forward = forward
    refiner._h3_sp_metrics = metrics
    refiner._h3_cudnn_backend = backend
    model._h3_input_refiner_profile = profile
    logger.info('Experimental H3 input Token Refiner: profile=%s SP=%d wire=%s attention=%s FFN=%s FP8_projections=%d; main DiT unchanged',
                profile, metrics['sp_size'], metrics['communication_dtype'], attention, ffn, metrics['fp8_ffn_projections'])

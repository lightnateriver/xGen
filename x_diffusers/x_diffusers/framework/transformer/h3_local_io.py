"""Pure-Ulysses local packed input/output, sharing the existing H3 block loop.

Text refiner keeps its full text context. Only raw video/audio rows owned by
this CP rank are projected. Final FP32 predictions, not hidden states, gather.
No per-request tensors are cached on the model.
"""
from dataclasses import dataclass

import torch

from x_base import gather_sequence, set_pad


@dataclass(frozen=True)
class LocalSpan:
    start: int
    length: int
    valid: int

    @classmethod
    def for_rank(cls, total, peers, rank):
        if total < 1 or peers < 1 or not 0 <= rank < peers:
            raise ValueError('Invalid H3 local sequence layout')
        length = (total + peers - 1) // peers
        start = rank * length
        return cls(start, length, max(0, min(length, total-start)))

    def slice_padded(self, tensor, dim=0, value=0):
        local = tensor.narrow(dim, min(self.start, tensor.shape[dim]), self.valid)
        if self.valid == self.length:
            return local.contiguous()
        shape = list(local.shape)
        shape[dim] = self.length-self.valid
        return torch.cat((local, tensor.new_full(shape, value)), dim=dim)

    def indices(self, packed_indices):
        source = ((packed_indices >= self.start) & (packed_indices < self.start+self.valid)).nonzero().flatten()
        return source, packed_indices.index_select(0, source)-self.start


def prepare_local(model, video, audio, text, positions, text_indices, video_indices, audio_indices):
    manager = model.parallel_manager
    peers = manager.sp_size
    rank = manager.sp_rank if peers > 1 else 0
    span = LocalSpan.for_rank(len(positions), peers, rank)
    for inputs, indices in ((video,video_indices),(audio,audio_indices),(text,text_indices)):
        if indices.ndim != 1 or len(indices) != inputs.shape[1]:
            raise ValueError('H3 packed modality indices must match the input rows')
        # Filtering to this CP span must not silently discard invalid global
        # indices. On CUDA the assertion does not synchronize valid requests.
        torch._assert_async(((indices >= 0) & (indices < len(positions))).all(),
                            'H3 packed modality index is outside the sequence')
    if peers > 1:
        set_pad('mmh3_seq', len(positions), manager.sp_group)
    # The text refiner is a separate full-context transformer, not token-wise.
    if getattr(model, '_h3_precompute_conditioning', False):
        from .h3_conditioning import project_conditioning
        embeds = project_conditioning(model, text)
    else:
        embeds = model.token_refiner(model.context_embedder(text.to(model.context_embedder.weight.dtype)))
    hidden = embeds.new_zeros((embeds.shape[0], span.length, embeds.shape[-1]))
    source, target = span.indices(text_indices)
    hidden.index_copy_(1, target, embeds.index_select(1, source))
    del embeds
    for inputs, indices, projection in ((video, video_indices, model.proj_in),
                                        (audio, audio_indices, model.audio_proj_in)):
        source, target = span.indices(indices)
        local = inputs.index_select(1, source).to(projection.weight.dtype)
        hidden.index_copy_(1, target, projection(local).to(hidden.dtype))
    # SP keeps local RoPE; shared TP columns consume the full real sequence.
    shared = getattr(model, '_h3_shared_sequence_exchange', None)
    if shared is not None:
        shared.set_tokens(len(positions))
        rotary = model.rope(positions)
    else:
        rotary = model.rope(span.slice_padded(positions))
    return hidden, rotary, span


def gather_local_output(model, hidden, temb, span, timestep_indices, video_indices, audio_indices):
    local_indices = span.slice_padded(timestep_indices)
    hidden = model.norm_out(hidden, temb, local_indices).to(model.proj_out.weight.dtype)
    video = model.proj_out(hidden)
    audio = model.audio_proj_out(hidden.to(model.audio_proj_out.weight.dtype))
    if video.dtype != torch.float32 or audio.dtype != torch.float32:
        raise ValueError('H3 TP/CP final prediction heads must remain FP32')
    width = video.shape[-1]
    prediction = torch.cat((video, audio), dim=-1)
    manager = model.parallel_manager
    if manager.sp_size > 1:
        pad = span.length*manager.sp_size - len(timestep_indices)
        prediction = gather_sequence(prediction, manager.sp_group, dim=1, pad=pad)
    return (prediction[..., :width].index_select(1, video_indices),
            prediction[..., width:].index_select(1, audio_indices))

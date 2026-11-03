from __future__ import annotations

import os
import torch

from x_base.device import device_type

from .globals import enable_phaa, set_phaa_split_num


def phaa_on_pipe(pipe: DiffusionPipeline, transformer_2=False, transformer_3=False, phaa_split_num=1, *args, **kwargs):
    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Wan"):
        inject_phaa_for_wan(pipe.transformer, phaa_split_num)
        if transformer_2:
            inject_phaa_for_wan(pipe.transformer_2, phaa_split_num)
        if transformer_3:
            inject_phaa_for_wan(pipe.transformer_3, phaa_split_num)
    elif "MiniMax" in pipe_cls_name:
        inject_phaa_for_minimax_h3(pipe.transformer, phaa_split_num)
    set_phaa_split_num(phaa_split_num)
    enable_phaa()


def check_split_num_correctness(attn, split_num):
    """Validate PHAA split num for the given attention module.

    Unequal chunk sizes are supported by ``run_phaa_attention`` (e.g. 7 heads/dev
    with split_num=2 → chunks [4, 3]), so divisibility is no longer required.
    """
    world_size = attn.parallel_manager.sp_size
    heads_per_device = attn.heads // world_size
    if split_num is not None and split_num > heads_per_device:
        raise ValueError(
            f"PHAA split num {split_num} must not exceed per-device attn heads {heads_per_device}")


def inject_phaa_for_wan(transformer, split_num):
    comm_stream = torch.get_device_module(device_type()).Stream()
    for block in transformer.blocks:
        check_split_num_correctness(block.attn1, split_num)
        check_split_num_correctness(block.attn2, split_num)
        block.attn1.processor.comm_stream = comm_stream
        block.attn2.processor.comm_stream = comm_stream


def inject_phaa_for_minimax_h3(transformer, split_num):
    comm_stream = torch.get_device_module(device_type()).Stream()
    for block in transformer.transformer_blocks:
        if hasattr(block, "attn"):
            check_split_num_correctness(block.attn, split_num)
            block.attn.processor.comm_stream = comm_stream
"""Compatibility imports for the shared H3 visual sequence-parallel implementation."""

from ..framework.transformer.h3_vision_sp import PAD_KEY, VisionSP, full_bf16_accumulation, install_vision_sp

__all__ = ["PAD_KEY", "VisionSP", "full_bf16_accumulation", "install_vision_sp"]

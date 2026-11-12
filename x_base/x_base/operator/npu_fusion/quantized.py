"""Quantized rows paired with token scales and their logical tensor shape."""
from typing import NamedTuple
import torch


class QuantizedRows(NamedTuple):
    q: torch.Tensor
    scale: torch.Tensor
    shape: tuple

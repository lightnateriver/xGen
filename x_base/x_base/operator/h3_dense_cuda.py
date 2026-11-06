# SPDX-License-Identifier: Apache-2.0
"""BF16 cuDNN attention; xGen retains its existing SP/PHAA communication."""
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class H3DenseCUDA:
    def __init__(self):
        self.calls = 0

    def __call__(self, query, key, value, *, attn_mask=None, head_offset=0):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        self.calls += 1
        return output

# SPDX-License-Identifier: Apache-2.0
"""Model-local Sage dispatch using xGen's existing attention manager."""
from x_base.operator.attention import AttentionManager


class H3SageCUDA:
    def __init__(self):
        self.calls = 0
        self.kernel_calls = 0
        self.manager = AttentionManager()
        self.manager.configure_cuda_attention('sage')
        original = self.manager.cuda_attention

        def observe(*args, **kwargs):
            output = original(*args, **kwargs)
            self.kernel_calls += 1
            return output

        self.manager.cuda_attention = observe

    def __call__(self, query, key, value, *, attn_mask=None, head_offset=0):
        output = self.manager.attention(query, key, value, attn_mask=attn_mask,
                                        dropout_p=0.0, is_causal=False)
        self.calls += 1
        return output

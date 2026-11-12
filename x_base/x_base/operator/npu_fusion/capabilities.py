"""Implementation limits, independent of task duration or reference count.

Widths are bounded by the current UB/reduction implementations. PHAA geometry
is fixed by the layout kernels. Unsupported calls must use native operators.
"""
SWIGLU_MAX_INNER = 14336
NORM_MAX_WIDTH = 6144
ROPE_HEAD_DIM = 128
ROPE_ROTARY_DIM = 96
PHAA_SP = 4
PHAA_CHUNKS = 7
PHAA_HEADS = 56
PHAA_HEAD_DIM = 128
PHAA_CHUNK_HEADS = 2
MERGE_MAX_SEQUENCE = 1 << 20
VECTOR_BLOCKS = 40

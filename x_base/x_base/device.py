"""Process-wide accelerator selection before importing vendor extensions."""
import os


def device_type():
    backend = os.environ.get("XGEN_DEVICE", "npu")
    if backend not in ("npu", "cuda"):
        raise ValueError("XGEN_DEVICE must be npu or cuda")
    return backend

"""Release unused accelerator memory after an H3 video has been saved."""
import gc
import logging
import time

import torch

from x_base.device import device_type

logger = logging.getLogger(__name__)


def release_request_memory():
    """Keep resident weights; callers must first release their request outputs."""
    started = time.perf_counter()
    accelerator = torch.get_device_module(device_type())
    gc.collect()
    accelerator.synchronize()
    allocated = accelerator.memory_allocated()
    reserved = accelerator.memory_reserved()
    accelerator.empty_cache()
    accelerator.synchronize()
    logger.info(
        "H3_REQUEST_MEMORY_RELEASE device=%s seconds=%.3f allocated_before=%d "
        "allocated_after=%d reserved_before=%d reserved_after=%d",
        device_type(), time.perf_counter() - started, allocated,
        accelerator.memory_allocated(), reserved, accelerator.memory_reserved(),
    )

"""One-block lookahead with two bounded pinned/device slots and event ownership."""
from concurrent.futures import ThreadPoolExecutor
import functools
import torch


class BlockPrefetch:
    def __init__(self, holder, blocks):
        self.holder = holder
        self.metrics = holder.metrics
        self.entries = [holder.subset(block) for block in blocks]
        layout = lambda entries: [(tuple(m.shape), m.dtype) for _, m in entries]
        if any(layout(entries) != layout(self.entries[0]) for entries in self.entries):
            raise ValueError("Two-slot prefetch requires matching transformer block layouts")
        self.device = torch.npu.current_device()
        self.stream = torch.npu.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refiner-h2d")
        self.host = [[torch.empty_like(m, device="cpu", pin_memory=True)
                      for _, m in self.entries[0]] for _ in range(2)]
        self.buffers = None
        self.done = [None, None]
        self.futures = {}
        for index, block in enumerate(blocks):
            forward = block.forward
            @functools.wraps(forward)
            def run(*args, _index=index, _forward=forward, **kwargs):
                return self.run(_index, _forward, args, kwargs)
            block.forward = run

    @torch.inference_mode()
    def _load(self, index):
        torch.npu.set_device(self.device)
        slot = index % 2
        if self.done[slot] is not None:
            self.done[slot].synchronize()
        entries = self.entries[index]
        size = sum(m.numel()*m.element_size() for _, m in entries)
        with self.metrics.host_span("refiner.weights.pinned_stage", bytes=size, block=index):
            for target, (_, master) in zip(self.host[slot], entries):
                target.copy_(master)
        with torch.npu.stream(self.stream):
            with self.metrics.span("refiner.weights.prefetch_h2d", sync=False,
                                   stream=self.stream, bytes=size, block=index):
                for target, source in zip(self.buffers[slot], self.host[slot]):
                    target.copy_(source, non_blocking=True)
            ready = torch.npu.Event()
            ready.record(self.stream)
        return ready

    def begin_step(self):
        if self.buffers is None:
            self.buffers = [[torch.empty_like(m, device="npu") for _, m in self.entries[0]]
                            for _ in range(2)]
        if self.futures:
            raise RuntimeError("Previous prefetch step was not drained")
        self.futures[0] = self.executor.submit(self._load, 0)

    def run(self, index, forward, args, kwargs):
        slot = index % 2
        with self.metrics.span("refiner.weights.block_h2d_wait", sync=False, bytes=0, block=index):
            ready = self.futures.pop(index).result()
            torch.npu.current_stream().wait_event(ready)
        for (tensor, _), buffer in zip(self.entries[index], self.buffers[slot]):
            tensor.data = buffer
        if index+1 < len(self.entries):
            self.futures[index+1] = self.executor.submit(self._load, index+1)
        try:
            with self.metrics.span("refiner.weights.block_compute", sync=False, block=index):
                return forward(*args, **kwargs)
        finally:
            done = torch.npu.Event()
            done.record()
            self.done[slot] = done
            # Device storage stays owned by the slot until its done event completes.
            for tensor, master in self.entries[index]:
                tensor.data = master

    def end_step(self):
        for future in self.futures.values():
            future.result()
        self.futures.clear()
        for event in self.done:
            if event is not None:
                event.synchronize()

    def release(self):
        self.end_step()
        self.buffers = None
        self.done = [None, None]

"""Opt-in H3 resource/metrics helper; not installed on legacy pipelines."""
import functools
import torch

from ..device import device_type


class CpuWeights:
    """Materialized immutable CPU masters, reused without disk or D2H copy-back."""
    def __init__(self, model, name, metrics, masters=None):
        self.model, self.name, self.metrics = model, name, metrics
        self.backend = device_type()
        self.entries = []
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.device.type == "meta":
                raise ValueError(f"Unexpected meta tensor in {name}")
            if masters is not None and id(tensor) in masters:
                master=masters[id(tensor)]
                if master.device.type!="cpu" or master.shape!=tensor.shape or master.dtype!=tensor.dtype:
                    raise ValueError("Incompatible immutable CPU master")
            else:
                master = tensor.detach().to("cpu", copy=True)
            self.entries.append((tensor, master))
        self.nbytes = sum(t.numel()*t.element_size() for _, t in self.entries)
        self.offload()

    def subset(self, module):
        ids = {id(t) for t in list(module.parameters()) + list(module.buffers())}
        return [(t, m) for t, m in self.entries if id(t) in ids]

    def onload(self, entries=None, label=None):
        selected = self.entries if entries is None else entries
        moved = [(t, m) for t, m in selected if t.device.type != self.backend]
        size = sum(m.numel()*m.element_size() for _, m in moved)
        with self.metrics.span(label or self.name+".h2d_wait", bytes=size):
            for tensor, master in moved:
                tensor.data = master.to(self.backend, non_blocking=False)

    def offload(self, entries=None, label=None):
        selected = self.entries if entries is None else entries
        with self.metrics.span(label or self.name+".release", bytes=0):
            getattr(torch, self.backend).synchronize()
            for tensor, master in selected:
                tensor.data = master

    def stream_blocks(self, mode="sync"):
        blocks = self.model.transformer_blocks
        if hasattr(self, "block_forwards"):
            if hasattr(self, "prefetch"):
                self.prefetch.release()
                self.prefetch.executor.shutdown(wait=True)
                del self.prefetch
            for block, forward in zip(blocks, self.block_forwards):
                block.forward = forward
        else:
            self.block_forwards = [block.forward for block in blocks]
        block_ids = {id(t) for block in blocks for t in list(block.parameters())+list(block.buffers())}
        self.nonblocks = [(t, m) for t, m in self.entries if id(t) not in block_ids]
        if mode == "resident":
            return
        if mode == "prefetch":
            from .block_prefetch import BlockPrefetch
            self.prefetch = BlockPrefetch(self, blocks)
            return
        if mode != "sync":
            raise ValueError("Unknown block loading mode: "+mode)
        for index, block in enumerate(blocks):
            selected = self.subset(block)
            forward = block.forward

            @functools.wraps(forward)
            def run(*args, _forward=forward, _entries=selected, _index=index, **kwargs):
                self.onload(_entries, self.name+".block_h2d_wait")
                try:
                    with self.metrics.span(self.name+".block_compute", block=_index):
                        return _forward(*args, **kwargs)
                finally:
                    self.offload(_entries, self.name+".block_release")
            block.forward = run

    def fingerprint(self):
        return dict(model_id=id(self.model), cpu_storage=[m.data_ptr() for _, m in self.entries], bytes=self.nbytes)

    def close(self):
        """Drain owned transfers and restore block methods before releasing storage."""
        if hasattr(self, "block_forwards"):
            self.stream_blocks("resident")
        self.offload()

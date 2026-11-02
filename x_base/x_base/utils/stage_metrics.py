"""Opt-in H3 resource/metrics helper; not installed on legacy pipelines."""
from contextlib import contextmanager
import time
import torch


class Telemetry:
    def __init__(self):
        self.active = False
        self.rows = []
        self.events = []
        self.origin = time.perf_counter()

    @contextmanager
    def span(self, name, sync=True, stream=None, **attributes):
        if not self.active:
            yield
            return
        if sync:
            torch.npu.synchronize()
        start = time.perf_counter()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            if sync:
                torch.npu.synchronize()
            row = dict(name=name, host_seconds=time.perf_counter()-start,
                       start_seconds=start-self.origin,
                       allocated_after=torch.npu.memory_allocated(), **attributes)
            self.rows.append(row)
            self.events.append((row, begin, end))

    @contextmanager
    def host_span(self, name, **attributes):
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.active:
                self.rows.append(dict(name=name, host_seconds=time.perf_counter()-start,
                    start_seconds=start-self.origin, **attributes))

    def resolve(self):
        torch.npu.synchronize()
        for row, begin, end in self.events:
            row["device_envelope_seconds"] = begin.elapsed_time(end) / 1000.
        return self.rows

    def reset(self):
        self.rows, self.events = [], []
        self.origin = time.perf_counter()
        self.active = True

    def host_snapshot(self):
        """Bind background work to these rows even after the next reset()."""
        recorder=Telemetry()
        recorder.rows,recorder.origin,recorder.active=self.rows,self.origin,True
        return recorder

"""Host-side failure propagation for distributed workers with blocking device work."""
from datetime import timedelta
import logging
import os
import threading

logger = logging.getLogger(__name__)


class DistributedWorkerFailed(RuntimeError):
    """The worker group cannot be reused after a rank or control-channel failure."""


class DistributedFailureGuard:
    """Terminate the worker group if any rank reports failure or loses its heartbeat.

    All ranks create the guard in the same order and enter each guarded operation.
    A dedicated Gloo group and thread stay independent of device collectives.
    Successful exit waits for every rank; the timeout bounds a control collective,
    not model execution. Fatal exit deliberately avoids device-group destruction,
    which can itself block after a failed collective.
    """

    def __init__(self, *, timeout=10.0, interval=0.2, exit_fn=None):
        import torch
        import torch.distributed as dist

        if timeout <= 0 or not 0 < interval < timeout:
            raise ValueError("Require 0 < heartbeat interval < control timeout")
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise ValueError("DistributedFailureGuard requires an initialized multi-rank group")
        if not dist.is_gloo_available():
            raise RuntimeError("Distributed failure monitoring requires the Gloo CPU backend")
        self.torch, self.dist = torch, dist
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout))
        self.interval = interval
        self._exit = os._exit if exit_fn is None else exit_fn
        self._failed = False
        self._closed = False
        self._active = False
        self._owner_thread = None
        self._depth = 0
        self._error = threading.Event()
        self._done = threading.Event()
        self._wakeup = threading.Event()
        self._thread = None

    def __enter__(self):
        if self._failed or self._closed:
            raise DistributedWorkerFailed("Distributed worker is unavailable; restart required")
        if self._active:
            if self._owner_thread != threading.get_ident():
                raise RuntimeError("DistributedFailureGuard cannot be shared concurrently")
            self._depth += 1
            return self
        self._active = True
        self._owner_thread = threading.get_ident()
        self._depth = 1
        self._error.clear()
        self._done.clear()
        self._wakeup.clear()
        self._thread = threading.Thread(target=self._monitor, name="xgen-rank-health", daemon=True)
        self._thread.start()
        return self

    def fail(self, error):
        """Publish failure before potentially blocking request cleanup begins."""
        self._failed = True
        self._error.set()
        self._wakeup.set()
        logger.error("Distributed rank %d failed: %s", self.rank, error,
                     exc_info=(type(error), error, error.__traceback__))

    def _terminate(self, reason):
        self._failed = True
        logger.critical("Distributed worker exiting rank=%d reason=%s restart_required=true", self.rank, reason)
        self._exit(1)

    def _monitor(self):
        try:
            while True:
                status = self.torch.tensor(
                    [int(self._error.is_set()), int(self._done.is_set())],
                    dtype=self.torch.int32, device="cpu",
                )
                self.dist.all_reduce(status, op=self.dist.ReduceOp.SUM, group=self.group)
                failures, finished = status.tolist()
                if failures:
                    self._terminate("rank_failure")
                    return
                if finished == self.world_size:
                    return
                self._wakeup.wait(self.interval)
                self._wakeup.clear()
        except Exception:
            # A dead peer or a stalled heartbeat must also stop surviving ranks.
            self._terminate("control_channel_failure")

    def __exit__(self, kind, error, traceback):
        if error is not None and not self._error.is_set():
            self.fail(error)
        if self._depth > 1:
            self._depth -= 1
            return False
        try:
            self._done.set()
            self._wakeup.set()
            self._thread.join()
        finally:
            self._active = False
            self._owner_thread = None
            self._depth = 0
        if self._failed and error is None:
            raise DistributedWorkerFailed("A distributed rank failed; restart the worker group")
        return False

    def close(self):
        """Release an inactive, healthy control group when its owner shuts down."""
        if self._active:
            raise RuntimeError("Cannot close an active distributed failure guard")
        if not self._closed and not self._failed:
            self.dist.destroy_process_group(self.group)
        self._closed = True

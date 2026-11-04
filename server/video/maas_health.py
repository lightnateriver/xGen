"""Read-only health snapshots for the optional MaaS execution profile."""

from dataclasses import dataclass, replace
from time import monotonic


_PHASE_TIMEOUTS = {"loading": 600, "generating": 1200, "saving": 180}


@dataclass(frozen=True)
class MaaSHealth:
    phase: str = "idle"
    since: float = 0.0
    model_loaded: bool = False
    group_confirmed: bool = False
    warmed: bool = False
    failed: bool = False

    def advance(self, phase, **changes):
        return replace(self, phase=phase, since=monotonic(), **changes)

    def describe(self, initialized, draining, expected, world_size):
        timeout = _PHASE_TIMEOUTS.get(self.phase)
        timed_out = timeout is not None and self.since > 0 and monotonic() - self.since >= timeout
        group_state = "confirmed" if self.group_confirmed else "unverified"
        if draining:
            state, reason = "draining", "shutdown_requested"
        elif self.failed or timed_out:
            state, reason = "degraded", "worker_failed" if self.failed else "progress_timeout"
            group_state = "failed" if self.failed else "unverified"
        elif not initialized or expected != world_size or not self.group_confirmed:
            state, reason = "starting", "workers_unverified"
        elif not self.model_loaded:
            state, reason = "starting", "model_loading" if self.phase == "loading" else "model_not_loaded"
        elif not self.warmed:
            state, reason = "starting", "warmup_pending"
        else:
            state, reason = "ready", None
        return {"state": state, "ready": state == "ready", "reason": reason,
                "phase": self.phase, "model_loaded": self.model_loaded,
                "workers": {"expected": expected, "group_state": group_state}}

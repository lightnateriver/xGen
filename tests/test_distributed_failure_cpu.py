"""Real two-process Gloo tests; no accelerator, models or external network."""
from datetime import timedelta
import importlib.util
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest

try:
    import torch.distributed as dist
except ImportError:
    dist = None

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, scenario, directory):
    # Capture expected fatal-worker diagnostics rather than polluting test output.
    with (Path(directory) / f"rank{rank}.log").open("w", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        dist.init_process_group(
            "gloo", rank=rank, world_size=2,
            init_method=(Path(directory) / "rendezvous").as_uri(),
            timeout=timedelta(seconds=8),
        )
        spec = importlib.util.spec_from_file_location(
            "worker_failure_guard", ROOT / "x_base/x_base/utils/distributed.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        guard = module.DistributedFailureGuard(timeout=1.0, interval=0.03)
        if scenario == "success":
            for iteration in range(2):
                with guard:
                    # A healthy, slow rank must not hit a whole-request deadline.
                    with guard:
                        if rank == 1:
                            time.sleep(1.2)
                (Path(directory) / f"rank{rank}-done{iteration}").touch()
            guard.close()
            dist.destroy_process_group()
            return

        with guard:
            (Path(directory) / f"rank{rank}-entered").touch()
            # Ensure both ranks are inside the guarded request before injecting failure.
            dist.barrier()
            if scenario == "writer" and rank == 0:
                raise OSError("injected video writer failure")
            if scenario == "metrics" and rank == 1:
                raise OSError("injected metric write failure")
            if scenario == "cleanup" and rank == 0:
                try:
                    pass
                finally:
                    raise OSError("injected writer cleanup failure")
            if scenario == "peer_exit" and rank == 1:
                os._exit(17)
            # This stands in for a rank blocked in a device collective. It uses
            # a different process group from the independent health thread.
            dist.barrier()
        (Path(directory) / f"rank{rank}-unexpected-success").touch()


@unittest.skipUnless(dist is not None and dist.is_gloo_available(), "requires CPU PyTorch with Gloo")
class DistributedFailureTests(unittest.TestCase):
    def run_scenario(self, scenario):
        with tempfile.TemporaryDirectory(prefix="xgen-gloo-regression-") as directory:
            context = multiprocessing.get_context("spawn")
            workers = [context.Process(target=worker, args=(rank, scenario, directory)) for rank in (0, 1)]
            started = time.monotonic()
            try:
                for process in workers:
                    process.start()
                deadline = started + 20
                for process in workers:
                    process.join(max(0, deadline - time.monotonic()))
                alive = [process.pid for process in workers if process.is_alive()]
                logs = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in Path(directory).glob("*.log"))
                self.assertFalse(alive, f"Worker hang in {scenario}: {alive}\n{logs[-4000:]}")
                codes = [process.exitcode for process in workers]
                if scenario == "success":
                    self.assertEqual(codes, [0, 0], logs[-4000:])
                    self.assertEqual(len(list(Path(directory).glob("*-done*"))), 4)
                else:
                    self.assertEqual(len(list(Path(directory).glob("*-entered"))), 2, logs[-4000:])
                    self.assertTrue(all(code != 0 for code in codes), (scenario, codes, logs[-4000:]))
                    self.assertFalse(list(Path(directory).glob("*-unexpected-success")))
                    self.assertIn("restart_required=true", logs)
                    if scenario == "peer_exit":
                        self.assertEqual(codes[1], 17)
            finally:
                for process in workers:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_success_waits_for_all_ranks_and_reuses_guard(self):
        self.run_scenario("success")

    def test_rank_zero_writer_failure_exits_peer_in_collective(self):
        self.run_scenario("writer")

    def test_nonzero_rank_metrics_failure_exits_entire_group(self):
        self.run_scenario("metrics")

    def test_cleanup_failure_exits_entire_group(self):
        self.run_scenario("cleanup")

    def test_hard_peer_exit_stops_surviving_rank(self):
        self.run_scenario("peer_exit")

if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from unittest.mock import patch

import bench_h3_npu as bench


class WorkerTreeTests(unittest.TestCase):
    def test_tracks_independent_rank_sessions_and_survivors(self):
        running = (
            "20 1 20 S\n"       # torchrun
            "21 20 21 Sl\n"    # rank in its own session
            "22 20 22 Sl\n"    # another rank
            "23 21 21 S\n"     # encoder child
            "24 20 24 Z\n"     # zombie
            "30 1 30 R\n"      # unrelated process
        )
        after_exit = "21 1 21 Sl\n23 21 21 S\n22 1 22 Z\n30 1 30 R\n"
        seen = set()
        with patch.object(bench.subprocess, "check_output", side_effect=[running, after_exit]):
            self.assertEqual(bench.group_pids(20, seen), [20, 21, 22, 23])
            self.assertEqual(bench.group_pids(20, seen), [21, 23])

    def test_sample_records_start_and_read_duration(self):
        meminfo = {"MemAvailable": "1 kB", "MemTotal": "2 kB", "Cached": "0 kB"}
        with (
            patch.object(bench.time, "monotonic", side_effect=[100, 113]),
            patch.object(bench.subprocess, "check_output", return_value=""),
            patch.object(bench, "fields", return_value=meminfo),
        ):
            sample = bench.sample()
        self.assertEqual(sample["monotonic_s"], 100)
        self.assertEqual(sample["sample_duration_s"], 13)


if __name__ == "__main__":
    unittest.main()

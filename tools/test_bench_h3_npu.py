import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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


class GroupMetadataTests(unittest.TestCase):
    def test_busy_preflight_publishes_requested_recipe_without_launching(self):
        for resolution, attention, features, refiner_steps in (
            ("768p", "vc", "all", 0), ("2k", "laser", "none", 3),
        ):
            with self.subTest(resolution=resolution), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                prompt = root / "prompt.txt"
                prompt.write_text("benchmark metadata contract", encoding="utf-8")
                args = SimpleNamespace(output=root, base_adaln=root / "t2va-21step.safetensors",
                                       ref_adaln=root / "ref2va-21step.safetensors", seed=42,
                                       hashes={prompt: hashlib.sha256(prompt.read_bytes()).hexdigest()}, warmup_per_case=False, repeat=1)
                cases = [(f"t2va-5s-{resolution}", "t2va", 5, resolution, [], prompt)]
                busy = dict(hbm={f"{index}:0": dict(name="Ascend910B3") for index in range(8)},
                            idle_cards=7, active_pids=[123])
                publish = Mock()
                with patch.object(bench, "sample", return_value=busy), patch.object(bench.subprocess, "Popen") as launch:
                    result = bench.run_group(f"t2va-{resolution}", cases, args, publish)
                launch.assert_not_called()
                self.assertEqual(result["error"], "eight_idle_cards_required")
                self.assertEqual([call.args[0] for call in publish.call_args_list], ["start", "end"])
                config = publish.call_args_list[0].args[1]
                self.assertEqual((config["attention"], config["h3_opts"], config["refiner_steps"]),
                                 (attention, features, refiner_steps))
                self.assertEqual((config["base_steps"], config["dit_linear"]), (20, "int8"))
                saved = json.loads((root / "groups" / f"t2va-{resolution}" / "config.json").read_text())
                self.assertEqual(saved, config)
                report = bench.markdown_event("start", config)
                self.assertIn(f" · {attention} · 融合请求 `{features}`", report)
                historical = {key: value for key, value in config.items() if key != "h3_opts"}
                self.assertIn("融合请求 `未记录`", bench.markdown_event("start", historical))


if __name__ == "__main__":
    unittest.main()

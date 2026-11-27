"""Real CPU tensor checks for reusable TP primitives; run with CPU PyTorch."""

import ast

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x_base/x_base"))

from tensor_parallelism import projections as tp


class ProjectionTests(unittest.TestCase):
    def test_uneven_partitions_cover_sequence_and_keep_halos_in_bounds(self):
        path = ROOT / "x_base/x_base/sequence_parallelism/comm.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "partition"
        )
        for total in (1, 17, 22):
            for peers in (2, 4):
                ownership = []
                for rank in range(peers):
                    dist = SimpleNamespace(
                        get_world_size=lambda group: peers, get_rank=lambda group: rank
                    )
                    namespace = dict(dist=dist)
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]), str(path), "exec"
                        ),
                        namespace,
                    )
                    lo, hi, halo_lo, halo_hi = namespace["partition"](
                        total, 3, object()
                    )
                    ownership.extend(range(lo, hi))
                    self.assertTrue(0 <= halo_lo <= lo <= hi <= halo_hi <= total)
                self.assertEqual(ownership, list(range(total)))

    def test_uneven_gather_removes_transport_padding_including_empty_rank(self):
        path = ROOT / "x_base/x_base/sequence_parallelism/comm.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "gather_time"
        )
        for total, peers, unit in ((3, 4, 2), (17, 4, 1)):
            value = torch.arange(total * unit, dtype=torch.float32).reshape(1, 1, -1)
            shards = [
                value[
                    ...,
                    total * rank // peers * unit : total * (rank + 1) // peers * unit,
                ]
                for rank in range(peers)
            ]
            width = max(shard.shape[-1] for shard in shards)

            def gather(received, payload, group):
                torch.testing.assert_close(
                    payload, F.pad(shards[0], (0, width - shards[0].shape[-1]))
                )
                for destination, shard in zip(received, shards):
                    destination.copy_(F.pad(shard, (0, width - shard.shape[-1])))

            namespace = dict(
                torch=torch,
                dist=SimpleNamespace(
                    get_world_size=lambda group: peers, all_gather=gather
                ),
            )
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
            actual = namespace["gather_time"](shards[0], total, object(), unit)
            self.assertTrue(torch.equal(actual, value))

    def test_fused_projection_partition_keeps_each_part_and_source_unchanged(self):
        weight = torch.arange(48 * 48).reshape(48, 48)
        original = weight.clone()
        for axis in (0, 1):
            for parts in (1, 2, 3):
                for peers in (2, 4):
                    width = weight.shape[axis] // parts
                    chunk = width // peers
                    for rank in range(peers):
                        indices = torch.cat(
                            [
                                torch.arange(
                                    part * width + rank * chunk,
                                    part * width + (rank + 1) * chunk,
                                )
                                for part in range(parts)
                            ]
                        )
                        actual = tp.shard_fused_tensor(
                            weight, axis=axis, rank=rank, world_size=peers, parts=parts
                        )
                        self.assertTrue(
                            torch.equal(actual, weight.index_select(axis, indices))
                        )
        self.assertTrue(torch.equal(weight, original))

    def test_invalid_projection_geometry_fails_before_any_mutation(self):
        weight = torch.ones(7, 5)
        for options in (
            {"axis": 0, "rank": 0, "world_size": 2},
            {"axis": 1, "rank": 2, "world_size": 2},
            {"axis": 2, "rank": 0, "world_size": 1},
            {"axis": 0, "rank": 0, "world_size": True},
        ):
            with self.assertRaises(ValueError):
                tp.shard_fused_tensor(weight, **options)
        self.assertTrue(torch.equal(weight, torch.ones(7, 5)))

    def test_column_shards_reconstruct_dense_qkv_and_swiglu_outputs(self):
        torch.manual_seed(7)
        for parts in (1, 2, 3):
            for peers in (2, 4):
                original = torch.nn.Linear(16, 48)
                value = torch.randn(2, 5, 16)
                outputs = []
                for rank in range(peers):
                    manager = SimpleNamespace(
                        tp_group=object(), tp_size=peers, tp_rank=rank
                    )
                    with patch("torch.distributed.get_world_size", return_value=peers):
                        layer = tp.shard_linear(
                            original, manager, column=True, parts=parts
                        )
                    self.assertIs(layer.tp_group, manager.tp_group)
                    outputs.append(layer(value))
                width = 48 // parts // peers
                reconstructed = torch.cat(
                    [
                        output[..., part * width : (part + 1) * width]
                        for part in range(parts)
                        for output in outputs
                    ],
                    dim=-1,
                )
                torch.testing.assert_close(reconstructed, original(value))

    def test_row_shard_reduces_before_adding_bias_once(self):
        torch.manual_seed(13)
        original = torch.nn.Linear(16, 12)
        value = torch.randn(2, 16)
        expected_sum = F.linear(value, original.weight, None)
        for rank in range(2):
            group = object()
            manager = SimpleNamespace(tp_group=group, tp_size=2, tp_rank=rank)
            local = value.chunk(2, dim=-1)[rank]

            def reduce(partial, received_group):
                self.assertIs(received_group, group)
                torch.testing.assert_close(
                    partial,
                    F.linear(local, original.weight.chunk(2, dim=1)[rank], None),
                )
                return expected_sum

            with patch("torch.distributed.get_world_size", return_value=2):
                layer = tp.shard_linear(
                    original, manager, column=False, reduction=reduce
                )
                torch.testing.assert_close(layer(local), original(value))

    def test_sequence_column_preserves_projection_and_exchange_order(self):
        value, full = torch.ones(1, 2, 3), torch.ones(1, 4, 3)
        exchange = SimpleNamespace(
            gather=Mock(return_value=full),
            scatter_columns=Mock(side_effect=lambda result, parts: result[:, :2]),
        )
        layer = tp.SequenceColumn(
            torch.nn.Linear(3, 6), exchange, parts=3, restore_sequence=True
        )
        result = layer(value)
        exchange.gather.assert_called_once_with(value)
        self.assertEqual(exchange.scatter_columns.call_args.args[1], 3)
        torch.testing.assert_close(result, layer.projection(full)[:, :2])

    def test_sequence_row_selects_rank_input_and_adds_bias_after_exchange(self):
        full = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
        projection = torch.nn.Linear(3, 2, bias=False)
        reduced = torch.ones(1, 1, 2)
        exchange = SimpleNamespace(
            rank=1,
            gather=Mock(return_value=full),
            reduce_scatter=Mock(return_value=reduced),
        )
        layer = tp.SequenceRow(
            projection,
            exchange,
            gather_input=True,
            bias=torch.tensor([2.0, 3.0]),
            partitions=2,
        )
        actual = layer(torch.zeros(1, 1, 6))
        torch.testing.assert_close(
            exchange.reduce_scatter.call_args.args[0], projection(full[..., 3:])
        )
        torch.testing.assert_close(actual, torch.tensor([[[3.0, 4.0]]]))



if __name__ == "__main__":
    unittest.main()

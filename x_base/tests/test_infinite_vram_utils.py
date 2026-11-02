"""
Unit tests for infinite_vram/utils.py.

Covers:
- foreach_copy_: quantization path (a4w4/a8w8 forces sequential copy),
  normal path (torch._foreach_copy_), AttributeError fallback
- print_gpu_memory: calls cuda memory stats
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.infinite_vram.utils import foreach_copy_, print_gpu_memory


class TestForeachCopy:
    def test_a4w4_path_uses_sequential_copy(self):
        dests = [torch.zeros(3), torch.zeros(3)]
        srcs = [torch.ones(3), torch.ones(3) * 2]
        with patch("x_base.infinite_vram.utils.infer_info") as mock_info:
            mock_info.use_matmul_a4w4 = True
            mock_info.use_matmul_a8w8 = False
            foreach_copy_(dests, srcs, non_blocking=True)
        assert torch.equal(dests[0], srcs[0])
        assert torch.equal(dests[1], srcs[1])

    def test_a8w8_path_uses_sequential_copy(self):
        dests = [torch.zeros(3)]
        srcs = [torch.ones(3)]
        with patch("x_base.infinite_vram.utils.infer_info") as mock_info:
            mock_info.use_matmul_a4w4 = False
            mock_info.use_matmul_a8w8 = True
            foreach_copy_(dests, srcs)
        assert torch.equal(dests[0], srcs[0])

    def test_normal_path_uses_foreach(self):
        dests = [torch.zeros(3)]
        srcs = [torch.ones(3)]
        with patch("x_base.infinite_vram.utils.infer_info") as mock_info:
            mock_info.use_matmul_a4w4 = False
            mock_info.use_matmul_a8w8 = False
            with patch("x_base.infinite_vram.utils.torch._foreach_copy_") as mock_foreach:
                foreach_copy_(dests, srcs, non_blocking=True)
                mock_foreach.assert_called_once_with(dests, srcs, non_blocking=True)

    def test_fallback_when_no_foreach(self):
        dests = [torch.zeros(3)]
        srcs = [torch.ones(3)]
        with patch("x_base.infinite_vram.utils.infer_info") as mock_info:
            mock_info.use_matmul_a4w4 = False
            mock_info.use_matmul_a8w8 = False
            with patch("x_base.infinite_vram.utils.torch._foreach_copy_", side_effect=AttributeError):
                foreach_copy_(dests, srcs)
        assert torch.equal(dests[0], srcs[0])


class TestPrintGpuMemory:
    def test_calls_cuda_stats(self):
        with patch("x_base.infinite_vram.utils.torch.cuda") as mock_cuda:
            mock_cuda.memory_allocated.return_value = 1024 * 1024 * 100
            mock_cuda.memory_reserved.return_value = 1024 * 1024 * 200
            print_gpu_memory("test")
            mock_cuda.memory_allocated.assert_called_once()
            mock_cuda.memory_reserved.assert_called_once()

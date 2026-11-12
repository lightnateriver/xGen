"""Opt-in compiled-library Meta contracts; this suite never initializes a device.

Run after building on the target CANN host:
    XGEN_NPU_META=1 python x_base/tests/test_npu_fusion_meta.py
"""
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(os.environ.get("XGEN_NPU_META") == "1", "requires the compiled target CANN libraries")
class NPUFusionMeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu  # noqa: F401 — register the device without initializing it.

        # Load only the operator package. Importing x_base also loads vendor
        # attention providers, which can initialize a device at import time.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'x_base/operator'))
        try:
            from npu_fusion import gate_residual, merge_quant, nq2, partial_rope, phaa_layout, phaa_preprocess, swiglu_quant
        finally:
            sys.path.pop(0)
        cls.torch = torch
        cls.gate, cls.merge, cls.norm = gate_residual, merge_quant, nq2
        cls.rope, cls.layout, cls.preprocess, cls.swiglu = partial_rope, phaa_layout, phaa_preprocess, swiglu_quant

    def tensor(self, *shape, dtype=None):
        return self.torch.empty(shape, device="meta", dtype=dtype or self.torch.bfloat16)

    def test_swiglu_and_width_boundary(self):
        q, scale = self.swiglu.swiglu_quantize_reuse(self.tensor(2, 17, 96))
        self.assertEqual(q.shape, (34, 48));self.assertEqual(scale.shape, (34,))
        self.assertEqual(q.dtype, self.torch.int8)
        with self.assertRaises(RuntimeError):
            self.swiglu.swiglu_quantize_reuse(self.tensor(1, 2, 28674))

    def test_gate_and_norm_contracts(self):
        value = self.tensor(2, 17, 64);table = self.tensor(3, 64)
        indices = self.tensor(17, dtype=self.torch.int64)
        self.assertEqual(self.gate.gate_residual(value, value, table, indices).shape, value.shape)
        q, scale = self.norm.norm_affine_quant(value, self.tensor(64), table, table, indices, cached=True)
        self.assertEqual(q.shape, (34, 64));self.assertEqual(scale.shape, (34,))
        with self.assertRaises(RuntimeError):
            self.norm.norm_affine_quant(self.tensor(1, 2, 6145), self.tensor(6145), self.tensor(3, 6145),
                                        self.tensor(3, 6145), self.tensor(2, dtype=self.torch.int64), cached=True)

    def test_rope_pair(self):
        q = self.tensor(2, 17, 3, 128);freq = self.tensor(17, 96, dtype=self.torch.float32)
        result = self.rope.partial_rope_pair(q, q, freq, freq)
        self.assertEqual(tuple(t.shape for t in result), (q.shape, q.shape))

    def test_preprocess_and_merge(self):
        q = self.tensor(1, 2, 4001, 128)
        prepared = self.preprocess.preprocess(q, q, q)
        self.assertEqual(tuple(t.shape for t in prepared), ((1, 2, 4096, 128),) * 3)
        self.assertTrue(all(t.dtype == self.torch.float16 for t in prepared))
        quantized, scale = self.merge.merge_quant(self.tensor(7, 4, 17, 1, 256))
        self.assertEqual(quantized.shape, (17, 7168));self.assertEqual(scale.shape, (17,))

    def test_layout_and_mutation_schema(self):
        q = self.tensor(1, 17, 56, 128)
        sends = [self.tensor(7, 4, 17, 1, 2, 128) for _ in range(3)]
        self.layout.pack_qkv(q, q, q, *sends)
        back = self.tensor(4, 17, 1, 256)
        self.layout.pack_back(self.tensor(1, 2, 67, 128), back, L=67, S=17)
        self.layout.merge_back(self.tensor(7, 4, 17, 1, 256), self.tensor(1, 17, 7168))
        schema = str(self.torch.ops.h3_phaa_layout.pack._schemas[""])
        self.assertIn("Tensor(a!)", schema);self.assertIn("Tensor(b!)", schema);self.assertIn("Tensor(c!)", schema)

    def test_device_remains_uninitialized(self):
        self.assertFalse(self.torch.npu.is_initialized())

    @classmethod
    def tearDownClass(cls):
        if cls.torch.npu.is_initialized():
            raise AssertionError("Meta contract checks initialized an NPU device")


if __name__ == "__main__":
    unittest.main(verbosity=2)

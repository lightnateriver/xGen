"""CPU-only flag contract tests; no accelerator dependencies required."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'x_diffusers/x_diffusers/adaptor/h3_parallel_config.py'
spec = importlib.util.spec_from_file_location('h3_parallel_config_contract', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def settings(**kwargs):
    base = dict(model='MiniMax-H3', task_type='ref2va', h3_parallel_mode='tp_cp', h3_dit_tp=2,
                sp=4, h3_tp_comm='bf16', h3_cp_comm='auto', h3_cuda_precision='fp8-sage')
    return SimpleNamespace(**(base | kwargs))


class TestH3ParallelConfig(unittest.TestCase):
    def test_shared_eight_rank_layout_is_not_a_cartesian_product(self):
        for wire in ('bf16','fp8'):
            args=settings(h3_parallel_mode='tp_cp_shared',h3_dit_tp=8,sp=8,h3_tp_comm=wire)
            config=module.H3ParallelConfig.from_args(args,'cuda',8)
            self.assertEqual((config.tp,config.cp,config.tp_comm,config.cp_comm),(8,8,wire,wire))
            for world in (1,4,16,64):
                with self.assertRaises(ValueError):module.H3ParallelConfig.from_args(args,'cuda',world)
            if wire == 'bf16':
                self.assertEqual(module.H3ParallelConfig.from_args(args,'npu',8).cp_comm,'bf16')
            else:
                with self.assertRaises(ValueError):module.H3ParallelConfig.from_args(args,'npu',8)
        for tp,cp in ((1,8),(2,4),(4,2),(8,1)):
            with self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_parallel_mode='tp_cp_shared',h3_dit_tp=tp,sp=cp),'cuda',8)
        with self.assertRaisesRegex(ValueError,'matching'):
            module.H3ParallelConfig.from_args(settings(h3_parallel_mode='tp_cp_shared',h3_dit_tp=8,sp=8,
                                                      h3_tp_comm='fp8',h3_cp_comm='bf16'),'cuda',8)
        with patch.dict(os.environ,{'XGEN_PHAA_DEVICE_JOIN':'1'}):
            with self.assertRaisesRegex(ValueError,'standalone PHAA'):
                module.H3ParallelConfig.from_args(settings(h3_parallel_mode='tp_cp_shared',h3_dit_tp=8,sp=8),'cuda',8)

    def test_old_callers_unchanged(self):
        for backend in ('cuda','npu'):
            config = module.H3ParallelConfig.from_args(SimpleNamespace(),backend,8)
            self.assertEqual(config.mode,'sp')

    def test_default_parser(self):
        args = module.add_arguments(argparse.ArgumentParser()).parse_args([])
        self.assertEqual(vars(args), dict(h3_parallel_mode='sp',h3_dit_tp=1,h3_tp_comm='bf16',h3_cp_comm='auto',h3_tp_comm_chunk_elements=262144,h3_parallel_warmup_mib=0))

    def test_warmup_is_opt_in_bounded_and_immutable(self):
        self.assertEqual(module.H3ParallelConfig.from_args(settings(),'cuda',8).warmup_mib,0)
        self.assertEqual(module.H3ParallelConfig.from_args(settings(h3_parallel_warmup_mib=512),'cuda',8).warmup_mib,512)
        for value in (-1,True,1.5,1025):
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_parallel_warmup_mib=value),'cuda',8)
        for backend,change in [('cuda',dict(h3_parallel_mode='sp',h3_dit_tp=1)),('npu',dict(h3_tp_comm='fp8')),('cuda',dict(model='Wan'))]:
            with self.subTest(backend=backend,change=change),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_parallel_warmup_mib=512,**change),backend,8)
        with self.assertRaisesRegex(ValueError,'Restart the manager'):
            module.validate_resident_args(settings(),settings(h3_parallel_warmup_mib=512))

    def test_transport_combinations(self):
        for tp in ('bf16','fp8'):
            for cp in ('bf16','fp8'):
                config = module.H3ParallelConfig.from_args(settings(h3_tp_comm=tp,h3_cp_comm=cp),'cuda',8)
                self.assertEqual((config.tp_comm,config.cp_comm),(tp,cp))

    def test_env_auto_and_explicit_override(self):
        with patch.dict(os.environ,{'XGEN_PHAA_FP8':'1'}):
            self.assertEqual(module.H3ParallelConfig.from_args(settings(),'cuda',8).cp_comm,'fp8')
            self.assertEqual(module.H3ParallelConfig.from_args(settings(h3_cp_comm='bf16'),'cuda',8).cp_comm,'bf16')

    def test_reject_ignored_parameters(self):
        for change in (dict(h3_dit_tp=2),dict(h3_tp_comm='fp8'),dict(h3_cp_comm='bf16')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_parallel_mode='sp',**change),'cuda',8)

    def test_reject_incompatible_routes_before_load(self):
        for change in (dict(sp=8),dict(sp=3),dict(h3_dit_tp=8),dict(ring_degree=2),
                       dict(ulysses_degree=8),dict(h3_refiner=True),dict(model='Wan'),
                       dict(h3_cuda_precision='auto'),dict(task_type='other')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(**change),'cuda',8)
        self.assertEqual(module.H3ParallelConfig.from_args(settings(),'npu',8).cp_comm,'bf16')

    def test_common_tasks(self):
        for task in ('t2va','fl2va','ref2va'):
            self.assertEqual(module.H3ParallelConfig.from_args(settings(task_type=task),'cuda',8).tp,2)

    def test_tp4_cp2_and_reject_additive_card_count(self):
        for wire in ('bf16','fp8'):
            config=module.H3ParallelConfig.from_args(settings(h3_dit_tp=4,sp=2,h3_tp_comm=wire),'cuda',8)
            self.assertEqual((config.tp,config.cp),(4,2))
        with self.assertRaises(ValueError):
            module.H3ParallelConfig.from_args(settings(h3_dit_tp=4,sp=4),'cuda',8)

    def test_packet_size_is_explicit_bounded_and_transport_specific(self):
        args=settings(h3_dit_tp=4,sp=2,h3_tp_comm='fp8',h3_tp_comm_chunk_elements=2097152)
        self.assertEqual(module.H3ParallelConfig.from_args(args,'cuda',8).tp_chunk_elements,2097152)
        for value in (0,129,True,4194432):
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_tp_comm_chunk_elements=value),'cuda',8)
        for change in (dict(h3_parallel_mode='sp',h3_dit_tp=1),dict(h3_tp_comm='bf16')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                module.H3ParallelConfig.from_args(settings(h3_tp_comm_chunk_elements=2097152,**change),'cuda',8)


if __name__ == '__main__':
    unittest.main()

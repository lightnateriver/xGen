"""CPU contracts for migrating optional video fusions onto the v0.6 baseline.

These do not validate compiled Ascend kernels or generated-video quality.
Run directly with unittest in a CPU Torch + Diffusers environment.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ROOT / 'x_base/x_base/operator'
ADAPTORS = ROOT / 'x_diffusers/x_diffusers/adaptor'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


options = load('_h3_npu_options_cpu', ADAPTORS / 'h3_npu_options.py')
vc = load('_h3_vc_cpu', OPERATORS / 'vc_attention.py')
runtime = load('_h3_runtime_cpu', OPERATORS / 'npu_fusion/runtime.py')
signature = load('_h3_signature_cpu', OPERATORS / 'npu_fusion/tensor_signature.py').signature
caps = load('_h3_caps_cpu', OPERATORS / 'npu_fusion/capabilities.py')


def args(**values):
    return SimpleNamespace(model='MiniMax-H3', atten='laser', h3_opts=None, **values)


class OptionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_unset_fusions_are_off(self):
        value = args()
        self.assertEqual(options.resolve_h3_features(value), ())

    def test_cli_none_overrides_environment(self):
        os.environ['H3_OPT_FEATURES'] = 'all'
        value = args(); value.h3_opts = 'none'
        self.assertEqual(options.resolve_h3_features(value), ())

    def test_feature_order_and_individual_overrides(self):
        value = args(); value.h3_opts = 'all,-rope'
        value.h3_opt_rope = True
        value.h3_opt_swiglu = False
        self.assertEqual(options.resolve_h3_features(value), tuple(n for n in options.H3_FUSION_FEATURES if n != 'swiglu'))

    def test_unknown_feature_rejected(self):
        value = args(); value.h3_opts = 'silently_misspelled'
        with self.assertRaises(ValueError): options.resolve_h3_features(value)

    def test_boolean_cli_spellings(self):
        parser = argparse.ArgumentParser()
        options.add_arguments(parser)
        value = parser.parse_args(['--h3-opts','all','--no-h3-opt-rope','--no-h3_opt_swiglu'])
        selected = options.resolve_h3_features(value)
        self.assertNotIn('rope', selected); self.assertNotIn('swiglu', selected)

    def test_cpu_and_image_reject_explicit_video_fusions(self):
        value = args(); value.h3_opts = 'rope'
        with self.assertRaises(ValueError): options.validate(value, 'cuda')
        value.model = 'MiniMax-H3-Image'
        with self.assertRaises(ValueError): options.validate(value, 'npu')

    def test_image_defaults_do_not_touch_video_transformer(self):
        value = args(); value.model = 'MiniMax-H3-Image'
        options.configure_model(SimpleNamespace(), value, 'npu')
        self.assertEqual(value.h3_fused_ops, ())

    def test_default_configuration_does_not_install_runtime(self):
        configure = Mock()
        pipe = SimpleNamespace(transformer=SimpleNamespace(configure_fused_ops=configure))
        options.configure_model(pipe, args(), 'npu')
        configure.assert_not_called()

    def test_disabling_existing_fusions_is_explicit(self):
        configure = Mock()
        pipe = SimpleNamespace(transformer=SimpleNamespace(_h3_fused_ops=('rope',), configure_fused_ops=configure))
        options.configure_model(pipe, args(), 'npu')
        configure.assert_called_once_with(())

    def test_vc_and_fusions_are_independent(self):
        value = args(); value.atten = 'vc'
        self.assertEqual(options.validate(value, 'npu'), ())
        with self.assertRaises(ValueError): options.validate(value, 'cuda')

    def test_request_context_without_runtime_is_noop(self):
        with options.request_context(SimpleNamespace()): pass


class VCTests(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 2, 11, 128, dtype=torch.bfloat16)
        self.fallback = Mock(return_value=torch.zeros_like(self.q))
        self.provider = Mock(side_effect=lambda q,k,v,**kw:q + v)
        loader = Mock(); loader.info = {'expcast': True}
        self.module = SimpleNamespace(load_ascend_expcast=Mock(return_value=loader), ascend_attention_expcast=self.provider)

    def initialized(self):
        backend = vc.VCAttention()
        with patch.object(vc.importlib, 'import_module', return_value=self.module):
            backend.initialize(); backend.initialize()
        self.module.load_ascend_expcast.assert_called_once()
        return backend

    def test_supported_metadata_dispatches_and_initializes_once(self):
        backend = self.initialized()
        with patch.object(vc, '_is_npu', return_value=True):
            actual = backend(self.q,self.q,self.q,fallback=self.fallback)
        self.assertTrue(torch.equal(actual, self.q + self.q))
        self.provider.assert_called_once(); self.fallback.assert_not_called()

    def test_native_execution_failure_is_not_hidden(self):
        backend = self.initialized()
        self.provider.side_effect = RuntimeError('native failure')
        with patch.object(vc, '_is_npu', return_value=True), self.assertRaisesRegex(RuntimeError,'native failure'):
            backend(self.q,self.q,self.q,fallback=self.fallback)
        self.fallback.assert_not_called()

    def test_cpu_mask_causal_dropout_and_gradient_fall_back(self):
        backend = self.initialized()
        for change in [{}, {'attn_mask':torch.ones(11,11)}, {'is_causal':True}, {'dropout_p':0.1}]:
            backend(self.q,self.q,self.q,fallback=self.fallback,**change)
        with patch.object(vc, '_is_npu', return_value=True):
            q = self.q.clone().requires_grad_()
            backend(q,q,q,fallback=self.fallback)
        self.assertEqual(self.fallback.call_count,5)
        self.provider.assert_not_called()

    def test_unsupported_dtype_width_and_scale_fall_back(self):
        backend = self.initialized()
        with patch.object(vc, '_is_npu', return_value=True):
            for q in [self.q.float(),self.q[..., :32]]:
                backend(q,q,q,fallback=self.fallback)
            for scale in [float('nan'),float('inf'),0,-1,70000,True,1e-30]:
                backend(self.q,self.q,self.q,fallback=self.fallback,scale=scale)
        self.assertEqual(self.fallback.call_count,9)
        self.provider.assert_not_called()

    def test_missing_provider_falls_back_only_at_initialization(self):
        backend = vc.VCAttention()
        with patch.object(vc.importlib,'import_module',side_effect=ImportError('not installed')) as importer:
            backend.initialize(); backend.initialize()
        importer.assert_called_once()
        backend(self.q,self.q,self.q,fallback=self.fallback)
        self.fallback.assert_called_once()


class AttentionSelectionTests(unittest.TestCase):
    def test_default_sdpa_and_legacy_laser_selection_are_preserved(self):
        tree = ast.parse((OPERATORS/'attention.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AttentionManager')
        namespace = {'torch':torch, 'logger':logging.getLogger('test.attention'), 'LA_AVAILABLE':True,
                     'device_type':lambda:'npu', 'base_attention_infer':torch.nn.functional.scaled_dot_product_attention}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'attention-manager-cpu','exec'),namespace)
        manager = namespace['AttentionManager']()
        manager.configure_npu_attention('laser')
        self.assertTrue(manager.is_laser_attention)
        self.assertFalse(manager.is_vc_attention)
        manager.configure_npu_attention('none')
        self.assertFalse(manager.is_laser_attention)
        self.assertFalse(manager.is_vc_attention)
        self.assertIsNone(manager._vc_backend)
        q = torch.randn(1,2,7,16)
        with torch.no_grad():
            expected = torch.nn.functional.scaled_dot_product_attention(q,q,q)
            actual = manager.attention(q,q,q)
        self.assertTrue(torch.equal(actual,expected))


tree = ast.parse((ROOT/'x_diffusers/x_diffusers/framework/transformer/h3_npu_fusion.py').read_text(encoding='utf-8'))
scope = dict(torch=torch, F=torch.nn.functional, threading=threading, weakref=weakref, contextmanager=contextmanager,
             logger=logging.getLogger('test.h3.fusions'), Diagnostics=runtime.Diagnostics, Kernels=runtime.Kernels,
             no_hooks=runtime.no_hooks, tensors_ok=runtime.tensors_ok, signature=signature, caps=caps,
             SwiGLU=SwiGLU, FeedForward=FeedForward, MiniMaxH3TransformerBlock=MiniMaxH3TransformerBlock,
             _original_h3_block_forward=MiniMaxH3TransformerBlock.forward)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),
             'h3-fusion-functions','exec'),scope)


class CountLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(16,16,bias=False)
        self.calls = 0
    def forward(self,x):
        self.calls += 1
        return super().forward(x)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_embedder = CountLinear()
        self.token_refiner = CountLinear()
        self.transformer_blocks = torch.nn.ModuleList()
        self.eval()


class FusionRuntimeTests(unittest.TestCase):
    def model(self, features=('text_cache',)):
        model = TinyModel()
        scope['configure_h3_fused_ops'](model,features)
        return model

    def test_default_block_forward_is_bitwise_unchanged(self):
        torch.manual_seed(42)
        block = MiniMaxH3TransformerBlock(32,2,16,64,16,1e-6,1e-6).eval()
        x = torch.randn(1,7,32); temb = torch.randn(3,16); indices = torch.arange(7)%3
        with torch.no_grad():
            expected = block(x,temb,indices,None)
            actual = scope['h3_block_forward'](block,x,temb,indices,None)
        self.assertTrue(torch.equal(expected,actual))

    def test_cpu_explicit_features_fall_back_to_original_block(self):
        torch.manual_seed(42)
        block = MiniMaxH3TransformerBlock(32,2,16,64,16,1e-6,1e-6).eval()
        model = self.model(('norm_quant','gate_residual','swiglu'))
        block._h3_transformer_ref = weakref.ref(model)
        x = torch.randn(1,7,32); temb = torch.randn(3,16); indices = torch.arange(7)%3
        with torch.no_grad():expected = block(x,temb,indices,None)
        with scope['h3_inference_context'](model):actual = scope['h3_block_forward'](block,x,temb,indices,None)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(sum(model._h3_fusion_diagnostics.hits.values()),0)

    def test_request_cache_reuses_only_inside_one_request(self):
        model = self.model(); value = torch.randn(3,16)
        for count in [1,2]:
            with scope['h3_inference_context'](model):
                a = scope['_h3_cached_text_call'](model,'context_embedder',value)
                b = scope['_h3_cached_text_call'](model,'context_embedder',value)
                self.assertIs(a,b)
                self.assertEqual(model.context_embedder.calls,count)
            self.assertEqual(model._h3_text_cache,{})

    def test_mutation_invalidates_request_cache(self):
        model = self.model(); value = torch.randn(3,16)
        with scope['h3_inference_context'](model):
            scope['_h3_cached_text_call'](model,'context_embedder',value)
            value.add_(1)
            scope['_h3_cached_text_call'](model,'context_embedder',value)
            model.context_embedder.weight.add_(1)
            scope['_h3_cached_text_call'](model,'context_embedder',value)
        self.assertEqual(model.context_embedder.calls,3)

    def test_failure_clears_request_state_and_releases_lock(self):
        model = self.model(); value = torch.randn(3,16)
        with self.assertRaisesRegex(RuntimeError,'request failure'):
            with scope['h3_inference_context'](model):
                scope['_h3_cached_text_call'](model,'context_embedder',value)
                raise RuntimeError('request failure')
        self.assertFalse(model._h3_request_active)
        self.assertFalse(model._h3_fusion_diagnostics.in_request)
        self.assertEqual(model._h3_text_cache,{})
        with scope['h3_inference_context'](model):pass

    def test_active_request_rejects_reconfiguration(self):
        model = self.model()
        with scope['h3_inference_context'](model):
            with self.assertRaises(RuntimeError):scope['configure_h3_fused_ops'](model,())
        scope['configure_h3_fused_ops'](model,())
        self.assertFalse(model._h3_fused_ops)


if __name__=='__main__':
    unittest.main(verbosity=2)

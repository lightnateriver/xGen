"""Execute real workflow hooks with fake ComfyUI/cache/device objects."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1] / "x_comfyui/adaptor"


class PromptSettingsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in ("X_QUANT", "X_TEXT_TP", "X_ADALN_CACHE_FILE", "X_ADALN_TRANSFORMER_PATH",
                     "X_ATTENTION_MODE", "X_PHAA_SPLIT", "X_ROPE_FUSED"):
            os.environ.pop(name, None)
        os.environ["X_SP_SIZE"] = "8"
        self.modules = patch.dict(sys.modules, {})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        package = ModuleType("_prompt_test_adaptor")
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        self.env = self.load(package.__name__ + ".env", "env.py")
        self.hooks = self.load(package.__name__ + ".model_hooks", "model_hooks.py")
        self.records = []
        owner = self

        class BaseModel:
            def __init__(self):
                self.diffusion_model = object()

            def load_model_weights(self, *args):
                return "loaded"

        class Executor:
            def __init__(self):
                self.caches = SimpleNamespace(outputs=SimpleNamespace(
                    initialized=True, cache={}, subcaches={},
                    cache_key_set=SimpleNamespace(get_data_key=lambda node: node,
                                                  get_subcache_key=lambda node: node),
                ))
                self.fail = False

            async def execute_async(self, *args):
                owner.records.append(dict(quant=owner.env.get_quant(), text_tp=owner.env.get_text_tp(),
                                          cache=owner.env.get_adaln_cache_file(),
                                          path=owner.env.get_adaln_transformer_path(),
                                          rope=owner.rope.is_rope_fused,
                                          phaa=owner.phaa.PHAA_SPLIT_NUM))
                if self.fail:
                    raise RuntimeError("execution failed")
                return "executed"

        self.Executor = Executor
        self.model = SimpleNamespace(BaseModel=BaseModel, patch_model_classes=Mock(),
                                     _apply_sp_and_quant=Mock(), _apply_quant=Mock(), _apply_fsdp=Mock())
        self.attention = Mock()
        self.phaa = ModuleType("x_base.phaa.globals")
        self.phaa.ENABLE_PHAA = False
        self.phaa.PHAA_SPLIT_NUM = None
        self.rope = SimpleNamespace(is_rope_fused=False)
        self.rope.enable_rope_fused = lambda: setattr(self.rope, "is_rope_fused", True)
        base = ModuleType("x_base")
        base.attention_manager = self.attention
        base.phaa = ModuleType("x_base.phaa")
        base.phaa.globals = self.phaa
        sys.modules.update({
            "execution": SimpleNamespace(PromptExecutor=Executor),
            "safetensors.torch": SimpleNamespace(load_file=Mock()),
            "x_comfyui.framework.transformer": SimpleNamespace(minimax_h3=self.model),
            "x_base": base, "x_base.phaa": base.phaa, "x_base.phaa.globals": self.phaa,
            "x_base.operator.rope": SimpleNamespace(rope_manager=self.rope),
        })
        self.hooks.install_model_hooks()
        self.executor = Executor()

    def load(self, name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def prompt(self, **inputs):
        return {
            "control": {"class_type": "NPUTurbo", "inputs": inputs},
            "clip": {"class_type": "CLIPLoader", "inputs": {}},
            "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0]}},
            "model": {"class_type": "UNETLoader", "inputs": {}},
        }

    def execute(self, prompt):
        coroutine = self.executor.execute_async(prompt, "id")
        try:
            coroutine.send(None)
        except StopIteration as complete:
            return complete.value
        self.fail("Fake executor unexpectedly suspended")

    def cache(self):
        output = self.executor.caches.outputs
        output.cache = {key: object() for key in ("clip", "text", "model")}
        output.subcaches = dict(output.cache)
        return output

    def test_removing_node_resets_quant_and_tp_and_invalidates_models(self):
        self.execute(self.prompt(quant="enable", text_tp="enable"))
        cache = self.cache()
        self.execute({"clip": {"class_type": "CLIPLoader", "inputs": {}}})
        self.assertEqual([(r["quant"], r["text_tp"]) for r in self.records], [("int8", 8), ("", 0)])
        self.assertEqual(cache.cache, {})
        self.assertEqual(cache.subcaches, {})

    def test_node_removal_restores_enabled_environment_defaults(self):
        os.environ.update(X_QUANT="int8", X_TEXT_TP="4")
        self.execute(self.prompt(quant="disable", text_tp="disable"))
        self.execute({})
        self.assertEqual((self.records[-1]["quant"], self.records[-1]["text_tp"]), ("int8", 4))

    def test_tp_change_selectively_clears_clip_and_dependents(self):
        self.execute(self.prompt(text_tp="enable"))
        cache = self.cache()
        self.execute(self.prompt(text_tp="disable"))
        self.assertEqual(set(cache.cache), {"model"})
        self.assertEqual(set(cache.subcaches), {"model"})

    def test_node_removal_with_same_effective_values_keeps_cache(self):
        os.environ.update(X_QUANT="int8", X_TEXT_TP="4")
        self.execute(self.prompt(quant="enable", text_tp="enable"))
        cache = self.cache()
        before = dict(cache.cache)
        self.execute({})
        self.assertEqual(cache.cache, before)

    def test_unchanged_workflow_keeps_cached_models(self):
        prompt = self.prompt(quant="enable", text_tp="enable")
        self.execute(prompt)
        cache = self.cache()
        before = dict(cache.cache)
        self.execute(prompt)
        self.assertEqual(cache.cache, before)

    def test_environment_tp_size_change_invalidates_clip_even_without_toggle_change(self):
        os.environ["X_TEXT_TP"] = "2"
        self.execute(self.prompt(text_tp="enable"))
        cache = self.cache()
        os.environ["X_TEXT_TP"] = "4"
        self.execute(self.prompt(text_tp="enable"))
        self.assertEqual(self.records[-1]["text_tp"], 4)
        self.assertEqual(set(cache.cache), {"model"})

    def test_adaln_path_change_and_node_removal_invalidate_models(self):
        os.environ.update(X_ADALN_CACHE_FILE="env-cache", X_ADALN_TRANSFORMER_PATH="env-weights")
        self.execute(self.prompt(adaln="enable", adaln_cache_file="node-cache", adaln_transformer_path="node-weights"))
        cache = self.cache()
        self.execute(self.prompt(adaln="enable", adaln_cache_file="other-cache", adaln_transformer_path="other-weights"))
        self.assertEqual(cache.cache, {})
        self.cache()
        self.execute({})
        self.assertEqual((self.records[-1]["cache"], self.records[-1]["path"]), ("env-cache", "env-weights"))
        self.assertEqual(cache.cache, {})

    def test_adaln_disable_and_removal_restore_environment(self):
        os.environ.update(X_ADALN_CACHE_FILE="env-cache", X_ADALN_TRANSFORMER_PATH="env-weights")
        self.execute(self.prompt(adaln="disable", adaln_cache_file="ignored"))
        self.assertIsNone(self.records[-1]["cache"])
        self.execute({})
        self.assertEqual((self.records[-1]["cache"], self.records[-1]["path"]), ("env-cache", "env-weights"))

    def test_runtime_switches_return_to_environment_when_node_is_removed(self):
        self.execute(self.prompt(attention_mode="sageattn", phaa_split="4", rope_fused="enable"))
        self.assertEqual((self.records[-1]["rope"], self.records[-1]["phaa"]), (True, 4))
        self.attention.enable_sage_attention.assert_called_once()
        self.execute({})
        self.assertEqual((self.records[-1]["rope"], self.records[-1]["phaa"]), (False, None))
        self.assertEqual(self.attention.disable_sage_attention.call_count, 2)

    def test_environment_runtime_switches_are_restored(self):
        os.environ.update(X_ATTENTION_MODE="laserattn", X_PHAA_SPLIT="2", X_ROPE_FUSED="true")
        self.execute(self.prompt(attention_mode="sdpa", rope_fused="disable"))
        self.execute({})
        self.attention.enable_laser_attention.assert_called_once()
        self.assertEqual((self.records[-1]["rope"], self.records[-1]["phaa"]), (True, 2))

    def test_overrides_are_ready_before_model_loaders_execute(self):
        self.execute(self.prompt(quant="enable", text_tp="enable", adaln="enable", adaln_cache_file="cache"))
        base = self.model.BaseModel()
        base.load_model_weights({})
        self.model._apply_quant.assert_called_once_with(base, quant="int8")
        self.assertEqual(self.model._apply_sp_and_quant.call_args.kwargs["cache_file"], "cache")

    def test_failed_execution_does_not_leak_settings_to_next_workflow(self):
        self.executor.fail = True
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.execute(self.prompt(quant="enable", text_tp="enable"))
        self.executor.fail = False
        cache = self.cache()
        self.execute({})
        self.assertEqual((self.records[-1]["quant"], self.records[-1]["text_tp"]), ("", 0))
        self.assertEqual(cache.cache, {})

    def test_multiple_control_nodes_are_rejected_before_execution(self):
        prompt = self.prompt()
        prompt["another"] = {"class_type": "NPUTurbo", "inputs": {}}
        with self.assertRaisesRegex(ValueError, "one NPUTurbo"):
            self.execute(prompt)
        self.assertFalse(self.records)

    def test_hook_installation_is_idempotent(self):
        self.hooks.install_model_hooks()
        self.assertEqual(self.execute({}), "executed")
        self.assertEqual(len(self.records), 1)

    def test_node_and_prompt_hook_share_effective_settings(self):
        nodes = self.load("_prompt_test_adaptor.nodes", "nodes.py")
        inputs = dict(attention_mode="sdpa", phaa_split="1", quant="enable", adaln="enable",
                      adaln_cache_file="cache", adaln_transformer_path="weights", text_tp="enable", rope_fused="enable")
        expected = self.env.resolve_prompt_options(self.prompt(**inputs))
        self.assertEqual(nodes.NPUTurbo().apply(**inputs), ())
        actual = (self.env.get_quant(), self.env.get_text_tp(), self.env.get_adaln_cache_file(), self.env.get_rope_fused())
        self.assertEqual(actual, (expected.quant, expected.text_tp, expected.adaln_cache_file, expected.rope_fused))


if __name__ == "__main__":
    unittest.main()

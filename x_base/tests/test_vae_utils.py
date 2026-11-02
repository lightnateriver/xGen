"""
Unit tests for vae_parallelism/utils.py.

Covers:
- enable_lightning: sets lightning and return_output flags
- enable_vae_lightning: delegates to vae and video_processor
- add_lightning_init: decorator adds lightning/return_output defaults to __init__
"""
from unittest.mock import MagicMock

from x_base.vae_parallelism.utils import (
    enable_lightning,
    enable_vae_lightning,
    add_lightning_init,
)


class TestEnableLightning:
    def test_sets_lightning_flag(self):
        obj = MagicMock()
        enable_lightning(obj)
        assert obj.lightning is True
        assert obj.return_output is False

    def test_return_output_true(self):
        obj = MagicMock()
        enable_lightning(obj, return_output=True)
        assert obj.return_output is True


class TestEnableVaeLightning:
    def test_delegates_to_vae_and_processor(self):
        pipe = MagicMock()
        enable_vae_lightning(pipe, return_output=True)
        pipe.vae.enable_lightning.assert_called_once_with(True)
        pipe.video_processor.enable_lightning.assert_called_once_with(True)

    def test_default_return_output_false(self):
        pipe = MagicMock()
        enable_vae_lightning(pipe)
        pipe.vae.enable_lightning.assert_called_once_with(False)


class TestAddLightningInit:
    def test_adds_lightning_attributes(self):
        @add_lightning_init
        class Foo:
            def __init__(self):
                self.existing = True

        obj = Foo()
        assert obj.lightning is False
        assert obj.return_output is True
        assert obj.existing is True

    def test_preserves_init_args(self):
        @add_lightning_init
        class Bar:
            def __init__(self, x, y=10):
                self.x = x
                self.y = y

        obj = Bar(5, y=20)
        assert obj.x == 5
        assert obj.y == 20
        assert obj.lightning is False

    def test_returns_class(self):
        @add_lightning_init
        class Baz:
            pass
        assert isinstance(Baz(), Baz)

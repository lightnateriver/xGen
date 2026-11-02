"""
Unit tests for sequence_parallelism/padding.py.

Covers:
- PadManager: set, set_from_group, get, get_or_default, is_set, clear
- Global singleton: get_pad_manager
- Deprecated wrappers: set_pad, get_pad (with DeprecationWarning)
- PadNotSetError (padding.py 的直接依赖异常)
"""
import warnings
from unittest.mock import patch, MagicMock

import pytest

from x_base.sequence_parallelism.padding import (
    PadManager,
    get_pad_manager,
    set_pad,
    get_pad,
    PAD_DICT,
)
from x_base.sequence_parallelism.errors import PadNotSetError


# ============================================================
# PadManager
# ============================================================

class TestPadManagerSet:
    def test_exact_divisible(self):
        mgr = PadManager()
        pad = mgr.set("seq_len", 128, 4)
        assert pad == 0

    def test_needs_padding(self):
        mgr = PadManager()
        pad = mgr.set("seq_len", 130, 4)
        assert pad == 2

    def test_sp_size_equals_dim(self):
        mgr = PadManager()
        pad = mgr.set("seq_len", 4, 4)
        assert pad == 0

    def test_dim_smaller_than_sp_size(self):
        mgr = PadManager()
        pad = mgr.set("seq_len", 3, 8)
        assert pad == 5

    def test_dim_one(self):
        mgr = PadManager()
        pad = mgr.set("seq_len", 1, 4)
        assert pad == 3

    def test_overwrite_existing(self):
        mgr = PadManager()
        mgr.set("seq_len", 130, 4)
        pad = mgr.set("seq_len", 128, 4)
        assert pad == 0

    def test_multiple_keys(self):
        mgr = PadManager()
        mgr.set("seq_len", 130, 4)
        mgr.set("hidden_dim", 64, 8)
        assert mgr.get("seq_len") == 2
        assert mgr.get("hidden_dim") == 0


class TestPadManagerSetFromGroup:
    def test_delegates_to_set(self):
        mgr = PadManager()
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.padding.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            pad = mgr.set_from_group("seq_len", 130, mock_group)
        assert pad == 2
        mock_dist.get_world_size.assert_called_once_with(mock_group)


class TestPadManagerGet:
    def test_existing_key(self):
        mgr = PadManager()
        mgr.set("seq_len", 130, 4)
        assert mgr.get("seq_len") == 2

    def test_missing_key_raises_pad_not_set_error(self):
        mgr = PadManager()
        with pytest.raises(PadNotSetError) as exc_info:
            mgr.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert exc_info.value.name == "nonexistent"


class TestPadManagerGetOrDefault:
    def test_existing_key(self):
        mgr = PadManager()
        mgr.set("seq_len", 130, 4)
        assert mgr.get_or_default("seq_len") == 2

    def test_missing_key_returns_zero(self):
        mgr = PadManager()
        assert mgr.get_or_default("nonexistent") == 0

    def test_missing_key_custom_default(self):
        mgr = PadManager()
        assert mgr.get_or_default("nonexistent", 99) == 99


class TestPadManagerIsSet:
    def test_not_set(self):
        mgr = PadManager()
        assert mgr.is_set("seq_len") is False

    def test_is_set(self):
        mgr = PadManager()
        mgr.set("seq_len", 130, 4)
        assert mgr.is_set("seq_len") is True


class TestPadManagerClear:
    def test_clear_empties_all(self):
        mgr = PadManager()
        mgr.set("a", 130, 4)
        mgr.set("b", 64, 8)
        mgr.clear()
        assert mgr.is_set("a") is False
        assert mgr.is_set("b") is False


# ============================================================
# Global singleton
# ============================================================

class TestGetPadManager:
    def test_returns_instance(self):
        mgr = get_pad_manager()
        assert isinstance(mgr, PadManager)

    def test_singleton_identity(self):
        mgr1 = get_pad_manager()
        mgr2 = get_pad_manager()
        assert mgr1 is mgr2


# ============================================================
# Deprecated wrappers: set_pad / get_pad
# ============================================================

class TestSetPadDeprecated:
    def test_emits_deprecation_warning(self):
        with patch("x_base.sequence_parallelism.padding.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                set_pad("test_key", 130, MagicMock())
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()

    def test_computes_correct_pad(self):
        with patch("x_base.sequence_parallelism.padding.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pad = set_pad("test_deprecated", 130, MagicMock())
        assert pad == 2


class TestGetPadDeprecated:
    def test_emits_deprecation_warning(self):
        with patch("x_base.sequence_parallelism.padding.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                set_pad("test_get_deprecated", 130, MagicMock())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_pad("test_get_deprecated")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_returns_value_from_manager(self):
        with patch("x_base.sequence_parallelism.padding.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                set_pad("test_get_val", 130, MagicMock())
                val = get_pad("test_get_val")
        assert val == 2

    def test_missing_key_raises_pad_not_set_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(PadNotSetError):
                get_pad("absolutely_nonexistent_key_xyz")

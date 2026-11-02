"""
Unit tests for phaa/globals.py.

Covers:
- Default values: PHAA_SPLIT_NUM=None, ENABLE_PHAA=False
- set_phaa_split_num / get_phaa_split_num
- enable_phaa / is_phaa_enabled
"""
import x_base.phaa.globals as g


class TestDefaultValues:
    def test_split_num_default_none(self):
        assert g.PHAA_SPLIT_NUM is None

    def test_enable_phaa_default_false(self):
        assert g.ENABLE_PHAA is False


class TestSetGetSplitNum:
    def test_set_and_get(self):
        original = g.PHAA_SPLIT_NUM
        try:
            g.set_phaa_split_num(4)
            assert g.get_phaa_split_num() == 4
        finally:
            g.PHAA_SPLIT_NUM = original

    def test_overwrite(self):
        original = g.PHAA_SPLIT_NUM
        try:
            g.set_phaa_split_num(2)
            g.set_phaa_split_num(8)
            assert g.get_phaa_split_num() == 8
        finally:
            g.PHAA_SPLIT_NUM = original


class TestEnablePhaa:
    def test_enable_changes_flag(self):
        original = g.ENABLE_PHAA
        try:
            g.enable_phaa()
            assert g.is_phaa_enabled() is True
        finally:
            g.ENABLE_PHAA = original

    def test_default_is_not_enabled(self):
        original = g.ENABLE_PHAA
        try:
            g.ENABLE_PHAA = False
            assert g.is_phaa_enabled() is False
        finally:
            g.ENABLE_PHAA = original

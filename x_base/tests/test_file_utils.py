"""
Unit tests for utils/file.py.

Covers:
- read_text: reads and strips file content
- list_cases: filters dirs with image.jpg + prompt.txt, sorts numeric before alphabetic
"""
import os
import tempfile

import pytest

from x_base.utils.file import read_text, list_cases


class TestReadText:
    def test_reads_and_strips(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("  hello world  \n", encoding="utf-8")
        assert read_text(str(f)) == "hello world"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert read_text(str(f)) == ""

    def test_multiline_strips_outer(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        assert read_text(str(f)) == "line1\nline2"


class TestListCases:
    def test_filters_dirs_with_image_and_prompt(self, tmp_path):
        (tmp_path / "case1").mkdir()
        (tmp_path / "case1" / "image.jpg").write_text("", encoding="utf-8")
        (tmp_path / "case1" / "prompt.txt").write_text("", encoding="utf-8")
        (tmp_path / "case2").mkdir()
        (tmp_path / "case2" / "image.jpg").write_text("", encoding="utf-8")
        result = list_cases(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("case1")

    def test_ignores_dirs_missing_prompt(self, tmp_path):
        (tmp_path / "no_prompt").mkdir()
        (tmp_path / "no_prompt" / "image.jpg").write_text("", encoding="utf-8")
        assert list_cases(str(tmp_path)) == []

    def test_ignores_dirs_missing_image(self, tmp_path):
        (tmp_path / "no_image").mkdir()
        (tmp_path / "no_image" / "prompt.txt").write_text("", encoding="utf-8")
        assert list_cases(str(tmp_path)) == []

    def test_ignores_files_in_root(self, tmp_path):
        (tmp_path / "image.jpg").write_text("", encoding="utf-8")
        (tmp_path / "prompt.txt").write_text("", encoding="utf-8")
        assert list_cases(str(tmp_path)) == []

    def test_numeric_sort_before_alpha(self, tmp_path):
        for name in ["b_case", "1", "2", "a_case"]:
            d = tmp_path / name
            d.mkdir()
            (d / "image.jpg").write_text("", encoding="utf-8")
            (d / "prompt.txt").write_text("", encoding="utf-8")
        result = list_cases(str(tmp_path))
        names = [os.path.basename(r) for r in result]
        assert names.index("1") < names.index("2")
        assert names.index("2") < names.index("a_case")

    def test_empty_dir(self, tmp_path):
        assert list_cases(str(tmp_path)) == []

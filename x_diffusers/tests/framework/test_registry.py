"""
Unit tests for x_diffusers/framework/pipeline/registry.py.

Covers:
- _read_model_index: existing file, missing file
- register_hf_pipeline_class: registration, duplicate raises
- get_pipeline_cls_by_hf_class_name: missing _class_name raises, unregistered class raises, registered class returns
- is_registered_mappings: registered, not registered, missing _class_name
"""
import json
import os
from unittest.mock import patch, MagicMock

import pytest

from x_diffusers.framework.pipeline.registry import (
    _read_model_index,
    register_hf_pipeline_class,
    get_pipeline_cls_by_hf_class_name,
    is_registered_mappings,
    _CLASSNAME_TO_PIPELINE,
)


class TestReadModelIndex:
    def test_existing_file(self, tmp_path):
        idx = tmp_path / "model_index.json"
        idx.write_text(json.dumps({"_class_name": "FooPipeline"}), encoding="utf-8")
        result = _read_model_index(str(tmp_path))
        assert result["_class_name"] == "FooPipeline"

    def test_missing_file(self, tmp_path):
        result = _read_model_index(str(tmp_path / "nonexistent"))
        assert result == {}


class TestRegisterHfPipelineClass:
    def test_registers_class(self):
        original = _CLASSNAME_TO_PIPELINE.copy()
        try:
            @register_hf_pipeline_class("TestPipeline_123")
            class TestPipe:
                pass
            assert _CLASSNAME_TO_PIPELINE["TestPipeline_123"] is TestPipe
        finally:
            _CLASSNAME_TO_PIPELINE.clear()
            _CLASSNAME_TO_PIPELINE.update(original)

    def test_duplicate_raises(self):
        original = _CLASSNAME_TO_PIPELINE.copy()
        try:
            @register_hf_pipeline_class("TestPipeline_dup")
            class A:
                pass
            with pytest.raises(ValueError, match="Duplicate"):
                @register_hf_pipeline_class("TestPipeline_dup")
                class B:
                    pass
        finally:
            _CLASSNAME_TO_PIPELINE.clear()
            _CLASSNAME_TO_PIPELINE.update(original)


class TestGetPipelineClsByHfClassName:
    def test_missing_class_name_raises(self, tmp_path):
        idx = tmp_path / "model_index.json"
        idx.write_text(json.dumps({}), encoding="utf-8")
        with pytest.raises(ValueError, match="_class_name missing"):
            get_pipeline_cls_by_hf_class_name(str(tmp_path), torch_dtype=MagicMock())

    def test_unregistered_class_raises(self, tmp_path):
        idx = tmp_path / "model_index.json"
        idx.write_text(json.dumps({"_class_name": "UnregisteredPipe"}), encoding="utf-8")
        with pytest.raises(ValueError, match="not registered"):
            get_pipeline_cls_by_hf_class_name(str(tmp_path), torch_dtype=MagicMock())

    def test_registered_class_returns_pipeline(self, tmp_path):
        original = _CLASSNAME_TO_PIPELINE.copy()
        try:
            mock_cls = MagicMock()
            mock_cls.from_pretrained.return_value = "pipeline_instance"
            _CLASSNAME_TO_PIPELINE["RegisteredPipe_123"] = mock_cls
            idx = tmp_path / "model_index.json"
            idx.write_text(json.dumps({"_class_name": "RegisteredPipe_123"}), encoding="utf-8")
            result = get_pipeline_cls_by_hf_class_name(str(tmp_path), torch_dtype=MagicMock())
            assert result == "pipeline_instance"
        finally:
            _CLASSNAME_TO_PIPELINE.clear()
            _CLASSNAME_TO_PIPELINE.update(original)


class TestIsRegisteredMappings:
    def test_registered(self, tmp_path):
        original = _CLASSNAME_TO_PIPELINE.copy()
        try:
            _CLASSNAME_TO_PIPELINE["SomePipe_456"] = MagicMock()
            idx = tmp_path / "model_index.json"
            idx.write_text(json.dumps({"_class_name": "SomePipe_456"}), encoding="utf-8")
            assert is_registered_mappings(str(tmp_path)) is True
        finally:
            _CLASSNAME_TO_PIPELINE.clear()
            _CLASSNAME_TO_PIPELINE.update(original)

    def test_not_registered(self, tmp_path):
        idx = tmp_path / "model_index.json"
        idx.write_text(json.dumps({"_class_name": "UnknownPipe"}), encoding="utf-8")
        assert is_registered_mappings(str(tmp_path)) is False

    def test_missing_class_name(self, tmp_path):
        idx = tmp_path / "model_index.json"
        idx.write_text(json.dumps({}), encoding="utf-8")
        assert is_registered_mappings(str(tmp_path)) is False

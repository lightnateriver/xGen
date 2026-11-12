"""Image text precision defaults and explicit rollback, without checkpoint loading."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from x_diffusers.adaptor import h3_image


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("image_cli", ROOT / "example/diffusers/infer_minimax_h3_image.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("backend,explicit,expected", [
    ("npu", None, True), ("npu", True, True), ("npu", False, False),
    ("cuda", None, False), ("cuda", False, False),
])
def test_backend_default_and_explicit_precision(backend, explicit, expected):
    args = SimpleNamespace(image_text_int8=explicit)
    h3_image.resolve_image_text_options(args, backend)
    assert args.image_text_int8 is expected
    h3_image.resolve_image_text_options(args, backend)
    assert args.image_text_int8 is expected


@pytest.mark.parametrize("backend,expected", [("npu", True), ("cuda", False)])
def test_python_arguments_without_new_attribute(backend, expected):
    args = SimpleNamespace()
    h3_image.resolve_image_text_options(args, backend)
    assert args.image_text_int8 is expected


def test_explicit_int8_rejected_on_cuda():
    with pytest.raises(ValueError, match="requires NPU"):
        h3_image.resolve_image_text_options(SimpleNamespace(image_text_int8=True), "cuda")


@pytest.mark.parametrize("backend", ["npu", "cuda"])
@pytest.mark.parametrize("flags,expected", [([], None), (["--image_text_int8"], True),
    (["--no-image_text_int8"], False), (["--image_text_int8", "--no-image_text_int8"], False)])
def test_cli_resolves_auto_and_explicit_choices(backend, flags, expected):
    args = cli.build_parser().parse_args(["--pretrained_model_name_or_path", "weights", "--prompt", "test",
                                         "--save_path", "output.png", *flags])
    assert args.image_text_int8 is expected
    if backend == "cuda" and expected is True:
        with pytest.raises(ValueError, match="requires NPU"):
            h3_image.resolve_image_text_options(args, backend)
    else:
        h3_image.resolve_image_text_options(args, backend)
        assert args.image_text_int8 is (backend == "npu" if expected is None else expected)


@pytest.mark.parametrize("entry", ["load_pipeline", "run_image"])
@pytest.mark.parametrize("backend,explicit,expected", [
    ("npu", None, True), ("npu", False, False), ("cuda", None, False),
])
def test_python_entry_resolves_before_validation_without_loading_weights(monkeypatch, entry, backend, explicit, expected):
    args = SimpleNamespace(image_text_int8=explicit)
    class ReachedValidation(Exception):
        pass
    def validate(actual, world):
        assert actual is args and actual.image_text_int8 is expected
        raise ReachedValidation
    monkeypatch.setattr(h3_image, "device_type", lambda: backend)
    monkeypatch.setattr(h3_image, "validate_options", validate)
    with pytest.raises(ReachedValidation):
        getattr(h3_image, entry)(args)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_default_accepts_matching_text_tp_and_keeps_other_input_guards(tmp_path, world):
    for name in ["modular_model_index.json", "transformer_diffusers/config.json", "vae_diffusers/config.json",
                 "text_encoder/config.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "pdd_config.json").write_text('{"num_inference_steps":6,"sigmas":[0,0,0,0,0,0,0]}')
    args = SimpleNamespace(sp=world, text_encoder_tp=world, num_inference_steps=6, width=4096, height=4096,
                           min_generation_area=4194304, image_path_list=[], pretrained_model_name_or_path=str(tmp_path))
    h3_image.resolve_image_text_options(args, "npu")
    h3_image.validate_options(args, world)
    args.num_inference_steps = 46
    with pytest.raises(ValueError, match="six PDD"):
        h3_image.validate_options(args, world)

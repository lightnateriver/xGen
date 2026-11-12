"""Opt-in NPU H3 fusion settings shared by CLI and service."""
import argparse
import os
from contextlib import nullcontext


H3_FUSION_FEATURES = (
    "swiglu", "gate_residual", "norm_quant", "rope", "phaa_layout",
    "laser_preprocess", "text_cache", "shared_qkv", "merge_quant",
)


def resolve_h3_features(args):
    """CLI list overrides environment; individual flags override that list."""
    value = getattr(args, "h3_opts", None)
    if value is None:
        value = os.environ.get("H3_OPT_FEATURES", "")
        if "H3_OPT_FEATURES" not in os.environ and os.environ.get("H3_OPT_X1_X5") == "1":
            value = "shared_qkv,merge_quant"
    selected = set()
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            selected.update(H3_FUSION_FEATURES)
        elif token in ("none", "off"):
            selected.clear()
        elif token.startswith("-") and token[1:].replace("-", "_") in H3_FUSION_FEATURES:
            selected.discard(token[1:].replace("-", "_"))
        elif token.replace("-", "_") in H3_FUSION_FEATURES:
            selected.add(token.replace("-", "_"))
        else:
            raise ValueError(f"Unknown H3 fusion feature: {token!r}")
    for name in H3_FUSION_FEATURES:
        override = getattr(args, "h3_opt_" + name, None)
        if override is True:
            selected.add(name)
        elif override is False:
            selected.discard(name)
    args.h3_fused_ops = tuple(name for name in H3_FUSION_FEATURES if name in selected)
    return args.h3_fused_ops


def add_arguments(group):
    group.add_argument("--h3-opts", "--h3_opts", default=None,
                       help="NPU H3 fusions: comma-separated names, all, none or -feature. Defaults to H3_OPT_FEATURES.")
    for name in H3_FUSION_FEATURES:
        flag = "--h3-opt-" + name.replace("_", "-")
        group.add_argument(flag, "--h3_opt_" + name, dest="h3_opt_" + name,
                           action=argparse.BooleanOptionalAction, default=None)


def validate(args, backend):
    features = resolve_h3_features(args)
    if features and (backend != "npu" or args.model != "MiniMax-H3"):
        raise ValueError("H3 fusions selected by --h3-opts require NPU MiniMax-H3")
    if getattr(args, "atten", "none") == "vc" and backend != "npu":
        raise ValueError("VC-Attention requires Ascend NPU")
    return features


def configure_model(pipe, args, backend):
    features = validate(args, backend)
    if args.model == "MiniMax-H3":
        configure = getattr(pipe.transformer, "configure_fused_ops", None)
        if features or getattr(pipe.transformer, "_h3_fused_ops", ()):
            if configure is None:
                raise ValueError("Selected H3 transformer does not support NPU fusions")
            configure(features)


def request_context(transformer):
    context = getattr(transformer, "inference_context", None)
    return context() if context is not None else nullcontext()

import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import wrap, lambda_auto_wrap_policy, size_based_auto_wrap_policy, \
    transformer_auto_wrap_policy
from transformers.models.umt5.modeling_umt5 import UMT5Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from ..operator.matmul import WeightQuantLinearModule


def _patch_missing_reset_parameters(module):
    """给所有缺失 reset_parameters 的子模块补一个 no-op。

    PyTorch FSDP 的 _materialize_meta_module 会遍历所有子模块调用 reset_parameters()
    来从 meta device 物化参数（分配内存），但物化后的值会被 rank 0 的广播覆盖。
    reset_parameters() 是 nn.Linear/nn.Conv2d 等内置模块的约定，不是 nn.Module 的接口要求。
    自定义模块（如 UMT5LayerNorm、Qwen3VL RMSNorm、MiniMaxH3RotaryPosEmbed）没有实现它，
    会导致 AttributeError。这里给所有缺失的子模块补一个 no-op，使其通过 FSDP 的物化检查。
    """
    for sub in module.modules():
        if not hasattr(sub, "reset_parameters"):
            sub.reset_parameters = lambda *args, **kwargs: None


def _fsdp_param_init_fn(module):
    """FSDP param_init_fn: 将 meta device 上的参数/缓冲区物化到 NPU。

    替代 FSDP 默认的 _materialize_meta_module，后者依赖 reset_parameters() 来物化参数。
    问题在于：
    1. 自定义模块（MiniMaxH3RotaryPosEmbed、UMT5LayerNorm 等）没有 reset_parameters
    2. _materialize_meta_module 的 try/except 在循环外部，一个模块失败会导致后续所有模块
       都停留在 meta device 上，且 warning 在多进程输出中被淹没

    本函数只做 to_empty(device=npu)，不调用 reset_parameters。
    参数值未初始化，但 sync_module_states=True 会从 rank 0 广播正确值。
    """
    module.to_empty(device=torch.npu.current_device(), recurse=False)


def _sync_ignored_modules(ignored_modules):
    """物化并同步 FSDP ignored_modules 中的参数到 NPU。

    FSDP 的 sync_module_states=True 只同步 FSDP 管理的模块。
    ignored_modules（如 WeightQuantLinearModule）在非 rank-0 上停留在 meta device，
    在 rank 0 上停留在 CPU — 两者都不在 NPU 上，导致 forward 时 device mismatch。

    本函数在 FSDP 包装后调用：
    1. 非 rank-0: to_empty 从 meta 物化到 NPU（值未初始化）
    2. rank 0: .to() 从 CPU 移到 NPU
    3. 从 rank 0 广播正确权重到所有 rank
    """
    if not ignored_modules:
        return

    device = torch.npu.current_device()

    if not dist.is_initialized():
        # 单进程：直接移到 NPU
        for module in ignored_modules:
            module.to(device)
        return

    world_size = dist.get_world_size()

    for module in ignored_modules:
        # Check if any parameter is on meta (transformer_fsdp moves to meta
        # for non-rank-0; transformer_lambda_fsdp does not)
        has_meta = any(p.is_meta for p in module.parameters(recurse=True))
        if has_meta:
            # Parameters on meta: materialize to NPU (values uninitialized)
            module.to_empty(device=device, recurse=True)
        else:
            # Parameters on CPU or already on NPU: just move
            module.to(device)

    # 从 rank 0 广播，确保所有 rank 权重一致
    if world_size > 1:
        for module in ignored_modules:
            for param in module.parameters(recurse=True):
                if param is not None:
                    dist.broadcast(param, src=0)
            for buf in module.buffers(recurse=True):
                dist.broadcast(buf, src=0)


def transformer_fsdp(pipe, is_need_dtype, dtype=torch.bfloat16, transformer_2=False):
    transformer_size_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=20000
    )
    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    if is_need_dtype:
        pipe.transformer = pipe.transformer.to(torch.bfloat16)

    # Collect WeightQuantLinearModule instances to ignore during FSDP flattening.
    # FSDP cannot flatten integer dtype tensors (int8/int4 from matmul_quant).
    weight_quant_layers = []
    def _recursive_search(module, prefix=''):
        for name, child in module.named_children():
            child_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, WeightQuantLinearModule):
                weight_quant_layers.append(child)
            _recursive_search(child, child_name)
    _recursive_search(pipe.transformer)

    # For sync_module_states to work, only rank 0 should hold real params.
    if dist.is_initialized() and dist.get_rank() != 0:
        pipe.transformer = pipe.transformer.to_empty(device="meta")

    pipe.transformer = FSDP(pipe.transformer,
                            auto_wrap_policy=transformer_size_wrap_policy,
                            mixed_precision=mp_policy,
                            device_id=torch.npu.current_device(),
                            sync_module_states=True,
                            param_init_fn=_fsdp_param_init_fn,
                            ignored_modules=weight_quant_layers if weight_quant_layers else None,)

    # Materialize and sync FSDP-ignored modules (WeightQuantLinearModule).
    # These were moved to meta (non-rank-0) or left on CPU (rank 0) and
    # are not synced by FSDP's sync_module_states. Without this, forward
    # pass fails with "Tensor on device meta is not on the expected device".
    _sync_ignored_modules(weight_quant_layers)

    if transformer_2:
        if is_need_dtype:
            pipe.transformer_2 = pipe.transformer_2.to(torch.bfloat16)
        weight_quant_layers_2 = []
        _recursive_search_t2 = lambda m, p='': [
            (weight_quant_layers_2.append(c), _recursive_search_t2(c, f'{p}.{n}' if p else n))
            for n, c in m.named_children()
        ]
        _recursive_search_t2(pipe.transformer_2)
        if dist.is_initialized() and dist.get_rank() != 0:
            pipe.transformer_2 = pipe.transformer_2.to_empty(device="meta")
        pipe.transformer_2 = FSDP(pipe.transformer_2,
                                  auto_wrap_policy=transformer_size_wrap_policy,
                                  mixed_precision=mp_policy,
                                  device_id=torch.npu.current_device(),
                                  sync_module_states=True,
                                  param_init_fn=_fsdp_param_init_fn,
                                  ignored_modules=weight_quant_layers_2 if weight_quant_layers_2 else None,)
        _sync_ignored_modules(weight_quant_layers_2)


def transformer_lambda_fsdp(
        model,
        device_id,
        is_need_dtype,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        process_group=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
):
    if is_need_dtype:
        model = model.to(param_dtype)

    weight_quant_layers = []

    def _recursive_search(module, prefix=''):
        for name, child in module.named_children():
            child_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, WeightQuantLinearModule):
                weight_quant_layers.append(child)
            _recursive_search(child, child_name)

    _recursive_search(model)

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype),
        device_id=device_id,
        ignored_modules=weight_quant_layers,
        sync_module_states=sync_module_states)
    _sync_ignored_modules(weight_quant_layers)
    return model


def text_encoder_fsdp(pipe, model_type: str, dtype=torch.bfloat16):
    if model_type == "t5":
        transformer_cls = {UMT5Block}
        pipe.text_encoder = pipe.text_encoder.to(dtype)
    elif model_type == "llama":
        transformer_cls = {LlamaDecoderLayer}
    elif model_type == "qwen3vl":
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLDecoderLayer
            transformer_cls = {Qwen3VLDecoderLayer}
        except ImportError:
            transformer_cls = set()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls,
    )

    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    # For sync_module_states to work, only rank 0 should hold real params.
    # Other ranks move to meta device to avoid loading full model on every NPU.
    if dist.is_initialized() and dist.get_rank() != 0:
        pipe.text_encoder = pipe.text_encoder.to_empty(device="meta")

    pipe.text_encoder = FSDP(pipe.text_encoder,
                             auto_wrap_policy=auto_wrap_policy,
                             mixed_precision=mp_policy,
                             device_id=torch.npu.current_device(),
                             sync_module_states=True,
                             param_init_fn=_fsdp_param_init_fn,
                             )


def fsdp_init(pipe, args, is_need_dtype, te_model_type: str, dtype=torch.bfloat16, transformer_2=False):
    # Check if distributed computing is available
    if not dist.is_available():
        raise RuntimeError(
            "Distributed computing is not available. Please ensure that PyTorch is installed with distributed support.")

    # Initialize FSDP for transformer
    if args.fsdp in ["transformer", "all"]:
        if "Wan" in args.model:
            pipe.transformer = transformer_lambda_fsdp(pipe.transformer, torch.npu.current_device(), is_need_dtype)
            if transformer_2:
                pipe.transformer_2 = transformer_lambda_fsdp(pipe.transformer_2, torch.npu.current_device(),
                                                             is_need_dtype)
            if args.joint:
                pipe.transformer_3 = transformer_lambda_fsdp(pipe.transformer_3, torch.npu.current_device(),
                                                             is_need_dtype) 
        else:
            transformer_fsdp(pipe, is_need_dtype, dtype=dtype, transformer_2=transformer_2)

    # Initialize FSDP for text encoder
    if args.fsdp in ["text_encoder", "all"]:
        text_encoder_fsdp(pipe, te_model_type, dtype=dtype)

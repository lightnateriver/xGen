"""ComfyUI loading and prompt hooks; model mechanisms receive explicit settings."""
import fnmatch
import logging
import os

from .env import (
    get_sp_size, get_phaa_split, get_quant, get_fsdp_enabled,
    get_adaln_cache_file, get_adaln_transformer_path,
    resolve_prompt_options,
)

logger = logging.getLogger("x_comfyui")


def _clear_clip_related_cache(executor, prompt):
    """Clear cache entries only for CLIPLoader nodes and their dependents.

    Unlike clearing the entire cache, this preserves VAE/UNet cache entries,
    so only the text encoder and nodes depending on it are reloaded.
    Used for text_tp changes.
    """
    cache = executor.caches.outputs
    if not getattr(cache, "initialized", False):
        return

    # 1. Find all CLIPLoader node IDs
    clip_loader_ids = set()
    for node_id, node_def in prompt.items():
        if isinstance(node_def, dict) and node_def.get("class_type") == "CLIPLoader":
            clip_loader_ids.add(node_id)

    if not clip_loader_ids:
        return

    # 2. Find all transitive dependents (BFS)
    affected_ids = set(clip_loader_ids)
    changed = True
    while changed:
        changed = False
        for node_id, node_def in prompt.items():
            if node_id in affected_ids:
                continue
            if not isinstance(node_def, dict):
                continue
            for input_val in node_def.get("inputs", {}).values():
                if isinstance(input_val, list) and len(input_val) == 2:
                    ancestor_id = str(input_val[0])
                    if ancestor_id in affected_ids:
                        affected_ids.add(node_id)
                        changed = True
                        break

    # 3. Clear cache entries for affected nodes
    for node_id in affected_ids:
        key = cache.cache_key_set.get_data_key(node_id)
        if key is not None and key in cache.cache:
            del cache.cache[key]
        subcache_key = cache.cache_key_set.get_subcache_key(node_id)
        if subcache_key is not None and subcache_key in cache.subcaches:
            del cache.subcaches[subcache_key]

    logger.info(
        f"x_comfyui: cleared cache for {len(affected_ids)} nodes "
        f"(CLIPLoader + dependents)"
    )


def apply_runtime_options(options):
    """Apply resolved compute switches, including environment fallback on removal."""
    from x_base import attention_manager
    import x_base.phaa.globals as phaa
    from x_base.operator.rope import rope_manager

    attention_manager.disable_sage_attention()
    attention_manager.disable_laser_attention()
    attention_manager.disable_rainfusion_attention()
    attention_manager.disable_sparse_attention()
    enable = {
        "sageattn": attention_manager.enable_sage_attention,
        "laserattn": attention_manager.enable_laser_attention,
        "rainfusion": attention_manager.enable_rainfusion_attention,
        "sparse": attention_manager.enable_sparse_attention,
    }
    if options.attention_mode != "sdpa":
        enable[options.attention_mode]()
    phaa.ENABLE_PHAA = options.phaa_split > 1
    phaa.PHAA_SPLIT_NUM = options.phaa_split if phaa.ENABLE_PHAA else None
    if options.rope_fused:
        rope_manager.enable_rope_fused()
    else:
        rope_manager.is_rope_fused = False


def install_model_hooks():
    """Install application hooks once, after registering model compute patches."""
    import execution
    from safetensors.torch import load_file
    from x_comfyui.framework.transformer import minimax_h3 as model

    BaseModel = model.BaseModel
    model.patch_model_classes()
    if getattr(BaseModel, "_xgen_h3_loading_hooks", False):
        return

    # Patch BaseModel.__init__ for auto SP + quant + AdaLN cache
    _original_base_init = BaseModel.__init__

    def _patched_base_init(self, *args, **kwargs):
        _original_base_init(self, *args, **kwargs)
        if hasattr(self, "diffusion_model"):
            model._apply_sp_and_quant(
                self, sp_size=get_sp_size(), phaa_split=get_phaa_split(),
                cache_file=get_adaln_cache_file(),
            )

    BaseModel.__init__ = _patched_base_init

    # Patch BaseModel.load_model_weights for auto FSDP + adaln stripping
    _original_load_weights = BaseModel.load_model_weights

    def _patched_load_weights(self, sd, unet_prefix="", assign=False):
        # If AdaLN cache is active, strip adaln weights to save ~26GB.
        diffusion_model = getattr(self, "diffusion_model", None)
        if (diffusion_model is not None
                and getattr(diffusion_model, "_npu_adaln_strip", False)):
            transformer_path = get_adaln_transformer_path()
            if transformer_path:
                # Strategy 1: load pre-stripped checkpoint
                if os.path.isfile(transformer_path):
                    sd = load_file(transformer_path)
                elif os.path.isdir(transformer_path):
                    sd = {}
                    for fname in sorted(os.listdir(transformer_path)):
                        if fname.endswith(".safetensors"):
                            sd.update(load_file(
                                os.path.join(transformer_path, fname)))
                else:
                    raise FileNotFoundError(
                        f"X_ADALN_TRANSFORMER_PATH not found: "
                        f"{transformer_path}")
                logger.info(
                    f"x_comfyui: loaded stripped checkpoint from "
                    f"{transformer_path} ({len(sd)} keys)"
                )
            else:
                # Strategy 2: filter adaln keys from original state dict
                ADALN_PATTERNS = [
                    "*blocks.*.adaln_proj.linear.weight",
                    "*blocks.*.adaln_proj.linear.bias",
                    "*final_layer.adaln_proj.linear.weight",
                    "*final_layer.adaln_proj.linear.bias",
                ]
                if not getattr(diffusion_model, "use_adaln_curves", False):
                    ADALN_PATTERNS.extend([
                        "*time_embedder.linear_1.weight",
                        "*time_embedder.linear_1.bias",
                        "*time_embedder.linear_2.weight",
                        "*time_embedder.linear_2.bias",
                    ])
                sd = {k: v for k, v in sd.items()
                      if not any(fnmatch.fnmatch(k, p)
                                 for p in ADALN_PATTERNS)}
                logger.info(
                    f"x_comfyui: filtered adaln keys from state dict "
                    f"({len(sd)} keys remaining)"
                )

        result = _original_load_weights(self, sd, unet_prefix, assign)
        # Apply quantization AFTER weight loading
        if not getattr(self, "_npu_quant_applied", False):
            model._apply_quant(self, quant=get_quant())
            self._npu_quant_applied = True
        if not getattr(self, "_npu_fsdp_applied", False):
            model._apply_fsdp(self, enabled=get_fsdp_enabled(), quant=get_quant())
            self._npu_fsdp_applied = True
        # Release any remaining adaln weights after loading
        diffusion_model = getattr(self, "diffusion_model", None)
        if diffusion_model is not None and hasattr(diffusion_model, "_adaln_cache"):
            model.release_adaln_weights_for_comfyui(diffusion_model)
        return result

    BaseModel.load_model_weights = _patched_load_weights

    # Patch PromptExecutor.execute_async to detect quant/text changes between runs.
    _original_execute_async = execution.PromptExecutor.execute_async

    last_options = None

    async def _patched_execute_async(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        nonlocal last_options
        if extra_data is None:
            extra_data = {}
        if execute_outputs is None:
            execute_outputs = []
        options = resolve_prompt_options(prompt)
        apply_runtime_options(options)
        if last_options is not None:
            if last_options.model_key != options.model_key:
                self.caches.outputs.cache.clear()
                self.caches.outputs.subcaches.clear()
                logger.info("x_comfyui: effective model settings changed; clearing model cache")
            elif last_options.text_tp != options.text_tp:
                _clear_clip_related_cache(self, prompt)
                logger.info("x_comfyui: effective text TP size changed; clearing CLIP cache")
        # Track the settings already visible to model loaders, even if execution fails.
        last_options = options
        return await _original_execute_async(self, prompt, prompt_id, extra_data, execute_outputs)

    execution.PromptExecutor.execute_async = _patched_execute_async

    logger.info("x_comfyui: MiniMax-H3 model patches applied (Attention, AdalnProj, _forward, BaseModel)")


    BaseModel._xgen_h3_loading_hooks = True

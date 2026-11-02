import folder_paths


class NPUWanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder", }),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "quantization": (["disabled", "w8a8"], {"default": "disabled", "tooltip": "Optional quantization method, 'disabled' acts as no quantization, 'w8a8' acts as online weight int8 and activation int8 quantization)"}),
                "load_device": (["main_device", "offload_device"], {"default": "offload_device",
                                                                    "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
            "optional": {
                "attention_mode": ([
                                       "sdpa",
                                       "sageattn",
                                       "laserattn",
                                   ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS",),
                "block_swap_args": ("BLOCKSWAPARGS",),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None,
                                                                 "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "extra_model": ("VACEPATH", {"default": None,
                                             "tooltip": "Extra model to add to the main model, ie. VACE or MTV Crafter"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None,
                                                                 "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
                "fantasyportrait_model": ("FANTASYPORTRAITMODEL", {"default": None, "tooltip": "FantasyPortrait model"}),
                "rms_norm_function": (["default", "pytorch"], {"default": "default",
                                                               "tooltip": "RMSNorm function to use, 'pytorch' is the new native torch RMSNorm, which is faster (when not using torch.compile mostly) but changes results slightly. 'default' is the original WanRMSNorm"}),
            }
        }

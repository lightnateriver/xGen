#!/bin/bash
#==============================================================
# MiniMax-H3 Turbo LoRA Merge
#
# 将 ComfyUI 格式 LoRA 融合进基座模型权重，生成标准 diffusers
# 格式 safetensors，可直接用于 x_gen 推理。
#
# 纯 CPU 操作，不需要 NPU/GPU。需要 ~16GB RAM。
# 修改下方路径变量后直接 bash 运行。
#==============================================================

# ---------- 路径配置 ----------
# ComfyUI LoRA 文件路径
LORA_PATH="/data/yjb/minimax-h3/x_gen/example/diffusers/weights/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# 基座模型 transformer 目录（top-level diffusers 格式，含 diffusion_pytorch_model-*.safetensors）
BASE_DIR="/data/models/minimax/transformer"

# 输出目录（merge 后的 8 步权重，top-level diffusers 格式）
OUTPUT_DIR="/data/models/minimax/transformer_8step"

# ---------- 运行 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/merge_lora.py" \
    --lora-path "${LORA_PATH}" \
    --base-dir "${BASE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --verify

#!/bin/bash
#==============================================================
# AdaLN 预计算缓存 + 剥离权重（推荐配置）
#
# 生成缓存文件 + 精简模型，单卡 + group offload 即可运行。
# 修改下方路径变量后直接 bash 运行。
#==============================================================

# ---------- 路径配置（按实际环境修改） ----------
# model_path 指向顶层 transformer 目录（不是 FL2VA/ 子目录）。
# 脚本自动推导上层 pipeline 路径（需含 modular_model_index.json）。
TRANSFORMER_PATH="/data/models/minimax/transformer"
CACHE_FILE="/data/models/minimax/adaln_cache/adaln_cache_50step.safetensors"
STRIP_DIR="/data/models/minimax/transformer_adaln_cache"

# ---------- 采样参数 ----------
STEPS=50
VIDEO_SHIFT=12.0
AUDIO_SHIFT=3.0

# ---------- 运行 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "${CACHE_FILE}")"

python "${SCRIPT_DIR}/precompute_adaln_cache.py" \
    --model_path "${TRANSFORMER_PATH}" \
    --steps ${STEPS} \
    --video_shift ${VIDEO_SHIFT} \
    --audio_shift ${AUDIO_SHIFT} \
    --output "${CACHE_FILE}" \
    --strip_output "${STRIP_DIR}"

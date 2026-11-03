#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the local H3 modular pipeline directory}"
: "${NVFP4_CHECKPOINT:?Set NVFP4_CHECKPOINT to the pruned FL2VA checkpoint}"
: "${FIRST_FRAME:?Set FIRST_FRAME to an input image}"
: "${PROMPT_FILE:?Set PROMPT_FILE to a UTF-8 prompt file}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Add ffmpeg to PATH before launching FL2VA audio/video inference." >&2
    exit 1
fi

export XGEN_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
GPUS="${GPUS:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./h3-nvfp4.mp4}"
if [[ "$OUTPUT_PATH" != */* ]]; then
    OUTPUT_PATH="./$OUTPUT_PATH"
fi

torchrun --standalone --nproc_per_node "$GPUS" "$SCRIPT_DIR/../infer.py" \
    --model MiniMax-H3 --task_type fl2va \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --h3_nvfp4_checkpoint "$NVFP4_CHECKPOINT" \
    --keyframe_image "$FIRST_FRAME" --prompt "$(cat -- "$PROMPT_FILE")" \
    --save_path "$OUTPUT_PATH" --num_inference_steps "${NUM_INFERENCE_STEPS:-51}" \
    --width 1344 --height 768 --frames 124 --save_fps 24 --seed 42 \
    --matmul_quant none --vae_matmul_quant none --vae_lightning none \
    --atten "${ATTENTION_BACKEND:-none}" --text_encoder_tp 1 --sp "$GPUS" "$@"

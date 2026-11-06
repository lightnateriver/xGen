#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the original BF16 FL2VA modular checkpoint}"
: "${PROMPT_FILE:?Set PROMPT_FILE to the matching UTF-8 prompt}"
GPUS="${GPUS:-8}"
PROFILE="${PROFILE:-fp8-bf16}"
TASK_TYPE="${TASK_TYPE:-t2va}"
OUTPUT_PATH="${OUTPUT_PATH:-./h3-fp8.mp4}"
[[ "$GPUS" == 8 ]] || { echo 'This profile requires eight GPUs' >&2; exit 2; }
[[ "$PROFILE" == fp8-bf16 || "$PROFILE" == none ]] || { echo 'PROFILE must be fp8-bf16 or none' >&2; exit 2; }
case "${SECONDS_VIDEO:-5}" in
    5) frames=124 ;; 10) frames=243 ;; 15) frames=362 ;;
    *) echo 'SECONDS_VIDEO must be 5, 10 or 15' >&2; exit 2 ;;
esac
image_args=()
case "$TASK_TYPE" in
    t2va) ;;
    fl2va) : "${FIRST_FRAME:?Set FIRST_FRAME for FL2VA}"; image_args=(--keyframe_image "$FIRST_FRAME") ;;
    *) echo 'TASK_TYPE must be t2va or fl2va' >&2; exit 2 ;;
esac
export XGEN_DEVICE=cuda TOKENIZERS_PARALLELISM=false
unset XGEN_H3_CUDA_VAE_TILES
unset XGEN_H3_FUSED_INDEXED_AFFINE XGEN_H3_FUSED_PARTIAL_ROPE XGEN_H3_FUSED_QK_RMS
unset XGEN_H3_FUSED_FF_AMAX XGEN_H3_FUSED_SWIGLU XGEN_H3_NVFP4_FUSED_AMAX
unset XGEN_PHAA_FP8 XGEN_PHAA_ASYNC_EXCHANGE XGEN_PHAA_DIRECT_RETURN
unset XGEN_PHAA_GATHER_PRIORITY XGEN_PHAA_FP8_ROW_TILING XGEN_PHAA_DEVICE_JOIN
unset XGEN_PHAA_FUSED_RETURN_PACK XGEN_PHAA_DIRECT_GATHER XGEN_PHAA_DUAL_COMM
export XGEN_PHAA_CHUNK_ORDER=natural NCCL_MAX_CTAS=2
torchrun --standalone --nproc_per_node "$GPUS" "$SCRIPT_DIR/../infer.py" \
    --model MiniMax-H3 --task_type "$TASK_TYPE" \
    --pretrained_model_name_or_path "$MODEL_PATH" --h3_cuda_precision "$PROFILE" \
    --prompt="$(cat -- "$PROMPT_FILE")" --save_path "$OUTPUT_PATH" \
    --dtype bf16 --num_inference_steps "${NUM_INFERENCE_STEPS:-20}" \
    --width 1344 --height 768 --frames "$frames" --save_fps 24 --seed 42 \
    --sp "$GPUS" --text_encoder_tp "$GPUS" --phaa_num "${PHAA_NUM:-2}" --h3_cuda_vae_sp \
    --matmul_quant none --vae_matmul_quant none --vae_lightning none --atten none \
    "${image_args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the original H3 modular checkpoint}"
: "${PROMPT_FILE:?Set PROMPT_FILE to the matching UTF-8 prompt}"
TASK_TYPE="${TASK_TYPE:-t2va}"
PROFILE="${PROFILE:-fp8-sage}"
[[ "$PROFILE" == fp8-bf16 || "$PROFILE" == fp8-sage ]] || { echo 'PROFILE must be fp8-bf16 or fp8-sage' >&2; exit 2; }
case "${SECONDS_VIDEO:-5}" in
    5) frames=124 ;; 10) frames=243 ;; 15) frames=362 ;;
    *) echo 'SECONDS_VIDEO must be 5, 10 or 15' >&2; exit 2 ;;
esac
image_args=()
case "$TASK_TYPE" in
    t2va) ;;
    fl2va) : "${FIRST_FRAME:?Set FIRST_FRAME for FL2VA}"; image_args=(--keyframe_image "$FIRST_FRAME") ;;
    ref2va) ;; # Pass --reference_paths and the ordered files after this script.
    *) echo 'TASK_TYPE must be t2va, fl2va or ref2va' >&2; exit 2 ;;
esac
export XGEN_DEVICE=cuda TOKENIZERS_PARALLELISM=false NCCL_MAX_CTAS=2
unset XGEN_PHAA_FP8 XGEN_PHAA_ASYNC_EXCHANGE XGEN_PHAA_DIRECT_RETURN
unset XGEN_PHAA_GATHER_PRIORITY XGEN_PHAA_FP8_ROW_TILING XGEN_PHAA_DEVICE_JOIN
unset XGEN_PHAA_FUSED_RETURN_PACK XGEN_PHAA_DIRECT_GATHER XGEN_PHAA_DUAL_COMM
unset XGEN_H3_FUSED_FF_AMAX XGEN_H3_FUSED_SWIGLU XGEN_H3_NVFP4_FUSED_AMAX
export XGEN_PHAA_CHUNK_ORDER=natural XGEN_H3_FUSED_OUTPUT_ADALN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node 8 "$SCRIPT_DIR/../infer.py" \
    --model MiniMax-H3 --task_type "$TASK_TYPE" \
    --pretrained_model_name_or_path "$MODEL_PATH" --h3_cuda_precision "$PROFILE" \
    --prompt="$(cat -- "$PROMPT_FILE")" --save_path "${OUTPUT_PATH:-./h3-tp-cp.mp4}" \
    --dtype bf16 --num_inference_steps "${NUM_INFERENCE_STEPS:-20}" \
    --width 1344 --height 768 --frames "$frames" --save_fps 24 --seed 42 \
    --h3_parallel_mode tp_cp --h3_dit_tp 2 --sp 4 --text_encoder_tp 8 \
    --h3_tp_comm "${TP_COMM:-bf16}" --h3_cp_comm "${CP_COMM:-fp8}" \
    --phaa_num 2 --h3_cuda_vae_sp \
    --matmul_quant none --vae_matmul_quant none --vae_lightning none --atten none \
    "${image_args[@]}" "$@"

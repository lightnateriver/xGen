#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Point XGEN_PYTHONPATH at the installed source trees
# only if x_base / x_diffusers have not already been installed in the NPU env.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the complete MiniMax-H3 Diffusers repository}"
: "${SLA_LORA_PATH:?Set SLA_LORA_PATH to minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors}"
N_NPUS="${N_NPUS:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

task_type=t2va
keyframe_args=()
if [[ -n "${FIRST_FRAME:-}" || -n "${LAST_FRAME:-}" ]]; then
    task_type=fl2va
    [[ -z "${FIRST_FRAME:-}" ]] || keyframe_args+=(--keyframe_image "$FIRST_FRAME")
    [[ -z "${LAST_FRAME:-}" ]] || keyframe_args+=(--keyframe_last_image "$LAST_FRAME")
fi

torchrun --nproc_per_node="$N_NPUS" --master_port="${MASTER_PORT:-29517}" \
    "$SCRIPT_DIR/../infer.py" \
    --model MiniMax-H3 \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --task_type "$task_type" \
    --h3_turbo_sla_lora "$SLA_LORA_PATH" \
    --num_inference_steps 4 \
    --dtype bf16 --atten none --matmul_quant none \
    --sp "$N_NPUS" --ulysses_degree "$N_NPUS" --ring_degree 1 \
    --phaa_num 0 \
    --width 1344 --height 768 --frames 124 --seed 42 --save_fps 24 \
    --save_path "${OUTPUT_PATH:-./outputs/minimax_h3_turbo_sla.mp4}" \
    --prompt "${PROMPT:-A cinematic shot of a sailboat crossing a calm blue lake. Natural water and wind sounds.}" \
    "${keyframe_args[@]}"

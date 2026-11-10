#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/activate-xgen.sh" ]]; then
    source "${PROJECT_ROOT}/activate-xgen.sh"
fi
set -u
cd "${SCRIPT_DIR}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29507}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"

export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export INF_NAN_MODE_FORCE_DISABLE=1

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export N_NPUS="${N_NPUS:-2}"

export TASK_NAME="${TASK_NAME:-minimax_h3_fl2va_2npu}"
export H3_VAE_BACKEND="${H3_VAE_BACKEND:-lightning}"

MODEL_PATH="${MODEL_PATH:-./weights/MiniMax-H3}"
FIRST_FRAME="${FIRST_FRAME:-${SCRIPT_DIR}/astronaut.jpg}"
LAST_FRAME="${LAST_FRAME:-${FIRST_FRAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/${TASK_NAME}.mp4}"
EXTRA_ARGS=()
CACHE_FILE="${ADALN_CACHE_FILE:-${MODEL_PATH}/adaln_cache/adaln_cache_51step.safetensors}"
CUSTOM_TRANSFORMER="${TRANSFORMER_PATH:-${MODEL_PATH}/transformer_adaln_cache}"
if [[ -f "${CACHE_FILE}" && -d "${CUSTOM_TRANSFORMER}" ]]; then
    EXTRA_ARGS+=(--adaln_cache_file "${CACHE_FILE}" --transformer_path "${CUSTOM_TRANSFORMER}")
fi

if [[ ! -f "${FIRST_FRAME}" ]]; then
    echo "Missing keyframe image: ${FIRST_FRAME}. Set FIRST_FRAME." >&2
    exit 1
fi
LAST_FRAME_ARGS=()
if [[ "${FIRST_FRAME_ONLY:-0}" != 1 ]]; then
    if [[ ! -f "${LAST_FRAME}" ]]; then
        echo "Missing keyframe image: ${LAST_FRAME}. Set LAST_FRAME." >&2
        exit 1
    fi
    LAST_FRAME_ARGS=(--keyframe_last_image "${LAST_FRAME}")
fi
mkdir -p "$(dirname -- "${OUTPUT_PATH}")"

if [[ "${ROPE_FUSED:-1}" == 1 ]]; then
    EXTRA_ARGS+=(--rope_fused)
fi

PROMPT='integrated_multimodal_description: [Shot 1] A small robot walks through a sunlit garden. The camera follows slowly.\noverall_soundscape: Gentle footsteps and birdsong.\nnon_diegetic_music: Soft instrumental music.'
if [[ -n "${PROMPT_FILE:-}" ]]; then
    PROMPT="$(cat -- "${PROMPT_FILE}")"
fi

torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT "${INFER_ENTRYPOINT:-../infer.py}" \
         --model MiniMax-H3 \
         --pretrained_model_name_or_path "${MODEL_PATH}" \
         --task_type fl2va \
         --keyframe_image "${FIRST_FRAME}" "${LAST_FRAME_ARGS[@]}" \
         --save_path "${OUTPUT_PATH}" \
         --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
         --width "${WIDTH:-1344}" \
         --height "${HEIGHT:-768}" \
         --frames "${FRAMES:-124}" \
         --seed 42 \
         --save_fps 24 \
         --prompt "${PROMPT}" \
         --matmul_quant "${MATMUL_QUANT:-int8}" \
         --vae_matmul_quant "${VAE_MATMUL_QUANT:-int8}" \
         --atten "${ATTENTION_BACKEND:-laser}" \
         --h3_vae_backend "$H3_VAE_BACKEND" \
         --text_encoder_tp "${TEXT_ENCODER_TP:-$N_NPUS}" \
         --sp $N_NPUS \
         --phaa_num "${PHAA_NUM:-7}" \
         "${EXTRA_ARGS[@]}"

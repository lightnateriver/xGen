#!/usr/bin/env bash
# Eight-card MiniMax-H3 examples through the existing offline entry points.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"

usage() {
  echo "Usage: bash example/diffusers/scripts/run_h3.sh <npu|cuda> <t2va|i2va|r2va> <5|10|15> <768p|2k> [images...]" >&2
  exit 2
}

[[ $# -ge 4 ]] || usage
backend=$1 task=$2 duration=$3 resolution=$4
shift 4
case "$backend" in npu|cuda) ;; *) usage ;; esac
case "$resolution" in 768p|2k) ;; *) usage ;; esac
case "$duration" in
  5) frames=124 ;;
  10) frames=243 ;;
  15) frames=362 ;;
  *) usage ;;
esac

base_transformer=${BASE_TRANSFORMER:-model-cache/base-t2va}
base_adaln=${BASE_ADALN:-model-cache/t2va-51step.safetensors}
case "$task" in
  t2va)
    [[ $# -eq 0 ]] || usage
    input_args=()
    ;;
  i2va|fl2va)
    [[ $# -le 1 ]] || usage
    task=fl2va
    first_frame=${1:-${FIRST_FRAME:-inputs/first-frame.png}}
    input_args=(--keyframe_image "$first_frame")
    ;;
  r2va|ref2va)
    task=ref2va
    base_transformer=${BASE_TRANSFORMER:-model-cache/base-ref2va}
    base_adaln=${BASE_ADALN:-model-cache/ref2va-51step.safetensors}
    if [[ $# -eq 0 ]]; then
      set -- inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
    fi
    input_args=(--reference_paths "$@")
    ;;
  *) usage ;;
esac

if [[ $backend == cuda && $resolution == 2k && $task != fl2va ]]; then
  echo "CUDA 2K requires FL2VA. Use i2va with one first frame, or choose NPU for T2VA/R2VA 2K." >&2
  exit 2
fi
if [[ ${H3_FAST:-0} == 1 && $backend/$task/$duration/$resolution != cuda/fl2va/5/768p ]]; then
  echo "H3_FAST=1 selects the measured CUDA FL2VA 5-second 768p profile." >&2
  exit 2
fi

export XGEN_DEVICE=$backend
export PYTHONPATH="$root/x_base:$root/x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
prompt_file=${PROMPT_FILE:-inputs/prompt.txt}
prompt=$(cat -- "$prompt_file")
output=${OUTPUT_PATH:-outputs/$backend-$task-${duration}s-$resolution.mp4}
mkdir -p -- "$(dirname -- "$output")"
base_model=${BASE_MODEL:-weights/base}
refiner_weights=${REFINER_WEIGHTS:-weights/refiner}
nvfp4_checkpoint=${NVFP4_CHECKPOINT:-weights/nvfp4/minimax_h3_fl2va_pruned_nvfp4.safetensors}

if [[ $backend == npu ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
  export INF_NAN_MODE_FORCE_DISABLE=${INF_NAN_MODE_FORCE_DISABLE:-1}
  if [[ $resolution == 2k ]]; then
    BASE_MODEL=$base_model BASE_TRANSFORMER=$base_transformer BASE_ADALN=$base_adaln \
      REFINER_WEIGHTS=$refiner_weights REFINER_PREPARED=${REFINER_PREPARED:-model-cache/refiner} \
      SECONDS_TO_GENERATE=$duration OUTPUT_VIDEO=$output PROMPT=$prompt \
      exec bash example/diffusers/scripts/infer_minimax_h3_refiner.sh --task_type "$task" "${input_args[@]}"
  fi
  launch=(--master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29530}")
  profile=(--transformer_path "$base_transformer" --adaln_cache_file "$base_adaln"
    --atten laser --matmul_quant int8 --vae_matmul_quant int8 --rope_fused
    --vae_lightning decoder --text_encoder_tp 8 --phaa_num 2)
else
  if [[ ${H3_FAST:-0} == 1 ]]; then
    export XGEN_PHAA_DEVICE_JOIN=1 XGEN_PHAA_ASYNC_EXCHANGE=1
    export XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1
    export XGEN_H3_FUSED_INDEXED_AFFINE=1 XGEN_H3_FUSED_PARTIAL_ROPE=1 XGEN_H3_FUSED_QK_RMS=1
    export XGEN_H3_CUDA_VAE_TILES=1 XGEN_H3_FUSED_FF_AMAX=1
    MODEL_PATH=${MODEL_PATH:-weights/h3} NVFP4_CHECKPOINT=$nvfp4_checkpoint \
      FIRST_FRAME=$first_frame PROMPT_FILE=$prompt_file OUTPUT_PATH=$output GPUS=8 \
      XGEN_FASTVIDEO_RECIPE=5s ATTENTION_BACKEND=sparge XGEN_SPARGE_TOPK=0.5 \
      exec bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh --h3_cuda_vae_sp
  fi
  unset XGEN_PHAA_FP8 XGEN_PHAA_GATHER_PRIORITY XGEN_PHAA_DEVICE_JOIN XGEN_PHAA_ASYNC_EXCHANGE
  unset XGEN_PHAA_DIRECT_RETURN XGEN_PHAA_FP8_ROW_TILING XGEN_PHAA_DUAL_COMM XGEN_PHAA_CHUNK_ORDER
  unset XGEN_PHAA_FUSED_RETURN_PACK XGEN_PHAA_DIRECT_GATHER XGEN_H3_CUDA_VAE_TILES
  unset XGEN_H3_FUSED_INDEXED_AFFINE XGEN_H3_FUSED_PARTIAL_ROPE XGEN_H3_FUSED_QK_RMS
  unset XGEN_H3_FUSED_FF_AMAX XGEN_H3_FUSED_SWIGLU XGEN_H3_NVFP4_FUSED_AMAX
  launch=(--standalone)
  profile=(--atten none --matmul_quant none --vae_matmul_quant none --vae_lightning none
    --text_encoder_tp 8 --phaa_num 4 --h3_cuda_vae_sp)
  if [[ $resolution == 768p && $task != ref2va ]]; then
    profile=(--atten none --matmul_quant none --vae_matmul_quant none --vae_lightning none
      --text_encoder_tp 8 --phaa_num 2 --h3_cuda_vae_sp --h3_cuda_precision "${H3_CUDA_PRECISION:-fp8-bf16}")
    if [[ ${H3_CUDA_PRECISION:-fp8-bf16} != none ]]; then
      export NCCL_MAX_CTAS=2
    fi
  fi
  if [[ $resolution == 2k ]]; then
    base_model=${MODEL_PATH:-weights/h3}
    profile=(--atten none --matmul_quant none --vae_matmul_quant none --vae_lightning none
      --text_encoder_tp 1 --phaa_num 2 --h3_nvfp4_checkpoint "$nvfp4_checkpoint" --h3_refiner
      --h3_refiner_weights "$refiner_weights" --h3_refiner_prepared "${REFINER_PREPARED:-model-cache/refiner-cuda}"
      --h3_refiner_seconds "$duration")
  fi
fi

width=1344 height=768
if [[ $resolution == 2k ]]; then
  width=2560 height=1440
fi
exec torchrun "${launch[@]}" --nproc_per_node=8 --max_restarts=0 "${XGEN_H3_BENCH_WORKER:-example/diffusers/infer.py}" \
  --model MiniMax-H3 --pretrained_model_name_or_path "$base_model" --task_type "$task" \
  --dtype bf16 --sp 8 --width "$width" --height "$height" --frames "$frames" --save_fps 24 \
  --num_inference_steps "${H3_BASE_STEPS:-50}" --seed "${H3_SEED:-42}" --prompt="$prompt" --save_path "$output" \
  "${profile[@]}" "${input_args[@]}"

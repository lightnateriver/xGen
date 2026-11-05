#!/usr/bin/env bash
# Optional experimental SR, using the existing XGEN CLI and model manager.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$root/x_diffusers:$root/x_base${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_MODEL:?Set the original MiniMax-H3 model directory}"
: "${BASE_TRANSFORMER:?Set the matching stripped Base transformer}"
: "${BASE_ADALN:?Set the matching Base AdaLN table}"
: "${REFINER_WEIGHTS:?Set Open-H3-Regenerate-2K source directory}"
: "${REFINER_PREPARED:?Set prepared Refiner directory}"
exec torchrun --nproc_per_node=8 --max_restarts=0 --master_port="${MASTER_PORT:-29558}" \
  "${XGEN_H3_BENCH_WORKER:-$root/example/diffusers/infer.py}" \
  --model MiniMax-H3 --task_type t2va --pretrained_model_name_or_path "$BASE_MODEL" \
  --transformer_path "$BASE_TRANSFORMER" --adaln_cache_file "$BASE_ADALN" \
  --dtype bf16 --num_inference_steps "${H3_BASE_STEPS:-50}" --seed "${H3_SEED:-42}" \
  --width 2560 --height 1440 --frames 124 --save_fps 24 \
  --sp 8 --text_encoder_tp 8 --phaa_num 2 --atten laser --matmul_quant int8 \
  --vae_matmul_quant none --vae_lightning none --inf_vram_blocks_num 0 --rope_fused \
  --h3_refiner --h3_refiner_weights "$REFINER_WEIGHTS" --h3_refiner_prepared "$REFINER_PREPARED" \
  --h3_refiner_seconds "${SECONDS_TO_GENERATE:-5}" \
  --save_path "${OUTPUT_VIDEO:-./h3-refiner-2k.mp4}" \
  --prompt="${PROMPT:-A small robot walks through a sunlit garden.}" "$@"

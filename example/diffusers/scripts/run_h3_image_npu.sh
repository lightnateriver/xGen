#!/usr/bin/env bash
# Selected four-NPU Image deployment: resident DiT W8A8 and Qwen decoder W8A8 TP4.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export XGEN_DEVICE=npu

exec bash "$SCRIPT_DIR/run_h3_image.sh" \
  --atten none --num_inference_steps 6 \
  --image_adaln_table --image_resident --image_int8 --image_text_int8 \
  --image_condition_cache --image_refiner_sp --image_vision_sp \
  --image_vae_parallel --image_ff_chunk 2048 --image_rgb8_output --image_local_input \
  "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL=${H3_IMAGE_MODEL:-$ROOT/weights/MiniMax-H3-Image}
export PYTHONPATH="$ROOT/x_base:$ROOT/x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
torchrun --nnodes=1 --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29673}" --nproc_per_node=4 "$ROOT/example/diffusers/infer_minimax_h3_image.py" \
  --pretrained_model_name_or_path "$MODEL" --sp 4 --text_encoder_tp 4 "$@"

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
frames=124
text_encoder_tp=1
recipe_args=()
case "${XGEN_FASTVIDEO_RECIPE:-}" in
    "") ;;
    5s|15s)
        if [[ "$GPUS" != 8 ]]; then
            echo "XGEN_FASTVIDEO_RECIPE requires GPUS=8" >&2
            exit 1
        fi
        text_encoder_tp=8
        export XGEN_PHAA_FP8=1 XGEN_PHAA_GATHER_PRIORITY=1
        recipe_args=(--phaa_num 4 --h3_nvfp4_resident)
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-sparge}"
        if [[ "$XGEN_FASTVIDEO_RECIPE" == 5s ]]; then
            export XGEN_PHAA_CHUNK_ORDER=natural NCCL_MAX_CTAS=2
        else
            frames=345
            export XGEN_PHAA_CHUNK_ORDER=tail-first NCCL_MAX_CTAS=1
        fi
        ;;
    *)
        echo "XGEN_FASTVIDEO_RECIPE must be 5s or 15s" >&2
        exit 1
        ;;
esac
if [[ "$OUTPUT_PATH" != */* ]]; then
    OUTPUT_PATH="./$OUTPUT_PATH"
fi

torchrun --standalone --nproc_per_node "$GPUS" "$SCRIPT_DIR/../infer.py" \
    --model MiniMax-H3 --task_type fl2va \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --h3_nvfp4_checkpoint "$NVFP4_CHECKPOINT" \
    --keyframe_image "$FIRST_FRAME" --prompt="$(cat -- "$PROMPT_FILE")" \
    --save_path "$OUTPUT_PATH" --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
    --width 1344 --height 768 --frames "$frames" --save_fps 24 --seed 42 \
    --matmul_quant none --vae_matmul_quant none --h3_vae_backend legacy \
    --atten "${ATTENTION_BACKEND:-none}" --text_encoder_tp "$text_encoder_tp" --sp "$GPUS" "${recipe_args[@]}" "$@"

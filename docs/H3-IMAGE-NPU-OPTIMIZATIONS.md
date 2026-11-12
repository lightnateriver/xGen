# MiniMax-H3-Image: v0.6 resident Ascend deployment

This is the six-step Image checkpoint, separate from H3 audio/video workflows. Keep the pinned Diffusers source from `requirements-h3-image.txt`. The original BF16 SP4/offload path remains available when the opt-in flags are absent.

## Selected NPU deployment

The selected NPU recipe retains Qwen3-VL decoder W8A8 with resident text TP4, alongside the resident W8A8 DiT and SP4 optimizations below. The NPU-specific launcher sets these options without changing the generic CLI defaults or the CUDA/video recipes:

```bash
export H3_IMAGE_MODEL="$MODEL_ROOT"
bash example/diffusers/scripts/run_h3_image_npu.sh \
  --prompt 'A product photograph of a green ceramic bowl on a neutral background.' \
  --width 4096 --height 4096 --save_path "$OUTPUT_PNG"
```

Keep the pinned Diffusers source on `PYTHONPATH` and select four idle NPUs before launch. The launcher reuses the existing SP4/text-TP4 entry point and adds `--image_text_int8` plus the resident DiT, AdaLN/cache, refiner/vision/VAE parallelism, FFN chunk2048, RGB8 and local-input flags. Text TP4 is already resident; `--image_text_resident` is unnecessary. For reusable Python workers, retain `image_text_int8=True` in startup arguments and load the pipeline once per rank.

The matched 4096x4096 T2I comparison measured 14.752s BF16 versus 14.617s W8A8 reused means (three samples each), without evidence of stable latency improvement. Peak allocated memory falls from 37.254 to 31.759 GiB per NPU. Only the retained 50 Qwen decoder layers' 350 attention/MLP projections are W8A8; embedding, norm, vision tower and lm_head retain their original precision. See [text W8A8 scope and measurements](H3-IMAGE-TEXT-INT8.md). TP4 text INT8 has been checkpoint-tested on this short-text T2I case, not the full 40-case matrix or reference editing.

## Historical full-matrix configuration

The full matrix uses four 64-GiB Ascend 910B1 cards, SP4, BF16 Qwen text TP4, BF16 interfaces, dynamic W8A8 DiT attention/FFN linears, and the checkpoint's original mixed-precision VAE. It keeps the six PDD heads, FP32 patch input/timestep/output projections, CPU seed 42, and the original tile geometry/overlap blending. The command below reproduces that historical text-BF16 configuration; the selected NPU launcher above additionally enables text W8A8.

```bash
export XGEN_DEVICE=npu
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers:$PINNED_DIFFUSERS/src"
torchrun --nproc_per_node=4 --master_port=29746 \
  example/diffusers/infer_minimax_h3_image.py \
  --pretrained_model_name_or_path "$MODEL_ROOT" \
  --prompt "Replace the background in <Picture 1> with a plain gray backdrop." \
  --image_path_list "$REFERENCE_JPG" --save_path "$OUTPUT_PNG" \
  --width 2048 --height 2048 --num_inference_steps 6 --seed 42 \
  --sp 4 --text_encoder_tp 4 --atten none \
  --image_adaln_table --image_resident --image_int8 \
  --image_condition_cache --image_refiner_sp --image_vision_sp \
  --image_vae_parallel --image_ff_chunk 2048 --image_rgb8_output --image_local_input
```

For multiple requests, call `load_pipeline(args)` once per rank and reuse the returned pipeline with `run_image(args, pipe)`. Startup flags belong to the resident worker configuration. Change prompts, reference paths and output sizes between requests, with the same startup flags. Request caches clear on both success and failure; cached media results never carry into the next request.

The AdaLN table is evaluated from this checkpoint's exact six-step one-row/two-row timestep layouts at initialization. It releases the large AdaLN projection weights. It is not the 21-/51-point video table. In the historical full-matrix configuration, W8A8 applies only to the main block attention/FFN linears; its text encoder, token refiner, AdaLN tables, VAE and PDD heads are not described as INT8. The selected NPU recipe adds the separately measured decoder text W8A8 option.

`--image_local_input` projects only each SP rank's owned image rows and scatters them into an in-place local packed buffer. Request-local refined text rows replace the larger intermediate condition caches. This avoids the full-width packed input allocation observed in the first matrix's largest combination. Output projection was already local to each SP rank before the low-dimensional PDD-head gather. The flag requires SP; it is separate from DiT TP.

## Exploration boundaries

* Pure SP2 (`--sp 2 --text_encoder_tp 2`) and DiT TP2 (`--sp 1 --image_dit_tp 2 --text_encoder_tp 2`) have been exercised on the fixed 2K edit. The full 40-case report uses SP4. TP shards the two SwiGLU projection halves separately and reduces row-parallel outputs. TP and fused QKV are separate experiments.
* Qwen visual SP reuses the v0.5 adapter. Image VAE tile parallelism separately distributes this image checkpoint's original encoder/ViT-decoder tiles and retains upstream stitching order. Audio/video streaming components are not loaded.
* RGB8 output transports already-quantized pixels as uint8 before D2H. It keeps PIL resize behavior and retains the original path for non-PIL/CPU output. The released truncation quantizer still runs before transport.
* Pure Laser produced nonfinite values in the last main block and black images. `--atten laser --image_laser_dense_tail 1` passed the fixed 2K edit and a six-step attention audit in the fused experimental recipe. This is a mixed backend, not validation of pure Laser or of all boundary inputs. The full matrix retains SDPA.
* Fused QKV, native SwiGLU and fused partial RoPE are opt-in experiments. Native SwiGLU requires swapping the static projection halves to its gate-first convention. Their measured E2E benefits on the small fixed input were small or inconsistent, so they are excluded from the full-matrix configuration.
* The Triton SwiGLU prototype reproduces eager BF16 rounding in its operator probe, but was substantially slower on the tested Ascend backend. It is not enabled in the selected recipe; its source is retained to reproduce the negative result.

## Validation and measurements

The fixed exploration input is one reference image with a local edit, 2048x2048, seed 42, six actual PDD steps. Source prompts/media are not distributed. Independent distributed CPU checks exercise full-transformer TP, bounded FFN, and tile ordering/blending against whole-image pooling/nearest-neighbor oracles. The original `tools/validate_h3_image_sp.py` checks uneven layouts and every PDD head.

```bash
torchrun --nproc_per_node=2 tools/validate_h3_image_optimizations.py
torchrun --nproc_per_node=4 tools/validate_h3_image_sp.py --device cpu --local-input
# On an explicitly selected idle NPU:
python tools/validate_h3_image_rgb8.py
```

The original 40-case archive and baseline report are immutable. The optimized runner uses identical prompt/JPEG bytes and warmup=0, repeat=1; it verifies six completed steps on all ranks, output sizes/hashes, preprocessing, pipeline/text storage identity, and NPU DiT storage pointers. Initialization is separate. Main time is the maximum synchronized pipeline span across ranks, excluding JPEG file opening and PNG saving. Per-case request/save time and allocated/reserved peaks are also retained.

Serial images per card-hour is `3600 / (cards * inference_seconds)`, not measured concurrent serving capacity. Driver HBM and process-group PSS use periodic samples and can miss short requests. Hardware placement differs from the old benchmark because one original device was occupied; the fresh control used the same new device placement. One/few samples do not establish P95 or statistical significance. See the [40-case performance and input-shape report](performance/h3-image-npu-40cases.md) for full-matrix outcomes and visual limitations.

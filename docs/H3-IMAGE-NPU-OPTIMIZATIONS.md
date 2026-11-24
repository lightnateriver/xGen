# MiniMax-H3-Image: v0.6 resident Ascend deployment

This is the six-step Image checkpoint, separate from H3 audio/video workflows. Keep the pinned Diffusers source from `requirements-h3-image.txt`. The original BF16 SP4/offload path remains available when the opt-in flags are absent.

## Measured configuration

The full matrix uses four 64-GiB Ascend 910B1 cards, SP4, Qwen text TP4, BF16 interfaces, dynamic W8A8 DiT attention/FFN linears, and the checkpoint's original mixed-precision VAE. It keeps the six PDD heads, FP32 patch input/timestep/output projections, CPU seed 42, and the original tile geometry/overlap blending.

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

The AdaLN table is evaluated from this checkpoint's exact six-step one-row/two-row timestep layouts at initialization. It releases the large AdaLN projection weights. It is not the 21-/51-point video table. W8A8 applies only to the main block attention/FFN linears; text encoder, token refiner, AdaLN tables, VAE and PDD heads are not described as INT8.

Use `--image_adaln_table_save /path/adaln.safetensors` to compute the table once and save it. Subsequent launches can use `--image_adaln_table_file /path/adaln.safetensors` in place of `--image_adaln_table`. The file is bound to the local checkpoint files and the six-step schedule. Regenerate it after replacing or moving the checkpoint. Diffusers still loads the original weights before installing the table.

Image and video tables share the Safetensors reader and atomic writer in `x_base.adaln_cache`. Their tensor layouts differ: Image keeps separate text/edit tables indexed by PDD step, whereas the video runtime uses modality rows and caches `norm_out` as well. Existing Image v1 table files remain readable.

For an offloaded component, `--image_pinned_cpu_offload` pins its CPU weights during startup so subsequent requests can reuse the pinned buffers. This increases pinned host memory use. A single NPU can combine this option for the text encoder with `--sp 1 --text_encoder_tp 1 --image_resident` for the DiT and a precomputed AdaLN table.

`--image_local_input` projects only each SP rank's owned image rows and scatters them into an in-place local packed buffer. Request-local refined text rows replace the larger intermediate condition caches. This avoids the full-width packed input allocation observed in the first matrix's largest combination. Output projection was already local to each SP rank before the low-dimensional PDD-head gather. The flag requires SP; it is separate from DiT TP.

For the experimental 910B3 BF16 comparison, start one process per NPU and set `--sp 1 --image_resident --image_adaln_table_file "$ADALN_FILE" --atten laser`. Set `INF_NAN_MODE_FORCE_DISABLE=1` in the environment. These settings do not establish full-matrix validation of pure Laser; the limitations below still apply. Use the corresponding options for each card count:

| NPUs | Additional xGen options |
| ---: | --- |
| 1 | `--text_encoder_tp 1 --image_pinned_cpu_offload --image_vae_decode_batch_size 4` |
| 2 | `--image_dit_tp 2 --text_encoder_tp 2 --image_vae_parallel --image_vae_decode_batch_size 4 --image_vae_gather_once` |
| 4 | `--image_dit_tp 4 --text_encoder_tp 4 --image_vae_parallel --image_vae_decode_batch_size 4 --image_vae_gather_once` |

The HTTP benchmark reuses the loaded pipeline and uses a no-compression PNG encoder in its benchmark adapter. The CLI saves through Pillow, so its total time includes a different PNG encoding path.

## VAE tile experiments

`--image_vae_decode_batch_size 4` decodes up to four equal-sized tiles per rank per call. It also works on a single NPU. Multi-device VAE tile parallelism gathers one tile per rank at a time by default, regardless of the decoder batch size. Add `--image_vae_gather_once` with `--image_vae_parallel` on multiple NPUs to gather all tiles in one collective. Both strategies restore the original spatial order and retain the original tile geometry. Batched decode can select different kernels and change individual pixel values, so run the image comparisons for your checkpoint and workloads before enabling it by default.

## Exploration boundaries

* Pure SP2 (`--sp 2 --text_encoder_tp 2`) and DiT TP2 (`--sp 1 --image_dit_tp 2 --text_encoder_tp 2`) have been exercised on the fixed 2K edit. The full 40-case report uses SP4. TP shards the two SwiGLU projection halves separately and reduces row-parallel outputs. TP now supports both separate Q/K/V projections and packed fused QKV. The fused-QKV/TP combination remains opt-in and requires dedicated numerical and device validation.
* Qwen visual SP reuses the v0.5 adapter. Image VAE tile parallelism separately distributes this image checkpoint's original encoder/ViT-decoder tiles and retains upstream stitching order. Audio/video streaming components are not loaded.
* RGB8 output transports already-quantized pixels as uint8 before D2H. It keeps PIL resize behavior and retains the original path for non-PIL/CPU output. The released truncation quantizer still runs before transport.
* Pure Laser produced nonfinite values in the last main block and black images. `--atten laser --image_laser_dense_tail 1` passed the fixed 2K edit and a six-step attention audit in the fused experimental recipe. This is a mixed backend, not validation of pure Laser or of all boundary inputs. The full matrix retains SDPA.
* Fused QKV, native SwiGLU and fused partial RoPE are opt-in experiments. Native SwiGLU requires swapping the static projection halves to its gate-first convention. Their measured E2E benefits on the small fixed input were small or inconsistent, so they are excluded from the full-matrix configuration.
* The Triton SwiGLU prototype reproduces eager BF16 rounding in its operator probe, but was substantially slower on the tested Ascend backend. It is not enabled in the selected recipe; its source is retained to reproduce the negative result.

`framework/vae/minimax_h3_image.py` owns the image VAE tile path. `framework/pipeline/minimax_h3_image.py` assembles the components and installs RGB8 output on the pipeline image processor. The model-specific AdaLN application and DiT optimizations live in `framework/transformer/h3_image_optimizations.py`.

## Validation and measurements

The fixed exploration input is one reference image with a local edit, 2048x2048, seed 42, six actual PDD steps. Source prompts/media are not distributed. The retained `tools/validate_h3_image_sp.py` checks uneven layouts and every PDD head.

```bash
torchrun --nproc_per_node=4 tools/validate_h3_image_sp.py --device cpu --local-input
```

The original 40-case archive and baseline report are immutable. The optimized runner uses identical prompt/JPEG bytes and warmup=0, repeat=1; it verifies six completed steps on all ranks, output sizes/hashes, preprocessing, pipeline/text storage identity, and NPU DiT storage pointers. Initialization is separate. Main time is the maximum synchronized pipeline span across ranks, excluding JPEG file opening and PNG saving. Per-case request/save time and allocated/reserved peaks are also retained.

Serial images per card-hour is `3600 / (cards * inference_seconds)`, not measured concurrent serving capacity. Driver HBM and process-group PSS use periodic samples and can miss short requests. Hardware placement differs from the old benchmark because one original device was occupied; the fresh control used the same new device placement. One/few samples do not establish P95 or statistical significance. See the [40-case performance and input-shape report](performance/h3-image-npu-40cases.md) for full-matrix outcomes and visual limitations.

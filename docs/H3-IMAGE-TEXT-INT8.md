# Experimental single-NPU H3 Image text W8A8

The released v0.6 text encoder uses BF16 weights: TP splits the decoder projections but does not quantize them. This opt-in experiment adds `--image_text_int8` and `--image_text_resident` for one Ascend NPU with text TP1. Defaults remain disabled.

`image_text_int8` dynamically quantizes the retained 50 Qwen decoder layers' attention q/k/v/o and MLP gate/up/down projections: 350 Linear modules. Embeddings, RMSNorm, vision tower and lm_head retain their original precision. Linear activations/weights use W8A8 and outputs remain BF16; attention stays dense SDPA. The existing xGen NPU dynamic-quant/matmul operator is reused. Quantization occurs before offloading hooks; TP linear communication is not replaced, and multi-rank use of these flags is rejected.

Use the [Image environment](H3-IMAGE.md#环境与权重) and [resident DiT options](H3-IMAGE-NPU-OPTIMIZATIONS.md), then select:

```bash
torchrun --nproc_per_node=1 example/diffusers/infer_minimax_h3_image.py \
  --pretrained_model_name_or_path "$MODEL_ROOT" \
  --prompt 'A product photograph of a green ceramic bowl on a neutral background.' \
  --save_path outputs/h3-image/text-int8.png --width 4096 --height 4096 \
  --sp 1 --text_encoder_tp 1 --atten none --seed 42 --num_inference_steps 6 \
  --image_adaln_table --image_resident --image_int8 \
  --image_condition_cache --image_ff_chunk 2048 --image_rgb8_output \
  --image_text_int8 --image_text_resident
```

Do not enable the SP-only local-input option on SP1. Multiple prompts can reuse the pipeline loaded once. Request-local condition caches are cleared as before. Original shared checkpoints are not modified; weights are quantized at initialization.

## Measured scope

One frozen 89-token T2I input, no reference images, 4096x4096, batch1, CPU seed42, six PDD steps; first and subsequent request each once on one 64-GiB Ascend 910B1. Both variants keep the main DiT W8A8 resident. These are sequential configuration comparisons, not a statistical benchmark.

| Text deployment | First inference s | Reused inference s | Peak allocated GiB | Peak reserved GiB | Initialization s |
|---|---:|---:|---:|---:|---:|
| BF16, leaf-level CPU/NPU offload | 62.603 | 60.969 | 23.637 | 27.254 | 25.616 |
| Decoder W8A8, whole text encoder resident | 53.223 | 52.098 | 50.548 | 52.898 | 45.436 |

Text parameter storage falls from 49.418 to 26.726 GiB (45.92%). Reused pipeline latency falls 14.55%; the comparison changes both precision and offload/residency, so it does not isolate quant-matmul speed. Peak NPU usage increases because text weights now stay on the device. Online quantization makes initialization longer; a one-off cold request is not necessarily faster. Main inference includes text, DiT and VAE, excluding initialization and PNG saving.

Both requests complete all six steps with one visible NPU and unchanged pipeline/representative DiT/text storage. Outputs decode as 4096x4096 and pass file/pixel hash checks. Visual review found normal content and lighting, with changed object geometry/reflections versus BF16. This is not pixel parity, and long prompts, reference-image editing and other dimensions have not been validated. Only the combined INT8-plus-resident text configuration was measured; INT8 with text offload is unmeasured.

Omit both flags for the released v0.6 behavior. The text flags are an experiment on a separate branch, not part of the existing v0.6 published defaults.

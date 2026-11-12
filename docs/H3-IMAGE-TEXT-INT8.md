# Experimental H3 Image text W8A8

The released v0.6 text encoder uses BF16 weights: TP splits the decoder projections but does not quantize them. This opt-in experiment adds `--image_text_int8` and `--image_text_resident` for Ascend NPU inference with text TP equal to the distributed world size. Defaults remain disabled. Single-NPU and TP4 checkpoint runs are documented below; TP2 is admitted by validation but has not been benchmarked with text INT8.

The selected NPU deployment retains this decoder W8A8 option. `example/diffusers/scripts/run_h3_image_npu.sh` enables it together with the resident SP4/text-TP4 recipe; generic CLI defaults remain unchanged. The choice favors lower NPU memory usage, without a claim of stable acceleration on the measured short prompt. See the [selected NPU deployment](H3-IMAGE-NPU-OPTIMIZATIONS.md#selected-npu-deployment).

`image_text_int8` dynamically quantizes the retained 50 Qwen decoder layers' attention q/k/v/o and MLP gate/up/down projections: 350 modules. Embeddings, RMSNorm, vision tower and lm_head retain their original precision. Linear activations/weights use W8A8 and outputs remain BF16; attention stays dense SDPA. The existing xGen NPU dynamic-quant/matmul operator is reused. Optional TP sharding precedes quantization of each owned weight shard. The TP modules retain their communication group and gather/reduce behavior; row-parallel bias is added once after reduction. Quantization precedes offload hooks. With TP greater than one the text encoder is already resident, so the INT8 flag alone isolates the precision change.

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

## Single-NPU measured scope

One frozen 89-token T2I input, no reference images, 4096x4096, batch1, CPU seed42, six PDD steps; first and subsequent request each once on one 64-GiB Ascend 910B1. Both variants keep the main DiT W8A8 resident. These are sequential configuration comparisons, not a statistical benchmark.

| Text deployment | First inference s | Reused inference s | Peak allocated GiB | Peak reserved GiB | Initialization s |
|---|---:|---:|---:|---:|---:|
| BF16, leaf-level CPU/NPU offload | 62.603 | 60.969 | 23.637 | 27.254 | 25.616 |
| Decoder W8A8, whole text encoder resident | 53.223 | 52.098 | 50.548 | 52.898 | 45.436 |

Text parameter storage falls from 49.418 to 26.726 GiB (45.92%). Reused pipeline latency falls 14.55%; the comparison changes both precision and offload/residency, so it does not isolate quant-matmul speed. Peak NPU usage increases because text weights now stay on the device. Online quantization makes initialization longer; a one-off cold request is not necessarily faster. Main inference includes text, DiT and VAE, excluding initialization and PNG saving.

Both requests complete all six steps with one visible NPU and unchanged pipeline/representative DiT/text storage. Outputs decode as 4096x4096 and pass file/pixel hash checks. Visual review found normal content and lighting, with changed object geometry/reflections versus BF16. This is not pixel parity, and long prompts, reference-image editing and other dimensions have not been validated. Only the combined INT8-plus-resident text configuration was measured; INT8 with text offload is unmeasured.

Omit both flags for the released v0.6 behavior. The text flags are an experiment on a separate branch, not part of the existing v0.6 published defaults.

## Four-NPU TP4 measured scope

A fresh matched comparison uses the same frozen 89-token T2I input at 4096x4096, batch1, seed42 and six PDD steps on four 64-GiB Ascend 910B1 NPUs. Each configuration loads once and executes one first request plus three reused requests. Both text encoders use TP4 and remain resident; only `image_text_int8` changes. Main DiT W8A8, SP4, refiner/vision SP, VAE tile parallelism, AdaLN tables, condition/RoPE cache, FFN chunk2048, RGB8 and local input packing are identical. Vision SP has no reference-image workload in this case. Dense SDPA is used; Laser is disabled.

| Metric | Text BF16 TP4 | Text W8A8 TP4 |
|---|---:|---:|
| First inference s | 16.265 | 16.438 |
| Reused inference mean s (3 requests) | 14.752 | 14.617 |
| Reused inference range s | 14.559–14.853 | 14.600–14.644 |
| Text model device span mean s (3 requests) | 0.1482 | 0.1969 |
| Peak allocated GiB / NPU | 37.254 | 31.759 |
| Peak reserved GiB / NPU | 38.902 | 33.018 |
| Text parameter storage GiB / NPU | 15.360 | 9.689 |
| Initialization s | 33.593 | 35.076 |

Inference excludes initialization and PNG saving. Text model time is the maximum across ranks of NPU events around the Qwen model, including stream waits/communication; it is not a bare kernel benchmark. Overall reused mean is 0.135s (0.92%) lower, within the observed run-to-run variation, while the text span increases by about 0.049s. This short-prompt case demonstrates a memory saving of 5.495 GiB/NPU in peak allocated memory, without evidence of a stable latency improvement. Fixed BF16-then-W8A8 order and three reused samples do not establish statistical significance. Long prompts and reference-image editing remain unmeasured with TP4 text INT8.

Every rank verifies 350 quantized TP projection weights, NPU residency, one model load and stable representative weight pointers across requests. All eight outputs are 4096x4096 and all ranks complete six steps. Global DiT entry shapes are `[1,16384,96]` image tokens, `[1,89,5120]` text embeddings and `[16473,3]` position IDs. The independent CPU communication oracle checks 12 gather/reduce/input-sharding/bias cases per rank with exact local `F.linear`; checkpoint inference exercises the actual W8A8 kernels.

For TP4, use `torchrun --nproc_per_node=4`, `--sp 4 --text_encoder_tp 4`, the same resident DiT/SP4 options above, and `--image_text_int8`. The `--image_text_resident` flag is redundant for TP4. A CPU routing check can be run with:

```bash
PYTHONPATH="$PWD/x_base:$PWD/x_diffusers" \
  torchrun --nproc_per_node=4 --master_port=29867 tools/validate_h3_image_text_tp.py
```

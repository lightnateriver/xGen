---
name: xgen-h3
description: Deploy xGen MiniMax-H3 video or MiniMax-H3-Image. Use for selecting CUDA or Ascend NPU, video profiles, six-step image generation/reference editing, offline torchrun, or the xGen HTTP/MaaS service.
---

# Run xGen MiniMax-H3

Resolve the repository root from this skill's location. Run the documented commands from that root. Ask for the hardware, task, duration, output resolution, and offline or online mode only when the user has not supplied them and the answer changes the launch path.

## Pick the route

| Route | Supported launch examples |
|---|---|
| NPU MiniMax-H3-Image | v0.6 separate six-step image pipeline: four-card SP4/text TP4, optional W8A8 resident DiT, image AdaLN table, refiner/vision/VAE tile parallelism, FFN chunks, RGB8 and local input packing. Dense SDPA is the validated full-matrix backend; pure Laser produced black images. Use the separate pinned Image Diffusers environment. Read the Image optimization guide; video Streaming/AdaLN tables are not Image assets |
| CUDA 5090 default T2VA/I2VA/R2VA | `cuda` or `cuda-5090` Base 768P selects Sage-FP8 full-parallel + native-stream + BF16 AdaLN tables: SP8, FP8 Linear/text TP8, PHAA direct packing, SAP/local input, request-local conditioning reuse and component parallelism. All50 DiT blocks stay resident. Prepare workflow-specific stripped checkpoints and matched tables first. Scope:8 GPUs; technical decoding and human quality review remain separate. See the composable guide for disable flags and rollback |
| NPU offline Base | T2VA/I2VA/R2VA 768P defaults to Laser INT8 full-parallel Streaming: SP8/text TP8/PHAA2,20 actual steps with matched21-point AdaLN, resident DiT, local IO, request-local condition reuse, Token Refiner SP/BF16 and component parallelism. Native-stream owns decoder tile parallelism; do not pass the CUDA-only VAE SP flag. T2VA/I2VA FFN chunk0; R2VA chunk2048. Real validation scope is recorded in the NPU default guide |
| NPU offline Refiner | T2VA/I2VA/R2VA 2K uses its separate Base+Refiner launcher and assets, with5/10/15-second examples |
| CUDA offline 768P | Default fp8-bf16: SP8/text TP8/PHAA2/50 resident blocks, FP8 Linear and BF16 attention/communication. The 5s/20-step T2VA and I2VA samples passed user review. R2VA keeps the legacy route; longer durations require target-shape validation |
| CUDA offline recipe 2 | Explicit fp8-sage: same FP8 Linear/text TP8/resident blocks, Sage INT8 Q/K + FP8 PV, FP8 transport, output indexed affine fusion, expandable segments. T2VA/I2VA 5/10/15s at 768P/20 steps technically passed; human quality pending. R2VA/2K/four-card are not validated for this recipe |
| CUDA offline 2K | I2VA with packed NVFP4 FL2VA Base plus the CUDA-prepared Refiner. The experimental profile accepts 5, 10 and 15 seconds |
| Legacy HTTP Base | Integer durations from 5 to 15 seconds. NPU uses a T2VA/I2VA service or a separate R2VA service. CUDA Base defaults to fp8-bf16 for T2VA/I2VA; R2VA remains on the legacy route |
| Legacy HTTP 2K | One workflow per instance. NPU supports all three tasks. CUDA requires FL2VA. Accepted durations are 5, 10 and 15 seconds |
| CUDA MaaS | 768P, integer 5..15 seconds; separate FL2VA/T2VA and Ref2VA instances. Recommended JSON files select native-stream and require a trusted native video_vae directory. Input limits and adaptive aspect ratio follow the MaaS API document |

For offline use, start with the [one-command examples](../../../README.md#离线生成):

R2VA Base has an independent experimental `--h3_condition_overlap` flag shared by CUDA/NPU, default **off** even under full-parallel presets. For the shell launcher, append `-- --h3_condition_overlap` after reference files. For HTTP, set it only in startup arguments/config, then restart to change it. The reference VAE branch overlaps Qwen preparation; existing precision/offload remain, peak memory can increase. `--h3_condition_overlap_timeout` defaults to600seconds for condition waits, not DiT or total generation. See [scope and validation](../../../docs/H3-CONDITION-OVERLAP.md); do not claim NPU hardware acceptance from CUDA measurements.

```bash
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k
bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 15 768p inputs/ref01.png
```

Use the [online launch and request guide](../../../docs/H3-ONLINE.md) for service configuration and payload examples. Legacy HTTP uses `/generate`, `/status/<id>` and `completed`. MaaS uses `/v1/tasks`, `/v1/tasks/<id>` and `succeeded`. Both expose `/download/<id>`. For MaaS, warm the service with one real request before routing traffic through `/ready`.

## Assets and flags

- v0.5 CUDA default is Sage-FP8 full-parallel + native-stream + BF16 AdaLN tables. Prepare assets using [the table guide](../../../docs/H3-CUDA-ADALN.md), including physically stripped Transformer weights, separately for FL2VA and Ref2VA. Default layout is MODEL/xgen-adaln/{fl2va,ref2va}/{transformer,tables}; table names contain actual step counts. Missing assets fail clearly. `--h3_cuda_adaln_mode owner` disables tables; `--h3_cuda_preset none --h3_cuda_precision fp8-bf16` restores the old reference recipe. Never silently generate tables, reduce image size, retry failed requests or claim table mode is whole-pipeline bitwise identical.


- For RTX5090 R2VA use [the preferred composable profile](../../../docs/H3-CUDA-COMPOSABLE.md) and the original Ref2VA modular checkpoint. `cuda-5090` is a positional shell argument, not a Python flag. The raw CLI selects `--h3_cuda_preset sage-fp8-full-parallel`; online uses `example/diffusers/configs/h3_cuda5090_ref2va.example.json`. CUDA Base 768P now defaults to this preset for T2VA/FL2VA/Ref2VA on SP8 + text TP8; NPU retains its separate v0.5 recipe. TP/CP remains experimental and off in this SP8 recipe. Reference audio/video,9-image and full-load capacity need their own tests; image-only runs do not exercise Audio Encoder parallelism.
- Use the [model source table](../../../README.md#h3-权重下载). Use Base/FL2VA Transformer assets for T2VA and I2VA, and Ref2VA assets for R2VA. I2VA takes one first frame. R2VA uses the [reference media limits](../../../docs/H3-DEPLOYMENT.md#r2va-多媒体参考).
- On NPU, follow [Base/Refiner preparation](../../../docs/H3-DEPLOYMENT.md#4-模型与目录准备). The stripped Transformer and AdaLN table must match the workflow and schedule. Set `num_inference_steps` and AdaLN preparation `--steps` to the actual Base evaluations (for example, 50). An existing 51-point table matches 50 steps. Refiner adds three.
- On CUDA, explicitly select the historical [fp8-bf16 profile](../../../docs/H3-CUDA-PRECISION.md) with a complete original FL2VA modular checkpoint. No packed NVFP4 file is required for default Base/MaaS. Auto keeps NPU, other models and explicit legacy/Refiner workflows unchanged. `H3_CUDA_PRECISION=none` selects the legacy offline path; service JSON/CLI can set h3_cuda_precision=none. H3_FAST=1 is a separate historical experiment, not the manually accepted profile. Never set XGEN_PHAA_FP8=1 for fp8-bf16. Precision is fixed at service startup, not a request option.
- CUDA standard recipe 2 is [Sage-FP8](../../../docs/H3-CUDA-SAGE.md). Run `H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p`; choose i2va and a first frame for I2VA, and 5/10/15 for duration. The launcher enables FP8 transport, output AdaLN fusion and expandable segments after clearing legacy flags. Direct torchrun and HTTP/MaaS require those environment variables before process startup. The standard recipe retains FP8 weights and is not INT8 Q/K communication. Sage HTTP/MaaS end-to-end has not been validated; never report offline timings as HTTP SLA or technical decode as human quality approval.
- Read [MaaS API](../../../server/video/MAAS-API.md) for bounds, queue errors, timeouts, first/last frames, reference media and delivery. MaaS currently exposes 768P only; use the documented FL2VA/T2VA or separate Ref2VA instance. Decoder selection is a startup option, not a generation-request field.

## Verify a run

The GPU Sage-FP8 full-parallel Streaming preset includes per-peer Align16 by default for synchronous PHAA FP8 transport, in both offline and HTTP entry points. Set `XGEN_PHAA_FP8_ALIGN16=0` consistently on all ranks and restart for independent rollback. Outside this preset the transport default remains off; explicit `1` opts into alignment for compatible CUDA synchronous FP8 recipes. NPU, BF16 transport and the independent asynchronous FP8 implementation do not use this layout. Use [the guide](../../../docs/H3-PHAA-FP8-ALIGNMENT.md) and [the 12-case online comparison](../../../docs/performance/h3-cuda-phaa-align16.md). Keep the same-node T2VA A/B separate from cross-node historical results, and retain human video/audio review as pending until reviewed.

CUDA recommended T2VA/I2VA/R2VA (`fp8-bf16`, `fp8-sage`, `sage-fp8-full-parallel`) now default to `native-stream` VAE decode. For run_h3.sh, set `H3_VAE_MODEL_PATH=weights/h3-native/video_vae` and `H3_VAE_TRUST_REMOTE_CODE=1` only for trusted native model code. Diffusers `vae` is not the native `video_vae` directory. CLI equivalents are `--h3_vae_model_path`, `--h3_vae_trust_remote_code`, and `--vae_matmul_quant none`. Explicit `--h3_vae_backend legacy|lightning|native-stream` overrides the recipe on CUDA and NPU. For script rollback, unset native path/trust variables and set `H3_VAE_BACKEND=legacy` or `lightning`. NPU Base also defaults to Laser INT8 full-parallel Streaming through its recommended launcher/config. Resident services require restart to change decoder. Historical full-pipeline performance and manual quality acceptance used legacy decode and do not automatically apply to native-stream. See [backend guide](../../../docs/H3-OMNI-VAE.md).

Run `python tools/collect_env.py` in the target environment. The launch examples use eight visible devices. On a shared node, check device memory, utilization and compute processes before reserving cards or starting work.

For an NPU offline performance matrix, use the [benchmark guide](../../../docs/H3-NPU-BENCHMARK.md). It covers 35 cases with resident model groups, optional per-case warmup, timestamped incremental reports and HBM/RSS/PSS sampling. Public step counts are actual Base evaluations. Read the [upgrade notes](../../../docs/H3-UPGRADE.md) before migrating old schedules. Each saved H3 video now triggers unused-memory reclamation; the historical stage-trim measurements do not validate that new policy. A completed technical media check does not imply visual or audio quality approval.

For an offline run, verify the MP4 exists and use FFprobe to check dimensions, frame rate and audio. Decode the complete video and audio with FFmpeg before claiming technical success. For a service, verify a real request reaches its success status and that the returned file downloads and decodes. Report visual quality separately from decode success. Report model initialization separately from warmed request latency, and keep stage timings separate from client E2E.

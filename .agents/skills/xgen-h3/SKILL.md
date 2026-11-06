---
name: xgen-h3
description: Deploy xGen MiniMax-H3 and generate audio-video from text, a first frame, or reference images. Use for selecting CUDA or Ascend NPU, 768P or 2K, 5/10/15-second profiles, offline torchrun, or the xGen HTTP/MaaS service.
---

# Run xGen MiniMax-H3

Resolve the repository root from this skill's location. Run the documented commands from that root. Ask for the hardware, task, duration, output resolution, and offline or online mode only when the user has not supplied them and the answer changes the launch path.

## Pick the route

| Route | Supported launch examples |
|---|---|
| NPU offline | T2VA, I2VA, R2VA at 768P or 2K, with 5, 10 or 15 seconds |
| CUDA offline 768P | Default fp8-bf16: SP8/text TP8/PHAA2/50 resident blocks, FP8 Linear and BF16 attention/communication. The 5s/20-step T2VA and I2VA samples passed user review. R2VA keeps the legacy route; longer durations require target-shape validation |
| CUDA offline recipe 2 | Explicit fp8-sage: same FP8 Linear/text TP8/resident blocks, Sage INT8 Q/K + FP8 PV, FP8 transport, output indexed affine fusion, expandable segments. T2VA/I2VA 5/10/15s at 768P/20 steps technically passed; human quality pending. R2VA/2K/four-card are not validated for this recipe |
| CUDA offline 2K | I2VA with packed NVFP4 FL2VA Base plus the CUDA-prepared Refiner. The experimental profile accepts 5, 10 and 15 seconds |
| Legacy HTTP Base | Integer durations from 5 to 15 seconds. NPU uses a T2VA/I2VA service or a separate R2VA service. CUDA Base defaults to fp8-bf16 for T2VA/I2VA; R2VA remains on the legacy route |
| Legacy HTTP 2K | One workflow per instance. NPU supports all three tasks. CUDA requires FL2VA. Accepted durations are 5, 10 and 15 seconds |
| CUDA MaaS | 5-second 768P landscape FL2VA with one inline first frame and 50 actual DiT evaluations |

For offline use, start with the [one-command examples](../../../README.md#离线生成):

```bash
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k
bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p
```

Use the [online launch and request guide](../../../docs/H3-ONLINE.md) for service configuration and payload examples. Legacy HTTP uses `/generate`, `/status/<id>` and `completed`. MaaS uses `/v1/tasks`, `/v1/tasks/<id>` and `succeeded`. Both expose `/download/<id>`. For MaaS, warm the service with one real request before routing traffic through `/ready`.

## Assets and flags

- Use the [model source table](../../../README.md#h3-权重下载). Use Base/FL2VA Transformer assets for T2VA and I2VA, and Ref2VA assets for R2VA. I2VA takes one first frame. R2VA uses the [reference media limits](../../../docs/H3-DEPLOYMENT.md#r2va-多媒体参考).
- On NPU, follow [Base/Refiner preparation](../../../docs/H3-DEPLOYMENT.md#4-模型与目录准备). The stripped Transformer and AdaLN table must match the workflow and schedule. Set `num_inference_steps` and AdaLN preparation `--steps` to the actual Base evaluations (for example, 50). An existing 51-point table matches 50 steps. Refiner adds three.
- On CUDA, use the default [fp8-bf16 profile](../../../docs/H3-CUDA-PRECISION.md) with a complete original FL2VA modular checkpoint. No packed NVFP4 file is required for default Base/MaaS. Auto keeps NPU, other models and explicit legacy/Refiner workflows unchanged. `H3_CUDA_PRECISION=none` selects the legacy offline path; service JSON/CLI can set h3_cuda_precision=none. H3_FAST=1 is a separate historical experiment, not the manually accepted profile. Never set XGEN_PHAA_FP8=1 for fp8-bf16. Precision is fixed at service startup, not a request option.
- CUDA standard recipe 2 is [Sage-FP8](../../../docs/H3-CUDA-SAGE.md). Run `H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p`; choose i2va and a first frame for I2VA, and 5/10/15 for duration. The launcher enables FP8 transport, output AdaLN fusion and expandable segments after clearing legacy flags. Direct torchrun and HTTP/MaaS require those environment variables before process startup. The standard recipe retains FP8 weights and is not INT8 Q/K communication. Sage HTTP/MaaS end-to-end has not been validated; never report offline timings as HTTP SLA or technical decode as human quality approval.
- Read [MaaS API](../../../server/video/MAAS-FL2VA-API.md) for bounds, queue errors, timeouts and delivery. The current MaaS input profile has no 2K, 10-second, 15-second, T2VA or R2VA route. Use legacy HTTP or offline as appropriate.

## Verify a run

Run `python tools/collect_env.py` in the target environment. The launch examples use eight visible devices. On a shared node, check device memory, utilization and compute processes before reserving cards or starting work.

For an NPU offline performance matrix, use the [benchmark guide](../../../docs/H3-NPU-BENCHMARK.md). It covers 35 cases with resident model groups, optional per-case warmup, timestamped incremental reports and HBM/RSS/PSS sampling. Public step counts are actual Base evaluations. Read the [upgrade notes](../../../docs/H3-UPGRADE.md) before migrating old schedules. Each saved H3 video now triggers unused-memory reclamation; the historical stage-trim measurements do not validate that new policy. A completed technical media check does not imply visual or audio quality approval.

For an offline run, verify the MP4 exists and use FFprobe to check dimensions, frame rate and audio. Decode the complete video and audio with FFmpeg before claiming technical success. For a service, verify a real request reaches its success status and that the returned file downloads and decodes. Report visual quality separately from decode success. Report model initialization separately from warmed request latency, and keep stage timings separate from client E2E.

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
| CUDA offline 768P | BF16 I2VA has an end-to-end result. BF16 T2VA and R2VA are candidates awaiting hardware and quality validation. Longer durations require a target-shape check |
| CUDA offline 2K | I2VA with packed NVFP4 FL2VA Base plus the CUDA-prepared Refiner. The experimental profile accepts 5, 10 and 15 seconds |
| Legacy HTTP Base | Integer durations from 5 to 15 seconds. NPU uses a T2VA/I2VA service or a separate R2VA service. CUDA BF16 T2VA and R2VA remain validation candidates |
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

- Use the [model source table](../../../README.md#h3-权重下载). Use Base/FL2VA Transformer assets for T2VA and I2VA, and Ref2VA assets for R2VA. I2VA takes one first frame. The launcher and 2K R2VA take five images.
- On NPU, follow [Base/Refiner preparation](../../../docs/H3-DEPLOYMENT.md#4-模型与目录准备). The stripped Transformer and AdaLN table must match the workflow and schedule. The 51 scheduler points produce 50 Base evaluations. Refiner adds three.
- On CUDA, read [CUDA deployment](../../../docs/H3-CUDA.md) for component layouts and packed weight requirements. `H3_FAST=1` on the offline launcher selects the measured 5-second 768P FL2VA profile. Use the BF16 route for a quality reference. Keep the documented quality and hardware limits attached to the result.
- Read [MaaS API](../../../server/video/MAAS-FL2VA-API.md) for bounds, queue errors, timeouts and delivery. The current MaaS input profile has no 2K, 10-second, 15-second, T2VA or R2VA route. Use legacy HTTP or offline as appropriate.

## Verify a run

Run `python tools/collect_env.py` in the target environment. The launch examples use eight visible devices. On a shared node, check device memory, utilization and compute processes before reserving cards or starting work.

For an offline run, verify the MP4 exists and use FFprobe to check dimensions, frame rate and audio. Decode the complete video and audio with FFmpeg before claiming technical success. For a service, verify a real request reaches its success status and that the returned file downloads and decodes. Report visual quality separately from decode success. Report model initialization separately from warmed request latency, and keep stage timings separate from client E2E.

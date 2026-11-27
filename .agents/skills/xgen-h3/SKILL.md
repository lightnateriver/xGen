---
name: xgen-h3
description: Deploy xGen MiniMax-H3 video or MiniMax-H3-Image. Use for selecting CUDA or Ascend NPU, video profiles, six-step image generation/reference editing, offline torchrun, or the xGen HTTP/MaaS service.
---

# Run xGen MiniMax-H3

Resolve the repository root from this skill's location and run commands there. Use the user's hardware, task, duration, resolution and offline/online choice; ask only for missing information that changes the route. Follow workspace AGENTS.md and existing authorization for devices, SSH, inputs and test cases. Keep credentials and machine-specific connection details out of source and logs.

## Select the route

| Route | Documentation |
|---|---|
| Video: T2VA, first/last frames, Ref2VA | [Video guide](../../../docs/H3-VIDEO-QUICKSTART.md) |
| CUDA Base | [CUDA dependencies, defaults and rollback](../../../docs/H3-CUDA.md) |
| NPU Base | [VC/fusion installation and defaults](../../../docs/H3-NPU-DEFAULT.md) |
| Six-step Image | [Separate Image environment and entry](../../../docs/H3-IMAGE.md) |
| Base → 2K Refiner | [Workflow and backend requirements](../../../docs/H3-VIDEO-QUICKSTART.md#refiner) |
| HTTP or CUDA MaaS | [Service guide](../../../docs/H3-ONLINE.md) and [API](../../../server/video/MAAS-API.md) |

For Video, recommended examples explicitly use `H3_BASE_STEPS=20` and matched assets; Image stays at six steps and 2K Refiner adds three. Low-level CUDA/2K script defaults can still be 50, so do not infer the step count from the route name.

## Prepare and launch

- Use [model preparation](../../../docs/H3-MODELS.md). T2VA/I2VA use FL2VA assets; Ref2VA uses separate weights/tables. Keep stripped checkpoints with their matching tables and preserve original weights for owner rollback. Never silently regenerate assets.
- Use a separate pinned Diffusers Image environment. Video AdaLN tables and streaming VAE are not Image assets. NPU Image text defaults to W8A8; CUDA text defaults to BF16.
- Run `python tools/collect_env.py` in the target runtime. Check device visibility, memory, utilization and processes before using the allocated cards. Launch examples have explicit multi-rank requirements; do not claim a topology works on another card count without checking it.
- The CUDA Base default is Sage-FP8 full-parallel/native-stream/table. NPU Base requests VC and Ascend fusions with INT8/full-parallel/native-stream. Install optional dependencies first and inspect actual hit/fallback logs.
- Native-stream requires a trusted native `video_vae` directory and explicit authorization. A Diffusers `vae` directory is not a substitute. Restart a resident service when changing precision, decoder, layout, fixed model, LoRA or table mode.
- [Configuration rules](../../../docs/H3-CONFIGURATION.md) own the shared switches and limitations. TP/CP, condition overlap, optional compilation, sparse attention and Turbo are explicit opt-ins. Do not automatically retry, resize inputs, reduce resolution, change precision, add offload or silently fall back to a different recipe after a failure.
- Ordinary HTTP and MaaS V2 have different status/error contracts. Current MaaS startup is CUDA-only; IR/2K mock output is not the real Refiner. Warm with a real request before relying on readiness.

## Verify the result

Keep prompt bytes, reference order, source video audio settings, seed, weights, actual steps and effective flags in the run record. Use the user's cases or the workspace defaults. Feature support and successful local functions do not establish device capacity or content quality.

For offline output, check the file, dimensions, frame rate, audio and complete FFmpeg decoding. For a service, verify the protocol's success status and that the published file downloads and decodes. Human image/video/audio review remains separate from technical success.

Report initialization, warmed generation, publication and client wait separately when measured. Keep errors and missing coverage visible. The repository's lightweight checks and their limits are documented under [architecture](../../../docs/ARCHITECTURE.md#validation); dedicated benchmark scripts and historical qualification records are no longer maintained here.

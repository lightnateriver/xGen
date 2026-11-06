# CUDA H3 Sage precision experiment

This optional offline recipe retains the original fp8-bf16 default. Select
`--h3_cuda_precision fp8-sage` explicitly. It reuses the same FP8 Linear,
FP8 text TP8, 50 resident DiT blocks, exact AdaLN cache, VAE SP8 tiles,
decoder staging and request-end reclamation. It installs Sage through the
existing xGen attention manager and model-local H3 attention callback.
No new scheduler, offloader or mandatory dependency is added. SageAttention
must already be installed in the CUDA environment.

On RTX 5090, the installed Sage dispatch uses INT8 Q/K and FP8 P×V.
These are attention activation formats, not model weight formats. The
recipe keeps the existing FP8 weights; it does not require NVFP4 weights.

`XGEN_PHAA_FP8=1` independently selects xGen's existing FP8 bidirectional
PHAA transport. All Q/K/V payloads are FP8 with row scales, reconstructed
as BF16 before attention; Sage then performs its own INT8 Q/K quantization.
This is **not INT8 Q/K All-to-All**. Quantized communication and Sage are
both approximate; visual and audio review is required for this combination.
Masked or otherwise unsupported Sage calls retain xGen's existing SDPA
fallback. Runtime `calls` versus `kernel_calls` distinguish that fallback.

Use the same complete FL2VA checkpoint and prompt files as the baseline:

```bash
XGEN_H3_FUSED_OUTPUT_ADALN=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
XGEN_PHAA_FP8=1 H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 \
  bash example/diffusers/scripts/run_h3.sh cuda t2va 15 768p
```

For I2VA use `cuda i2va 15 768p inputs/first-frame.png`; 5 and 10 seconds
use the corresponding duration and prompt. Eight CUDA devices are required.
This experiment has not been enabled as a new HTTP/MaaS preset.

To roll back, select `H3_CUDA_PRECISION=fp8-bf16`, unset `XGEN_PHAA_FP8`
and restart. Output AdaLN fusion and expandable segments remain independent
options. NPU and other models do not install this CUDA profile.

Transport compression does not halve resident model memory or guarantee a
smaller overall peak: receive tensors remain BF16 and quantization uses
temporary storage. Actual long-video peaks and performance must be measured.

## Measured 20-step run (2026-11-06)

Eight RTX 5090 devices, SP8, text TP8, PHAA2, 1344×768 at 24 FPS,
seed 42, all 50 DiT blocks resident through decode. The six cases reuse one
worker group. Each case has one complete 1-step warmup before its 20-step
measurement. The runtime source was frozen at `d428a9b`; later documentation
edits do not change this measured implementation.

| Task | Duration | Generate + save | DiT | Peak allocated / reserved per rank |
|---|---|---:|---:|---:|
| T2VA | 5s | 42.935 s | 34.518 s | 28.28 / 28.49 GiB |
| I2VA | 5s | 48.089 s | 29.741 s | 28.29 / 28.52 GiB |
| T2VA | 10s | 76.226 s | 63.565 s | 28.32 / 30.14 GiB |
| I2VA | 10s | 97.808 s | 78.950 s | 28.33 / 29.76 GiB |
| T2VA | 15s | 148.148 s | 130.677 s | 29.88 / 30.40 GiB |
| I2VA | 15s | 161.120 s | 138.185 s | 30.00 / 30.41 GiB |

Times and peaks use the maximum across eight ranks; model initialization
and CPU media verification are excluded from request latency. Stage event
envelopes may overlap and should not be added together. All six MP4s passed
complete video/audio decoding; all ranks completed 20 DiT evaluations and
used the Sage kernel for all 2,000 model-local attention callbacks per
measurement. Visual, lip-sync and listening acceptance remain pending.

The matching 15-second FP8 Linear / BF16 attention and transport configuration,
also using output AdaLN fusion and expandable segments, measured 293.385 s
for T2VA and 308.161 s for I2VA, with peaks of 29.88 and 30.00 GiB allocated.
Those formal requests followed an earlier separate-process 1-step check,
so first-touch behavior is not strictly matched. These are whole-recipe,
single-run references, not an isolated attention ablation or statistical mean.
The 15-second peak allocated memory was essentially unchanged by Sage.

The existing saver uses FFmpeg `-shortest`: requested generation frames are
124 / 243 / 362; the observed MP4 frame counts are 124 / 243 / 361. The
15-second audio stream is 15.075 seconds and its video is 15.041667 seconds.
No frames were added to alter these observed results.

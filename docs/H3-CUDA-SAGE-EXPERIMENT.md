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

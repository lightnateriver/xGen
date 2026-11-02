# Experimental H3 Base attention

v0.3 adds model-local Spark-H3 and Veda operators to the existing XGEN H3
processor and PHAA callback. `--h3_attention legacy` is the default. It keeps
the v0.2 attention selection, models, scheduler, offload, cache, VAE and codecs.
This feature is experimental; implementation is not a claim of validation.

Initial qualification: T2VA, batch one, 1344x768 Base, 124 native frames,
24fps, SP8, BF16 Q/K/V, 50 actual DiT evaluations (`num_inference_steps=51`).
The existing optional Refiner exports 2560x1440/120 frames with three Laser
steps. Only Base attention changes. I2VA/R2VA, other frame counts, arbitrary
masks, extra LoRAs and a sparse Refiner are rejected or unsupported here.
Veda was released for an eight-NFE recipe: quality on the original 50-step
Base must be measured; a faster kernel alone is not a quality pass.

## Source and optional compilation

Use the Apache-2.0 snapshot preceding the project's noncommercial license
change. Retain its third-party notices and CANN/Miowtion/Spark licenses.
No vLLM or vLLM-Omni runtime is installed or imported by this integration.
The existing Torch/TorchNPU/CANN installation must also provide compatible
Ascend Triton and CMake to compile the optional kernels. Main requirements
and default imports do not change.

```bash
git clone https://github.com/Huage001/Ascend-H3.git optional/Ascend-H3
git -C optional/Ascend-H3 checkout e02a57325029e67eafaf67d64b3d62a330f1e8d4
export PYTHONPATH="$PWD/optional/Ascend-H3/src:$PWD/x_diffusers:$PWD/x_base${PYTHONPATH:+:$PYTHONPATH}"
export H3_ASCEND_CACHE="$PWD/model-cache/h3-attention-native"
# Load the deployment environment's CANN setup first. Match the actual SoC.
python -c "from h3_ascend.build import build; build('spark',soc='Ascend910B1'); build('veda',soc='Ascend910B1')"
```

Build products stay in the external cache; they depend on Torch, TorchNPU,
CANN, ABI and SoC. Never reuse unverified binaries across environments.

## Parameters and examples

| Parameter | Default | Purpose |
|---|---|---|
| `--h3_attention` | `legacy` | `legacy`, `spark`, or `veda`; Base only |
| `--h3_veda_predictor` | unset | Local published predictor file; required only by Veda |

Use the existing 2K launcher and asset preparation in H3-DEPLOYMENT.md:

```bash
bash example/diffusers/scripts/infer_minimax_h3_refiner.sh --h3_attention spark
bash example/diffusers/scripts/infer_minimax_h3_refiner.sh \
  --h3_attention veda --h3_veda_predictor weights/veda/predictor.safetensors
```

The launcher requires the existing BASE_MODEL, BASE_TRANSFORMER, BASE_ADALN,
REFINER_WEIGHTS and REFINER_PREPARED environment variables. Do not replace
the 51-point AdaLN table or automatically enable an eight-step LoRA.
For 768P use the original CLI with the same Base configuration, width=1344,
height=768 and frames=124, without `--h3_refiner`, and add the desired backend.
Default H3 export may retain 124 native frames; the benchmark must state its
crop/encode accounting instead of hiding a 120-vs-124-frame difference.

Veda predictor source:
`Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview`, revision
`76f202874608115408d73280be9531c4ab888242`, file
`minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`.
FP8 file size is not the predictor's runtime memory footprint. The decoded
predictor has immutable CPU masters, is shared across layers/head chunks,
loads to NPU for Base, and offloads before Refiner/VAE. No condition cache or
request retry is added. Routing is recomputed for every Q/K call.

## Observability, recovery and validation

Each request writes `<output>.attention.rankN.json`: backend, actual Base
steps, per-rank global head IDs, predictor load count, and operator/pack/unpack
spans. Operator spans include prediction and routing; do not call them pure
attention-kernel time. First-use program/JIT/graph setup is separately marked
and must not be mistaken for steady-state latency. No silent fallback occurs.
After a request error or backend/predictor change, restart the worker. To roll
back, restart without the new flags or with `--h3_attention legacy`.

Qualification requires frozen inputs and separate numerical and performance
runs: compare the adapter against pinned upstream math on identical Q/K/V
and global head shards (including PHAA's 4+3 split), measure approximation
against dense attention separately, then generate paired Laser/Spark/Veda
5s videos at 768P/2K. Keep initialization, request latency, stage spans,
allocated/reserved HBM, driver HBM, PSS and release snapshots separate.
Full decoding and same-time screenshots are required; screenshots alone do
not establish temporal/audio quality. Repeated Laser controls establish the
existing nondeterminism floor before attributing output differences.

Validated numerical tolerances, hardware scope, memory increments, timing
tables, screenshots and failures are recorded only after actual execution.

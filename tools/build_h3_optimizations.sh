#!/usr/bin/env bash
# Run inside the existing Linux aarch64 CANN 9.1 / torch 2.10 / torch_npu environment.
# Usage: bash tools/build_h3_optimizations.sh
# Override PYTHON and BUILD_JOBS if required. No device inference is launched here.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELS="$ROOT/x_base/x_base/operator/npu_fusion"
PYTHON="${PYTHON:-python}"
BUILD_JOBS="${BUILD_JOBS:-4}"
: "${ASCEND_HOME_PATH:?Source the installed CANN set_env.sh first}"

build() {
    local project="$1"
    shift
    cmake -S "$KERNELS/$project" -B "$KERNELS/$project/build" \
        -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v "$PYTHON")"
    cmake --build "$KERNELS/$project/build" --parallel "$BUILD_JOBS" --target "$@"
}

build f1b h3_swiglu_quant_reuse
build f2a h3_gate_residual
build f2_nq2 h3_norm_affine_quant2_cached
build f3 h3_partial_rope
build f4 h3_phaa_layout
build f4_preprocess h3_phaa_preprocess
build f4_merge_quant h3_merge_quant

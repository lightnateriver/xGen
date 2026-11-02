#!/bin/bash
# link_x_comfyui.sh — deploy x_comfyui into ComfyUI
#
# Usage:
#   ./link_x_comfyui.sh ./ComfyUI
#
# What it does:
#   1. pip install -e x_base/  (if not installed)
#   2. pip install -e x_comfyui/  (if not installed)
#   3. ln -s x_comfyui/comfyui_entry  ComfyUI/custom_nodes/x_npu_turbo
#   4. cp x_comfyui/main_multi.py  ComfyUI/main_multi.py  (multi-card entry)
#
# After running, restart ComfyUI. NPUTurbo node appears in "NPU" category.
# For multi-card SP/FSDP, use:  torchrun --nproc_per_node=N main_multi.py

set -euo pipefail

# --- Resolve paths ---
# SCRIPT_DIR is x_gen/x_comfyui/ — x_base is a sibling: x_gen/x_base/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
X_GEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
X_BASE_DIR="${X_GEN_DIR}/x_base"
X_COMFYUI_DIR="${SCRIPT_DIR}"
ENTRY_DIR="${X_COMFYUI_DIR}/comfyui_entry"

COMFYUI_DIR="${1:-}"
if [[ -z "$COMFYUI_DIR" ]]; then
    echo "Usage: $0 ./ComfyUI"
    echo ""
    echo "Example:"
    echo "  $0 ./ComfyUI"
    echo "  $0 ./ComfyUI"
    exit 1
fi

CUSTOM_NODES="${COMFYUI_DIR}/custom_nodes"
LINK_NAME="${CUSTOM_NODES}/x_npu_turbo"

echo "=== x_comfyui installer ==="
echo "  x_gen root:   ${X_GEN_DIR}"
echo "  ComfyUI:      ${COMFYUI_DIR}"
echo ""

# --- Step 1: pip install x_base ---
# Use `pip show` instead of `import x_base` to avoid triggering NPU init
if pip show x_base &>/dev/null; then
    echo "[1/4] x_base already installed, skip"
else
    echo "[1/4] Installing x_base..."
    pip install -e "${X_BASE_DIR}"
fi

# --- Step 2: pip install x_comfyui ---
if pip show x_comfyui &>/dev/null; then
    echo "[2/4] x_comfyui already installed, skip"
else
    echo "[2/4] Installing x_comfyui..."
    pip install -e "${X_COMFYUI_DIR}"
fi

# --- Step 3: symlink into custom_nodes ---
if [[ -L "$LINK_NAME" || -e "$LINK_NAME" ]]; then
    echo "[3/4] Symlink already exists at ${LINK_NAME}, removing old..."
    rm -f "$LINK_NAME"
fi

mkdir -p "$CUSTOM_NODES"
ln -s "$ENTRY_DIR" "$LINK_NAME"
echo "[3/4] Symlinked: ${LINK_NAME} -> ${ENTRY_DIR}"

# --- Step 4: deploy main_multi.py (multi-card entry point) ---
MAIN_MULTI_SRC="${X_COMFYUI_DIR}/main_multi.py"
MAIN_MULTI_DST="${COMFYUI_DIR}/main_multi.py"
if [[ -f "$MAIN_MULTI_DST" ]] && ! diff -q "$MAIN_MULTI_SRC" "$MAIN_MULTI_DST" &>/dev/null; then
    echo "[4/4] main_multi.py exists but differs, updating..."
    cp "$MAIN_MULTI_SRC" "$MAIN_MULTI_DST"
elif [[ -f "$MAIN_MULTI_DST" ]]; then
    echo "[4/4] main_multi.py already up-to-date, skip"
else
    echo "[4/4] Deploying main_multi.py..."
    cp "$MAIN_MULTI_SRC" "$MAIN_MULTI_DST"
fi
echo "[4/4] Multi-card entry: ${MAIN_MULTI_DST}"

echo ""
echo "=== Done ==="
echo "Single card:  python main.py"
echo "Multi card:   X_SP_SIZE=N torchrun --nproc_per_node=N main_multi.py"
echo ""
echo "Restart ComfyUI, then add node: NPU → NPU Turbo"

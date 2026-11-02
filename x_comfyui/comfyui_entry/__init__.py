"""ComfyUI custom-node entry point for x_comfyui.

This is a *thin bridge* — it just re-exports the node registrations from
the pip-installed ``x_comfyui`` package.  All logic lives in the package.

Usage:
    Symlink this directory into ComfyUI's custom_nodes/:

        ln -s ./x_gen/x_comfyui/comfyui_entry \\
              ./ComfyUI/custom_nodes/x_npu_turbo

    Then restart ComfyUI — the NPUTurbo node will appear in the "NPU" category.
"""

import logging

logger = logging.getLogger("x_comfyui")

try:
    from x_comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:
    logger.error("Failed to import x_comfyui: %s", e)
    logger.error("Make sure x_base and x_comfyui are installed:")
    logger.error("  pip install -e x_base/ && pip install -e x_comfyui/")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Adapted from the pinned Ascend-H3 Veda wrapper, Apache-2.0.
# See NOTICE.md. Adds shared_bundle; original layer plans remain unchanged.
"""Dynamic per-head Veda routes; rank-local data uses global predictor indices."""

import json
import os

from h3_ascend.native import load
from h3_ascend.paths import source_root


class VedaAttention:
    def __init__(self, layout, heads, predictor_path=None, sparsity=0.9, shared_bundle=None, **kwargs):
        from .veda_vendor import NpuVeda

        predictor_path = predictor_path or os.environ.get("H3_ASCEND_VEDA_PREDICTOR")
        if not predictor_path:
            raise ValueError("set H3_ASCEND_VEDA_PREDICTOR to the published predictor.safetensors")
        rows = json.loads((source_root() / "configs/veda.json").read_text())
        width = layout.grid[2] * 32
        row = next(r for r in rows if (r["width"], r["frames"]) == (width, layout.frames))
        self.configs = [
            dict(row.get("miowtion_per_layer", {}).get(str(i), row["miowtion_config"]))
            for i in range(50)
        ]
        self.impl = NpuVeda(
            layout.grid,
            layout.prefix,
            predictor_path,
            shared_bundle=shared_bundle,
            device=kwargs.get("device", "npu:0"),
            head_chunk=heads.count,
            fused=True,
            stack_tiles=4,
            kernel_kind="schedule2",
            pack_channels=64,
            scatter_rows=64,
            route_kind="fused",
            pack_mode="split",
            reuse_plan=True,
            head_start=heads.start,
            num_heads=heads.count,
            sparsity=sparsity,
        )
        # Chunk size is part of the measured per-layer launch configuration.
        # Build its rank-local tensor views once, outside measured forwards.
        for layer, config in enumerate(self.configs):
            chunk = min(heads.count, config["head_chunk"])
            self.impl.groups[layer] = [
                (key, piece)
                for key, group in self.impl.groups[layer]
                for piece in group.split(chunk)
            ]
        # Adopt only the layer/geometry combinations confirmed faster in the
        # prior comparison. Other layers retain per-group gather and pooling.
        from h3_ascend._vendor.veda.veda_reorder import LayerPack

        self.workspaces = {
            layer: LayerPack(self.impl, layer, reuse=True)
            for layer in row.get("allhead_workspace_layers", [])
        }

    def __call__(self, q, k, v, layer=0, **kwargs):
        c = self.configs[layer]
        m = self.impl
        kinds = {
            "cached128": "cached128",
            "schedule128": "schedule128",
            "schedule2": "schedule2",
            "pipe3": "pipe3",
        }
        load("veda_" + kinds[c["kernel"]])
        m.kernel_kind = c["kernel"]
        m.stack_tiles = c["stack_tiles"]
        m.pack_channels = c["pack_channels"]
        m.scatter_rows = c["scatter_rows"]
        m.pack_mode = c["pack_mode"]
        m.prefix_dense = c["prefix_dense"]
        os.environ["MIOWTION_TASK_QGROUP"] = str(c["query_group"])
        os.environ["MIOWTION_TASK_QORDER"] = str(c.get("query_order", 0))
        m.pack_workspace = self.workspaces.get(layer)
        if m.pack_workspace is not None:
            m.pack_workspace.prepare((q, k, v))
        return m(q, k, v, layer)

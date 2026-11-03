# Adapted from Ascend-H3 e02a57325029e67eafaf67d64b3d62a330f1e8d4.
# Apache-2.0; see LICENSE.upstream and NOTICE.md. Only shared_bundle injection
# and helper import paths differ from the pinned NpuVeda implementation.

"""Exact published Miowtion layout/predictor/selection with Ascend attention.

Official pure PyTorch modules are imported unchanged from the pinned snapshot.
Only device dispatch and kernel transport differ. No compensation or mask budget
change is introduced. Kernels support paired KV64 or native logical KV128.
"""
import importlib as _importlib
from pathlib import Path
import sys
import dataclasses
import torch
import torch_npu
from h3_ascend._vendor.miowtion.veda import tiling, predictor, mask, bundle
from h3_ascend._vendor.miowtion.h3.geometry import geometry_from_latent_t
from h3_ascend._vendor.veda.probe_kernel import sparse

class NpuVeda:

    def __init__(self, grid, prefix, predictor_path, device='npu:0', head_chunk=8, fused=False, stack_tiles=1, transport_kind='flat', kernel_kind='pair64', pack_channels=32, scatter_rows=32, route_kind='official', pack_mode='fused', reuse_plan=False, prefix_dense=False, head_start=0, num_heads=56, sparsity=0.9, shared_bundle=None):
        self.head_start = int(head_start)
        self.num_heads = int(num_heads)
        if not 0 <= self.head_start < self.head_start+self.num_heads <= 56: raise ValueError("invalid global head shard")
        self.prefix_dense = prefix_dense
        self.scatter_impl = 'triton'
        self.scatter_cache = 0
        self.output_workspace = False
        self._output_buffers = {}
        self.pack_mode = pack_mode
        if route_kind == 'direct':
            assert kernel_kind in ('indexed128', 'lean128', 'fast128')
        self.route_kind = route_kind
        self.routes = {}
        self.pack_channels = pack_channels
        self.scatter_rows = scatter_rows
        self.kernel_kind = kernel_kind
        self.fused = fused
        self.stack_tiles = stack_tiles
        self.transport_kind = transport_kind
        self.grid = tuple(grid)
        self.prefix = prefix
        self.device = device
        self.head_chunk = head_chunk
        self.bundle = shared_bundle if shared_bundle is not None else bundle.load(predictor_path, device='cpu')
        self.bundle.predictor.requires_grad_(False).eval().to(device)
        self.plan = self.bundle.plans.select(geometry_from_latent_t('16:9', grid[0]))
        self.plan_grid = tuple(self.plan.grid)
        if not reuse_plan:
            assert self.plan_grid == self.grid
        self.layouts = {}
        self.cpu_layouts = {}
        self.groups = []
        self.blocks = {}
        self.sizes64 = {}
        self.inverse = {}
        seq = prefix + grid[0] * grid[1] * grid[2]
        self.seq = seq
        for layer in range(50):
            groups = []
            for group in self.plan.head_groups(layer, 'cpu'):
                selected_heads = group.heads[(group.heads >= self.head_start) & (group.heads < self.head_start+self.num_heads)]
                if not selected_heads.numel(): continue
                key = str(group.shape)
                if key not in self.layouts:
                    cpu = tiling.build_tile_layout([tiling.TiledSpan(prefix, self.grid, group.shape)], seq, seq)
                    layout = dataclasses.replace(cpu, **{f.name: getattr(cpu, f.name).to(device) for f in dataclasses.fields(cpu) if isinstance(getattr(cpu, f.name), torch.Tensor)})
                    self.layouts[key] = layout
                    self.cpu_layouts[key] = cpu
                    inv = torch.empty(seq, dtype=torch.long)
                    valid = cpu.perm >= 0
                    inv[cpu.perm[valid]] = torch.arange(cpu.num_slots)[valid]
                    self.inverse[key] = inv.to(device)
                    self.sizes64[key] = torch.stack((layout.valid_count.clamp(max=64), (layout.valid_count - 64).clamp(min=0)), dim=1).flatten().int()
                    self.blocks[key] = mask.column_blocks(layout, mask.Budget(ratio=1.0-sparsity))
                    if route_kind == 'fused':
                        from h3_ascend._vendor.veda.fused_route import Route
                        self.routes[key] = Route(layout, self.blocks[key])
                    if route_kind == 'direct':
                        from indexed_route import DirectRoute
                        self.routes[key] = DirectRoute(layout, self.blocks[key])
                for heads in selected_heads.split(head_chunk):
                    groups.append((key, (heads-self.head_start).to(device)))
            self.groups.append(groups)

    @torch.inference_mode()
    def __call__(self, q, k, v, layer, audit=False):
        assert q.shape == (self.seq, self.num_heads, 128)
        if self.pack_mode == 'transpose':
            q, k, v = [x.permute(1, 0, 2).contiguous().permute(1, 0, 2) for x in (q, k, v)]
        if self.output_workspace:
            assert self.fused
            stream = int(torch.npu.current_stream(q.device).npu_stream)
            if stream not in self._output_buffers:
                self._output_buffers[stream] = q.new_empty(self.seq + 1, self.num_heads, 128)
            out = self._output_buffers[stream]
        else:
            out = q.new_empty(self.seq + 1, self.num_heads, 128) if self.fused else q.new_zeros(self.seq + 1, self.num_heads, 128)
        receipts = []
        if self.prefix_dense:
            assert self.fused and self.kernel_kind in ('native128', 'cached128', 'schedule128', 'schedule2', 'pipe1', 'pipe3', 'hint2', 'hint3') and (not audit)
            dense = torch_npu.npu_fused_infer_attention_score(q[None, :self.prefix], k[None], v[None], num_heads=self.num_heads, num_key_value_heads=self.num_heads, input_layout='BSND', scale=128 ** (-0.5), actual_seq_lengths=[self.prefix], actual_seq_lengths_kv=[self.seq])[0][0]
            out[:self.prefix].copy_(dense)
            del dense
        for key, heads in self.groups[layer]:
            layout = self.layouts[key]
            if self.fused:
                from h3_ascend._vendor.veda.fused import pack, scatter, transport, transport_rows
                if getattr(self, 'pack_workspace', None) is not None:
                    qt, kt, vt, fq, fk = self.pack_workspace.pack(q, k, v, layout, heads, self.pack_channels, self.pack_mode)
                else:
                    if self.pack_mode == 'split':
                        from h3_ascend._vendor.veda.pack_split import pack
                    qt, kt, vt, fq, fk = pack(q, k, v, layout, heads, self.pack_channels)
            else:
                qt, kt, vt = [tiling.gather_tiles(x, layout, heads) for x in (q, k, v)]
                fq, fk = [predictor.pool_tiles(x, layout) for x in (qt, kt)]
            logits = self.bundle.predictor.layers[layer](fq, fk, heads+self.head_start)
            if self.route_kind == 'direct' and (not audit):
                metadata = self.routes[key](logits)
            elif self.route_kind == 'fused' and (not audit):
                block_mask = self.routes[key](logits)
            else:
                selected = mask.select_video_blocks(logits[:, :layout.n_video_tiles], layout, self.blocks[key])
                block_mask = mask.dense_block_mask(selected, layout)
            if self.fused:
                if self.kernel_kind in ('indexed128', 'lean128', 'fast128'):
                    from indexed_route import from_mask
                    packed = metadata if self.route_kind == 'direct' and (not audit) else from_mask(block_mask.contiguous())
                    op = getattr(torch.ops, {'indexed128': 'miowtion128index_ext', 'lean128': 'miowtion128lean_ext', 'fast128': 'miowtion128fast_ext'}[self.kernel_kind]).forward
                    result = op(qt, kt, vt, packed, layout.valid_count.int(), [128, 128], None, None, 'BNSD', heads.numel(), 128 ** (-0.5), 0, True, 128, self.stack_tiles)[0]
                elif self.kernel_kind in ('native128', 'cached128', 'pool128', 'fixed128', 'schedule128', 'schedule2', 'pipe1', 'pipe3', 'hint2', 'hint3'):
                    op = getattr(torch.ops, {'native128': 'miowtion128_ext', 'cached128': 'miowtion128cache_ext', 'pool128': 'miowtion128pool_ext', 'fixed128': 'miowtion128fixed_ext', 'schedule128': 'miowtion128schedule_ext', 'schedule2': 'miowtion128schedule2_ext', 'pipe1': 'miowtion128pipe1_ext', 'pipe3': 'miowtion128pipe3_ext', 'hint2': 'miowtion128hint2_ext', 'hint3': 'miowtion128hint3_ext'}[self.kernel_kind]).forward
                    if self.prefix_dense:
                        qt = qt[:, :, :layout.n_video_tiles * 128].contiguous()
                        block_mask = block_mask[:, :layout.n_video_tiles].contiguous()
                    result = op(qt, kt, vt, block_mask[None].contiguous(), layout.valid_count.int(), [128, 128], None, None, 'BNSD', heads.numel(), 128 ** (-0.5), 0, True, 128, self.stack_tiles)[0]
                else:
                    expanded = (transport_rows if self.transport_kind == 'rows' else transport)(block_mask)
                    result = torch.ops.sol_h3_bsa_ext.forward(qt, kt, vt, expanded, self.sizes64[key], [128, 64], None, None, 'BNSD', heads.numel(), 128 ** (-0.5), 0, True, 128, self.stack_tiles)[0]
                if self.scatter_impl == 'cann':
                    torch.ops.miowtion_scatter_ext.forward(result, self.inverse[key], heads, out, self.prefix if self.prefix_dense else 0, self.scatter_rows, self.scatter_cache)
                elif self.prefix_dense:
                    from h3_ascend._vendor.veda.hybrid_prefix import scatter_video
                    scatter_video(result, self.inverse[key], heads, out, self.prefix, self.scatter_rows)
                else:
                    scatter(result, self.inverse[key], heads, out, self.scatter_rows)
            else:
                result = sparse(qt, kt, vt, block_mask, layout.valid_count)
                tiling.scatter_tiles_(out, result, layout, heads)
            if audit:
                receipts.append(dict(shape=key, heads=heads.cpu(), features_q=fq.cpu(), features_k=fk.cpu(), logits=logits.cpu(), mask=block_mask.cpu(), selected=selected.index.cpu(), keep=selected.keep.cpu(), output=result.cpu()))
        result = out[:self.seq].clone() if self.output_workspace else out[:self.seq]
        return (result, receipts) if audit else result

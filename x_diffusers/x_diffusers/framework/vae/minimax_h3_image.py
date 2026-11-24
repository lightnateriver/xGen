"""Spatially tiled encode and decode for the MiniMax-H3-Image VAE."""

from types import MethodType

import torch


def _collect_vae_tiles(positions, decode, rank, world, group, batch_size, *, gather_once):
    import torch.distributed as dist

    per_rank = (len(positions) + world - 1) // world
    local_positions = [positions[min(i * world + rank, len(positions) - 1)] for i in range(per_rank)]
    batches = (decode(local_positions[offset : offset + batch_size]) for offset in range(0, per_rank, batch_size))
    if world == 1:
        return [tile for batch in batches for tile in batch]
    if gather_once:
        local = torch.stack([tile for batch in batches for tile in batch])
        gathered = local.new_empty((world * per_rank, *local.shape[1:]))
        dist.all_gather_into_tensor(gathered, local, group=group)
        return [gathered[(index % world) * per_rank + index // world] for index in range(len(positions))]

    # Keep each collective to one tile per rank, even if the decoder ran a batch.
    output = []
    for batch in batches:
        for local in batch:
            gathered = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(gathered, local, group=group)
            output.extend(gathered[:len(positions) - len(output)])
    return output


def install_image_vae_parallel(vae, group, *, gather_once=False, decode_batch_size=1):
    """Distribute original spatial tiles, retaining the exact upstream blend order and dtype."""
    import torch.distributed as dist
    if decode_batch_size < 1:
        raise ValueError("VAE decode batch size must be at least one")
    rank, world = (dist.get_rank(group), dist.get_world_size(group)) if group is not None else (0, 1)
    originals = {"encode": vae._encode, "decode": vae._decode}

    def tiled(self, x, mode):
        if not self.use_tiling or not getattr(self, "_xgen_tile_parallel_enabled", True):
            return originals[mode](x)
        batch_size = decode_batch_size if mode == "decode" else 1
        if world == 1 and batch_size == 1:
            return originals[mode](x)
        ratio = self.spatial_compression_ratio
        height, width = x.shape[-2:]
        if mode == "decode":
            height, width = height * ratio, width * ratio
        ys, yl, yo = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        xs, xl, xo = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        if len(ys) * len(xs) < world:
            return originals[mode](x)
        positions = [(y, h, xx, w) for y, h in zip(ys, yl) for xx, w in zip(xs, xl)]
        # All tiles in a given image have equal shape, including images below the tile size on one axis.
        def decode(tiles):
            if mode == "encode":
                y, h, xx, w = tiles[0]
                tile = x[..., y:y+h, xx:xx+w]
                return (self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1].contiguous(),)
            inputs = [x[..., y//ratio:(y+h)//ratio, xx//ratio:(xx+w)//ratio] for y, h, xx, w in tiles]
            batch = inputs[0] if len(inputs) == 1 else torch.cat(inputs)
            return self.decoder(self.post_quant_conv(batch)).contiguous().split(x.shape[0], dim=0)

        output = _collect_vae_tiles(positions, decode, rank, world, group, batch_size, gather_once=gather_once)
        rows = [output[i:i+len(xs)] for i in range(0, len(output), len(xs))]
        if mode == "encode":
            yo, xo = [n//ratio for n in yo], [n//ratio for n in xo]
        return self._stitch_tiles(rows, yo, xo)

    vae._encode = MethodType(lambda self, x: tiled(self, x, "encode"), vae)
    vae._decode = MethodType(lambda self, x: tiled(self, x, "decode"), vae)
    vae._xgen_tile_parallel_enabled = True

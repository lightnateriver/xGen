"""Distributed one-block H3 TP/CP validation, with synthetic inputs and no weights.

Examples (run from the checkout with its normal dependencies):
  XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=2 tools/validate_h3_tp_cp.py --device cpu
  XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=4 tools/validate_h3_tp_cp.py --cp 2

This tests real H3 blocks and xGen collectives, not generated video quality.
Use exactly the visible devices reserved for this test. No checkpoint is read.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist


def error(actual, expected):
    difference = actual.float()-expected.float()
    return dict(max_abs=difference.abs().max().item(),
                relative_rmse=(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-8)).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cp', type=int, default=1)
    parser.add_argument('--fp8-linear', action='store_true')
    parser.add_argument('--sage', action='store_true')
    parser.add_argument('--production-adapter', action='store_true',
                        help='Use the normal FP8 resident preparation, owner AdaLN and fusions on the tiny model')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cuda':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
    device = torch.device(args.device, int(os.environ['LOCAL_RANK'])) if args.device == 'cuda' else torch.device('cpu')
    # Import the same adapters used by the normal inference entry point.
    import x_diffusers  # noqa: F401
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from x_base.sequence_parallelism.mesh import ParallelManager
    from x_base.tensor_parallelism.state import get_tp_group, get_tp_world_size
    from x_base.tensor_parallelism.fp8_reduce import FP8Reduction, encode_chunk, decode_chunk
    from x_diffusers.adaptor.h3_parallel import prepare_block
    from x_diffusers.adaptor.h3_parallel_config import H3ParallelConfig
    from x_base.operator.h3_fp8 import H3FP8Linear

    world, rank = dist.get_world_size(), dist.get_rank()
    assert world % args.cp == 0 and world//args.cp in (1,2,4), 'Minimal test requires TP1/2/4 * CP = world'
    tp = world//args.cp
    text_manager = ParallelManager(1, 1, 1, world)
    text_group = get_tp_group()
    manager = ParallelManager(1, 1, args.cp, tp, register_tp_global=False)
    assert get_tp_group() is text_group and get_tp_world_size() == world
    from x_base.tensor_parallelism.layers import RowParallelLinear
    text_probe = RowParallelLinear(16*world,16,bias=False).to(device)
    text_probe.weight.data.fill_(1)
    with torch.inference_mode():
        text_output = text_probe(torch.full((1,16),rank+1,device=device,dtype=torch.bfloat16))
    assert torch.equal(text_output,torch.full_like(text_output,16*world*(world+1)//2))
    torch.manual_seed(42)
    # Different hidden/attention widths catch the H3 inner_dim != hidden_size trap.
    reference = MiniMaxH3Transformer3DModel(hidden_size=128, num_attention_heads=56 if args.sage else 8,
        attention_head_dim=128 if args.sage else 32, num_layers=1, num_refiner_layers=1, ffn_dim=256,
        in_channels=4, audio_in_channels=8, patch_size=(1,2,2), text_dim=64,
        freq_dim=32, time_embed_hidden_dim=128, time_embed_dim=32,
        rope_freq_dim=16 if args.sage else 4).to(torch.bfloat16)
    reference.proj_out.float()
    reference.audio_proj_out.float()
    for name in ('proj_in','audio_proj_in','time_embedder','rope'):
        getattr(reference,name).float()
    reference.eval().requires_grad_(False)
    if args.sage:
        assert args.device == 'cuda'
        from x_base.operator.h3_sage_cuda import H3SageCUDA
        backend = H3SageCUDA()
        for block in reference.transformer_blocks:
            block.attn._h3_attention_backend = backend
    # Disjoint TP groups must reduce different values without cross-group leakage.
    value = torch.arange(525, dtype=torch.float32, device=device).reshape(3,175)
    partial = (torch.sin(value+rank)*2).to(torch.bfloat16)
    reducer = FP8Reduction(chunk_elements=256)
    if tp > 1:
        gathered = [torch.empty_like(partial) for _ in range(tp)]
        dist.all_gather(gathered, partial, group=manager.tp_group)
        result = reducer(partial, manager.tp_group)
        reconstructed = sum((decode_chunk(encode_chunk(p)).reshape(-1)[:p.numel()].reshape_as(p) for p in gathered))
        assert torch.equal(result, reconstructed.to(result.dtype))
        result_peers = [torch.empty_like(result) for _ in range(tp)]
        dist.all_gather(result_peers, result, group=manager.tp_group)
        assert all(torch.equal(result_peers[0],peer) for peer in result_peers[1:])
        assert reducer.metrics['max_packet_bytes'] <= 2*132
    records = []
    # Test reusable models across changing shapes and layouts, including empty local
    # modalities, a CP tail pad, mixed reference/target timesteps and negative tags.
    combinations = [(comm,cp_wire) for comm in (('bf16','fp8') if tp > 1 else ('bf16',))
                    for cp_wire in (('bf16','fp8') if args.device == 'cuda' and args.cp > 1 else ('bf16',))]
    def make_fp8(model):
        attn = model.transformer_blocks[0].attn
        attn.to_qkv = H3FP8Linear(torch.cat([attn.to_q.weight,attn.to_k.weight,attn.to_v.weight]))
        attn.to_q = attn.to_k = attn.to_v = None
        attn.fused_projections = True
        block = model.transformer_blocks[0]
        for parent, name in ((attn.to_out,'0'),(block.ff.net[0],'proj'),(block.ff.net,'2')):
            layer = getattr(parent,name)
            setattr(parent,name,H3FP8Linear(layer.weight,layer.bias))
    for comm,cp_wire in combinations:
        model = copy.deepcopy(reference)
        config = H3ParallelConfig('tp_cp', tp, args.cp, comm, cp_wire)
        if args.production_adapter:
            assert args.fp8_linear and args.device == 'cuda'
            from x_diffusers.adaptor.h3_cuda_precision import prepare_transformer
            settings = SimpleNamespace(h3_parallel_mode='tp_cp',h3_dit_tp=tp,sp=args.cp,
                h3_tp_comm=comm,h3_cp_comm=cp_wire,model='MiniMax-H3',task_type='ref2va',
                h3_cuda_precision='fp8-sage' if args.sage else 'fp8-bf16')
            prepare_transformer(SimpleNamespace(transformer=model),settings,device)
            active_manager = model.parallel_manager
            reduction = getattr(model.transformer_blocks[0].attn.to_out[0],'reduce_fn',None)
        else:
            reduction = prepare_block(model.transformer_blocks[0], manager, config, fp8=args.fp8_linear) if tp > 1 else None
            if tp == 1 and args.fp8_linear: make_fp8(model)
            model.set_parallel_manager(manager)
            model._h3_parallel_config = config
            active_manager = manager
        model.transformer_blocks[0].attn._h3_tp_cp = True
        if args.device == 'cuda' and args.cp > 1:
            from x_base.phaa.globals import enable_phaa,set_phaa_split_num
            enable_phaa()
            set_phaa_split_num(2)
            model.transformer_blocks[0].attn.processor.comm_stream = torch.cuda.Stream()
            model.transformer_blocks[0].attn._h3_phaa_fp8 = cp_wire == 'fp8'
        if args.fp8_linear:
            # Reference uses the same W8A8 operations before TP layout changes.
            expected_model = copy.deepcopy(reference)
            make_fp8(expected_model)
        else:
            expected_model = copy.deepcopy(reference)
        model.to(device)
        expected_model.to(device)
        for total, masked in ((17,False), (22,False), (19,True)):
            torch.manual_seed(100+total)
            order = torch.randperm(total, device=device)
            ti, vi, ai = order[:3], order[3:-4], order[-4:]
            tags = torch.empty(total, dtype=torch.int64, device=device)
            tags[ti], tags[vi], tags[ai] = 1,0,2
            if masked:
                tags[-1] = -1
                # The legacy single-rank upstream forward does not implement
                # negative-tag masks. Use the same local boundary at CP1 as a
                # full-document oracle for this additional opt-in capability.
                expected_model.set_parallel_manager(SimpleNamespace(sp_size=1,sp_rank=0,sp_group=None))
                expected_model._h3_parallel_config = H3ParallelConfig('tp_cp',1,1,'bf16','bf16')
            indices = (torch.arange(total,device=device)%2).long()
            positions = torch.arange(total,device=device)[:,None].expand(-1,3).clone()
            values = dict(hidden_states=torch.randn(1,len(vi),16,device=device,dtype=torch.bfloat16),
                audio_hidden_states=torch.randn(1,len(ai),8,device=device,dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1,len(ti),64,device=device,dtype=torch.bfloat16),
                timestep=torch.tensor([0.6,0.0],device=device), timestep_indices=indices, token_tags=tags,
                position_ids=positions, video_indices=vi, audio_indices=ai, text_indices=ti,return_dict=False)
            with torch.inference_mode():
                expected = expected_model(**values)
                start = time.perf_counter()
                actual = model(**values)
                if args.device == 'cuda': torch.cuda.synchronize()
                call_seconds = time.perf_counter()-start
            assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in actual)
            errors = [error(a,b) for a,b in zip(actual,expected)]
            limit = 0.025 if comm == 'bf16' and not args.fp8_linear else 0.03
            assert max(e['relative_rmse'] for e in errors) < limit, errors
            # TP peers hold identical final predictions even with FP8 reduction.
            for output in actual:
                if tp > 1:
                    peers = [torch.empty_like(output) for _ in range(tp)]
                    dist.all_gather(peers, output, group=active_manager.tp_group)
                    assert all(torch.equal(peers[0],peer) for peer in peers[1:]), 'TP output differs by rank'
            records.append(dict(tp_wire=comm,cp_wire=cp_wire,tp=tp,cp=args.cp,total=total,masked=masked,errors=errors,
                                call_seconds=call_seconds,
                                tp_metrics=dict(reduction.metrics) if reduction is not None else {},
                                cp_metrics=dict(getattr(model.transformer_blocks[0].attn,'_h3_cp_metrics',{}))))
        if reduction is not None:
            assert reduction.metrics['calls'] == 6
        assert get_tp_group() is text_group, 'DiT changed the text encoder TP group'
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank,world=world,device=str(device),fp8_linear=args.fp8_linear,sage=args.sage,
                  production_adapter=args.production_adapter,
                  geometry=dict(layers=1,hidden=128,heads=reference.config.num_attention_heads,
                                head_dim=reference.config.attention_head_dim,ffn=256,
                                rotary_channels=6*reference.config.rope_freq_dim),
                  status='passed',cases=records,reduction_probe=reducer.metrics,
                  scope='synthetic one-block H3; not full checkpoint, video quality or performance')
    if args.device == 'cuda':
        report.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    (args.output/f'rank-{rank}.json').write_text(json.dumps(report,indent=2))
    dist.barrier()
    if rank == 0: print(json.dumps(dict(status='passed',ranks=world,case_count=len(records))))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

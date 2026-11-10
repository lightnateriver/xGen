#!/usr/bin/env python3
"""Prepare fingerprinted BF16 H3 tables without loading text/VAE/main DiT weights.

Example (actual DiT steps):
  python tools/adaln_cache/prepare_cuda_adaln.py \
    --model_path weights/h3/transformer --model_variant FL2VA \
    --steps 20 50 --output_dir model-cache/fl2va-tables \
    --strip_output model-cache/fl2va-no-adaln

Reuses the existing schedule, projection, stripping and cache implementation.
The original precompute tool and its NPU behavior remain unchanged.
"""
import argparse
import fnmatch
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from safetensors import safe_open
from safetensors.torch import save_file

from precompute_adaln_cache import DELETE_PATTERNS, _compute_silu_temb, _linear_forward, create_scheduler, strip_checkpoint


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def prepare(args):
    source=Path(args.model_path).resolve();output=Path(args.output_dir).resolve()
    if output.exists():raise ValueError('output_dir already exists; refusing to overwrite prepared assets')
    destination=Path(args.strip_output).resolve() if args.strip_output else None
    if destination is not None and (destination.exists() or destination==source
                                    or destination in source.parents or source in destination.parents):
        raise ValueError('strip_output must be a new directory outside the source checkpoint')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA preparation requested but CUDA is unavailable')
    output.mkdir(parents=True)
    config=json.loads((source/'config.json').read_text())
    index=json.loads((source/'diffusion_pytorch_model.safetensors.index.json').read_text())
    keys=sorted(key for key in index['weight_map'] if any(fnmatch.fnmatch(key,p) for p in DELETE_PATTERNS))
    if config['num_layers']!=50 or len(keys)!=106:raise ValueError('Expected the complete 50-block H3 AdaLN inventory')
    with init_empty_weights(include_buffers=True):model=MiniMaxH3Transformer3DModel.from_config(config)
    expected={key for key in model.state_dict() if any(fnmatch.fnmatch(key,p) for p in DELETE_PATTERNS)}
    if set(keys)!=expected:raise ValueError('AdaLN checkpoint keys do not match the H3 model configuration')
    state={};records=[];fingerprint=hashlib.sha256()
    for shard in sorted({index['weight_map'][key] for key in keys}):
        with safe_open(source/shard,framework='pt',device='cpu') as tensors:
            for key in keys:
                if index['weight_map'][key]!=shard:continue
                value=tensors.get_tensor(key)
                dtype=torch.float32 if key.startswith('time_embedder.') else torch.bfloat16
                if value.dtype!=dtype or not torch.isfinite(value).all():raise ValueError('Expected original FP32 time MLP and BF16 AdaLN weights: '+key)
                digest=hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()
                fingerprint.update(key.encode()+bytes.fromhex(digest))
                records.append(dict(key=key,shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size(),sha256=digest))
                state[key]=value
    missing,unexpected=model.load_state_dict(state,strict=False,assign=True)
    if unexpected or set(missing)!=set(model.state_dict())-set(state):raise ValueError('Unexpected state while loading only AdaLN weights')
    del state,value
    model.eval().requires_grad_(False)
    provenance=dict(model_variant=args.model_variant,adaln_weight_sha256=fingerprint.hexdigest(),
        config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
        removed_keys=keys,removed_weight_bytes=sum(r['bytes'] for r in records),weights=records,
        video_shift=args.video_shift,audio_shift=args.audio_shift,device=args.device,
        time_mlp_dtype='float32',projection_dtype='bfloat16',allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    write(output/'provenance.json',provenance)
    device=args.device;model.time_proj.to(device);model.time_embedder.to(device)
    const=torch.tensor([.999,1.],dtype=torch.float32)
    conditions=('none','video') if args.model_variant=='FL2VA' else ('none','video','audio','both')
    selected_constants={'none':[],'video':[0],'audio':[1],'both':[0,1]}
    tables={};embeddings={};samples={};expected_outputs={}
    with torch.inference_mode():
        for steps in args.steps:
            video=create_scheduler(args.video_shift,steps+1);audio=create_scheduler(args.audio_shift,steps+1)
            if len(video.timesteps)!=steps or len(audio.timesteps)!=steps:raise ValueError('Scheduler evaluation count mismatch')
            tensors={}
            for i,(v,a) in enumerate(zip(video.timesteps,audio.timesteps)):
                for condition in conditions:
                    ts=torch.unique(torch.cat((torch.stack([v,a]).cpu().float(),const[selected_constants[condition]])))
                    tensors[f'{condition}_step{i}_timesteps']=ts.contiguous()
                    embeddings[steps,i,condition]=_compute_silu_temb(model,ts,device)
                    if i in (0,steps//2,steps-1):
                        samples[steps,i,condition]=(ts,_compute_silu_temb(model,ts,device))
            tables[steps]=tensors
        selected={0,24,49}
        for layer,block in enumerate(model.transformer_blocks):
            print(json.dumps(dict(phase='precompute',block=layer,blocks=50)),flush=True)
            projection=block.adaln_proj.linear.to(device)
            for steps,tensors in tables.items():
                for i in range(steps):
                    for condition in conditions:
                        tensors[f'{condition}_step{i}_block{layer}']=_linear_forward(projection,embeddings[steps,i,condition]).cpu().contiguous()
            if layer in selected:
                for key,(_,embedding) in samples.items():expected_outputs[key,layer]=_linear_forward(projection,embedding).cpu()
            projection.to('meta')
        projection=model.norm_out.linear.to(device)
        for steps,tensors in tables.items():
            for i in range(steps):
                for condition in conditions:
                    tensors[f'{condition}_step{i}_final']=_linear_forward(projection,embeddings[steps,i,condition]).cpu().contiguous()
        for key,(_,embedding) in samples.items():expected_outputs[key,'final']=_linear_forward(projection,embedding).cpu()
        projection.to('meta')
    files=[]
    for steps,tensors in tables.items():
        path=output/f'adaln_{args.model_variant.lower()}_bf16_{steps}step.safetensors'
        metadata=dict(num_steps=str(steps+1),num_eval_steps=str(steps),video_shift=str(args.video_shift),audio_shift=str(args.audio_shift),
            model_variant=args.model_variant,num_blocks='50',conditioning_tables=','.join(conditions),quantization='none',adaln_weight_sha256=fingerprint.hexdigest())
        save_file(tensors,path,metadata=metadata)
        files.append(dict(steps=steps,file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=metadata))
    write(output/'tables.json',files)
    # Exercise the real loader against direct combined-timestep projection.
    from x_base.adaln_cache import AdalnScheduleCache
    lookup=[];hidden=config['hidden_size']
    for row in files:
        steps=row['steps']
        for condition in conditions:
            cache=AdalnScheduleCache(SimpleNamespace(transformer_blocks=[None]*50),cache_file=str(output/row['file']),conditioning_type=condition)
            for key,(ts,_) in samples.items():
                if key[0]!=steps or key[2]!=condition:continue
                cache.maybe_install(ts.to(device))
                for layer in (*sorted(selected),'final'):
                    actual=(torch.cat(cache.lookup_final(),dim=-1) if layer=='final' else torch.cat(cache.lookup_block(layer),dim=-1)).cpu()
                    expected=expected_outputs[key,layer]
                    if layer!='final':expected=expected.view(-1,6*hidden)
                    error=actual.float()-expected.float()
                    torch.testing.assert_close(actual,expected,rtol=2*torch.finfo(torch.bfloat16).eps,atol=.001)
                    lookup.append(dict(steps=steps,step=key[1],conditioning=condition,layer=layer,bitwise_equal=torch.equal(actual,expected),
                        max_abs=float(error.abs().max()),relative_rmse=float(error.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-20))))
            del cache;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
    write(output/'lookup-verification.json',lookup)
    if destination is not None:
        strip_checkpoint(str(source),str(destination))
        new=json.loads((destination/'diffusion_pytorch_model.safetensors.index.json').read_text())
        if set(new['weight_map'])!=set(index['weight_map'])-set(keys):raise ValueError('Unexpected stripped checkpoint inventory')
        manifest={k:v for k,v in provenance.items() if k not in ('weights','device')}
        manifest.update(retained_weight_bytes=new['metadata']['total_size'],source_index_sha256=hashlib.sha256((source/'diffusion_pytorch_model.safetensors.index.json').read_bytes()).hexdigest())
        write(destination/'.xgen_adaln_source.json',manifest)
    write(output/'completed.json',dict(status='completed',tables=files,lookup_samples=len(lookup),stripped=destination is not None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',required=True)
    parser.add_argument('--model_variant',choices=('FL2VA','Ref2VA'),required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--strip_output')
    parser.add_argument('--steps',type=int,nargs='+',default=[20,50])
    parser.add_argument('--video_shift',type=float,default=12.)
    parser.add_argument('--audio_shift',type=float,default=3.)
    args=parser.parse_args()
    args.device='cuda'
    if any(step<1 for step in args.steps) or len(args.steps)!=len(set(args.steps)):parser.error('steps must be distinct positive actual evaluation counts')
    if not all(0<shift<float('inf') for shift in (args.video_shift,args.audio_shift)):parser.error('scheduler shifts must be positive and finite')
    torch.set_num_threads(4)
    # Match x_base.sequence_parallelism.mesh.initialize in CUDA H3 inference.
    torch.backends.cuda.matmul.allow_tf32=True
    prepare(args)


if __name__=='__main__':main()

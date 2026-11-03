"""Model-local optional operators; original XGEN SP/PHAA owns communication."""
import inspect
import json
import logging
from pathlib import Path
import time

import torch
import torch.distributed as dist

from x_base.infinite_vram.immutable_weights import CpuWeights
from x_base.utils.stage_metrics import Telemetry
from x_diffusers.adaptor.h3_attention_config import key, validate

logger = logging.getLogger(__name__)


class BaseAttentionSession:
    def __init__(self, model, args):
        from h3_ascend.context import VideoLayout, HeadShard
        self.layout_type, self.head_type = VideoLayout, HeadShard
        self.model, self.backend_key = model, key(args)
        self.backend = args.h3_attention
        self.sparsity = getattr(args, 'h3_veda_sparsity', 0.9)
        self.metrics = Telemetry()
        self.programs = {}
        self.bundle, self.weights = None, None
        self.calls, self.failed = 0, False
        self.permutation = None
        self.signature = inspect.signature(model.forward)
        self.layout_key = None
        self.request = None
        self.predictor_loads = 0
        started = time.perf_counter()
        if self.backend == 'veda':
            from h3_ascend._vendor.miowtion.veda import bundle
            self.bundle = bundle.load(args.h3_veda_predictor, device='cpu')
            self.bundle.predictor.requires_grad_(False).eval()
            self.weights = CpuWeights(self.bundle.predictor, 'veda.predictor', self.metrics)
            self.predictor_loads = 1
        self.initialization_seconds = time.perf_counter()-started
        self.predictor_identity = self.weights.fingerprint() if self.weights is not None else None
        model.register_forward_pre_hook(self.prepare, with_kwargs=True)
        model.register_forward_hook(self.finished, with_kwargs=True, always_call=True)
        for layer, block in enumerate(model.transformer_blocks):
            if block.attn.heads != 56:
                raise ValueError('H3 sparse operators require the original 56-head model')
            block.attn._h3_attention_backend = self.operator(block.attn, layer)
        logger.warning('H3_ATTENTION_READY experimental=true backend=%s predictor_loads=%s',
                       self.backend, self.predictor_loads)

    def begin(self, args):
        validate(args)
        if key(args) != self.backend_key:
            raise ValueError('Restart the worker to change H3 attention settings')
        if self.failed:
            raise RuntimeError('H3 attention worker failed; restart required')
        if self.calls:
            raise RuntimeError('Previous H3 request did not complete the configured steps')
        self.request = args
        self.expected_steps = args.num_inference_steps - 1
        self.metrics.reset()
        self.started = time.perf_counter()
        self.operator_calls = 0
        self.routed_heads = set()

    def prepare(self, module, positional, kwargs):
        if self.request is None or self.failed:
            raise RuntimeError('Begin a validated H3 attention request before forward')
        values = self.signature.bind(*positional, **kwargs).arguments
        if self.calls == 0:
            text, audio, video = (values[k] for k in ('text_indices','audio_indices','video_indices'))
            used = values['position_ids'].shape[0]
            i2va = self.request.task_type == 'fl2va'
            grid = (72 if i2va else 37, 24, 42)
            condition_rows = 24 * 42 if i2va else 0
            if len(video) != grid[0] * grid[1] * grid[2] + condition_rows:
                raise ValueError('Unexpected 768P Base latent grid or first-frame token count')
            permutation = torch.cat((text, audio, video)).long()
            if len(permutation) != used or not torch.equal(permutation.sort().values,
                                                          torch.arange(used, device=permutation.device)):
                raise ValueError('H3 modalities must cover every real token exactly once')
            shape = (len(text), len(audio), len(video))
            if self.layout_key != shape:
                self.programs.clear()
                self.layout_key = shape
            self.permutation = permutation
            # FL2VA puts the first frame before the generated rows in video_indices.
            # Keep it dense. The upstream 345-frame profile supplies tuning only.
            segments = (len(text), len(audio), condition_rows) if i2va else (len(text), len(audio))
            self.layout = self.layout_type(grid, sum(segments), len(text), segments, frames=345 if i2va else 124)
            if self.weights is not None:
                self.weights.onload()
        self.calls += 1

    def operator(self, attention, layer):
        def run(q, k, v, head_offset=0, attn_mask=None):
            if attn_mask is not None:
                raise ValueError('Sparse H3 operators reject arbitrary masks; SP tail padding is separate')
            if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
                raise ValueError('H3 attention requires equal batch-one BNSD Q/K/V with D=128')
            if any(x.dtype != torch.bfloat16 for x in (q,k,v)):
                raise ValueError('Experimental H3 sparse operators require BF16 Q/K/V')
            parallel = getattr(attention, 'parallel_manager', None)
            rank = dist.get_rank(parallel.sp_group) if parallel is not None else 0
            size = parallel.sp_size if parallel is not None else 1
            start = rank*(56//size)+head_offset
            heads = self.head_type(start, q.shape[1])
            program_key = (start, q.shape[1])
            first_use = program_key not in self.programs
            if first_use:
                with self.metrics.host_span('h3_attention.program_initialize'):
                    if self.backend == 'spark':
                        from h3_ascend.backends.spark import SparkAttention
                        program = SparkAttention(self.layout)
                    else:
                        from x_base.operator.h3_attention.veda import VedaAttention
                        program = VedaAttention(self.layout,heads,
                            predictor_path=self.request.h3_veda_predictor, sparsity=self.sparsity,
                            shared_bundle=self.bundle,device=str(q.device))
                    self.programs[program_key] = program
            self.routed_heads.update(range(start, start+q.shape[1]))
            with self.metrics.span('h3_attention.pack',sync=False):
                packed = [x.transpose(1,2)[0].index_select(0,self.permutation).contiguous() for x in (q,k,v)]
            with self.metrics.span('h3_attention.'+self.backend,sync=False,layer=layer,head_start=start,
                                   heads=q.shape[1],step=self.calls-1,first_use=first_use):
                result = self.programs[program_key](*packed,layer=layer)
            if result.shape != packed[0].shape or result.dtype != q.dtype:
                raise ValueError('Sparse H3 operator changed output shape or dtype')
            with self.metrics.span('h3_attention.unpack',sync=False):
                output = torch.zeros_like(q)
                output.index_copy_(2,self.permutation,result.transpose(0,1)[None])
            self.operator_calls += 1
            return output
        return run

    def finished(self, module, positional, kwargs, output):
        if output is None:
            self.failed = True
            if self.weights is not None:
                self.weights.offload()
            return
        if self.calls == self.expected_steps:
            parallel = getattr(self.model,'parallel_manager',None)
            rank_in_group = dist.get_rank(parallel.sp_group) if parallel is not None else 0
            group_size = parallel.sp_size if parallel is not None else 1
            expected_heads = set(range(rank_in_group*(56//group_size),(rank_in_group+1)*(56//group_size)))
            expected_calls = self.expected_steps*len(self.model.transformer_blocks)*max(1,self.request.phaa_num)
            if self.routed_heads != expected_heads or self.operator_calls != expected_calls:
                self.failed = True
                raise RuntimeError('Unexpected sparse attention head coverage or operator call count')
            if self.weights is not None:
                self.weights.offload()
                if self.weights.fingerprint() != self.predictor_identity:
                    self.failed = True
                    raise RuntimeError('Veda predictor CPU master identity changed')
            record = dict(experimental=True,backend=self.backend,actual_base_dit=self.expected_steps,
                          sparsity=self.sparsity if self.backend == 'veda' else None,
                          operator_calls=self.operator_calls,global_heads=sorted(self.routed_heads),
                          predictor_loads=self.predictor_loads,initialization_seconds=self.initialization_seconds,
                          predictor_identity=self.predictor_identity,
                          base_seconds=time.perf_counter()-self.started,phases=self.metrics.resolve())
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = Path(self.request.save_path).with_suffix(f'.attention.rank{rank}.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(record),encoding='utf-8')
            self.metrics.active = False
            self.calls = 0
            self.request = None
            logger.info('H3_ATTENTION_COMPLETE backend=%s rank=%s operator_calls=%s predictor_loads=%s',
                        self.backend,rank,self.operator_calls,self.predictor_loads)


def install(model, args):
    validate(args)
    if getattr(model,'_h3_attention_session',None) is not None:
        if model._h3_attention_session.backend_key != key(args):
            raise ValueError('Restart to change H3 attention backend')
        return
    model._h3_attention_session = BaseAttentionSession(model,args)

"""Opt-in Ref2VA encoder overlap on the native modular pipeline, CUDA or NPU."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import timedelta
import logging
import math
import os
import threading
import time
import weakref

logger = logging.getLogger(__name__)


def option_key(args):
    return (bool(getattr(args, 'h3_condition_overlap', False)),
            float(getattr(args, 'h3_condition_overlap_timeout', 600.)))


def validate(args, backend):
    if not option_key(args)[0]:
        return
    if backend not in ('cuda', 'npu') or getattr(args, 'model', None) != 'MiniMax-H3':
        raise ValueError('h3_condition_overlap requires CUDA or NPU MiniMax-H3')
    if getattr(args, 'task_type', None) != 'ref2va' or getattr(args, 'h3_refiner', False):
        raise ValueError('h3_condition_overlap currently supports Base Ref2VA only')
    if getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        raise ValueError('h3_condition_overlap requires the native SP layout')
    timeout = option_key(args)[1]
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('h3_condition_overlap_timeout must be finite and positive')
    if os.environ.get('DEBUG_TIME'):
        raise ValueError('Disable DEBUG_TIME barriers when using h3_condition_overlap; use its metrics instead')


def validate_manager(manager, args, backend):
    validate(args, backend)
    key = option_key(args)
    previous = getattr(manager, '_h3_condition_overlap_key',
                       option_key(getattr(manager, 'init_args', None)) if manager.pipe is not None else key)
    if manager.pipe is not None and previous != key:
        raise ValueError('Restart the manager/service to change H3 condition overlap options')
    manager._h3_condition_overlap_key = key


def _destroy_groups(groups):
    import torch.distributed as dist
    if dist.is_initialized():
        for group in reversed(groups):
            dist.destroy_process_group(group)


class _CPUTransport:
    """Transport native VAE tile bytes without concurrent accelerator collectives."""
    def __init__(self, owner):
        self.owner = owner

    @property
    def group(self):
        return self.owner.vae_group

    def gather(self, received, payload):
        import torch
        import torch.distributed as dist
        host = payload.detach().cpu()
        buffers = [torch.empty_like(host) for _ in received]
        dist.all_gather(buffers, host, group=self.group)
        for target, source in zip(received, buffers):
            target.copy_(source)
        self.owner.metrics['vae_cpu_gathers'] += 1
        self.owner.metrics['vae_payload_bytes'] += host.numel() * host.element_size()


class ConditionOverlap:
    def __init__(self, pipe, args):
        import torch
        import torch.distributed as dist
        from x_base.device import device_type

        self.pipe = weakref.proxy(pipe)
        self.timeout = option_key(args)[1]
        self.device = torch.device(device_type(), torch.get_device_module(device_type()).current_device())
        self.accelerator = torch.get_device_module(self.device.type)
        self.stream = self.accelerator.Stream(device=self.device)
        self.lock = threading.Lock()
        self.text_done = threading.Event()
        self.cancelled = threading.Event()
        self.failed = self.active = False
        self.future = self.pool = None
        self.metrics = {}
        self.vae_group = self.audio_gate = None
        groups = []
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world != getattr(args, 'sp', 1):
            raise ValueError('h3_condition_overlap requires one model SP group, without extra DP ranks')
        if world > 1:
            if not dist.is_gloo_available():
                raise ValueError('h3_condition_overlap requires CPU Gloo for multi-rank execution')
            try:
                for _ in range(2):
                    groups.append(dist.new_group(ranks=list(range(world)), backend='gloo',
                                                 timeout=timedelta(seconds=self.timeout)))
                self.vae_group, self.audio_gate = groups
            except BaseException:
                _destroy_groups(groups)
                raise
        weakref.finalize(pipe, _destroy_groups, groups)
        self.transport = _CPUTransport(self)
        self._install_blocks(pipe)
        self._guard_audio(pipe)

    def _install_blocks(self, pipe):
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VATextEncoderStep as Text,
            MiniMaxH3Ref2VAReferenceEncoderStep as Reference,
        )
        owner = self

        class OverlapText(Text):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def __call__(self, components, state):
                return owner.text(self.original, components, state)

        class JoinReference(Reference):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def __call__(self, components, state):
                return owner.reference(self.original, components, state)

        found = []
        def visit(block):
            for name, child in getattr(block, 'sub_blocks', {}).items():
                if isinstance(child, (Text, Reference)):
                    found.append((block, name, child))
                else:
                    visit(child)
        visit(pipe._blocks)
        if (sum(isinstance(child, Text) for _, _, child in found) != 1
                or sum(isinstance(child, Reference) for _, _, child in found) != 1):
            raise ValueError('Unsupported Ref2VA encoder block layout for condition overlap')
        self.replacements = []
        for parent, name, original in found:
            wrapped = OverlapText(original) if isinstance(original, Text) else JoinReference(original)
            self.replacements.append((parent, name, original, wrapped))
            if isinstance(original, Reference):
                self.original_reference = original

    def _guard_audio(self, pipe):
        from functools import wraps
        original = pipe.audio_vae.encode
        @wraps(original)
        def encode(*values, **kwargs):
            if self.active:
                if not self.text_done.wait(self.timeout):
                    raise TimeoutError('Text completion gate for reference audio timed out')
                if self.cancelled.is_set():
                    raise RuntimeError('Reference encoding cancelled after text preparation failed')
            return original(*values, **kwargs)
        self.original_audio_encode, self.audio_encode = original, encode

    def _run_reference(self, components, shadow, inference_mode):
        import torch
        vae = components.vae
        previous = getattr(vae, '_h3_overlap_transport', None)
        vae._h3_overlap_transport = self.transport
        started = time.perf_counter()
        try:
            with (torch.inference_mode(inference_mode), torch.no_grad(),
                  self.accelerator.device(self.device), self.accelerator.stream(self.stream)):
                _, shadow = self.original_reference(components, shadow)
                self.stream.synchronize()
            return shadow
        finally:
            if previous is None:
                del vae._h3_overlap_transport
            else:
                vae._h3_overlap_transport = previous
            self.metrics['reference_wall_s'] = time.perf_counter() - started

    def text(self, original, components, state):
        import torch
        import torch.distributed as dist
        refs = state.get('normalized_references')
        if not self.active or not refs:
            return original(components, state)
        if self.future is not None:
            raise RuntimeError('Reference encoding was already submitted for this request')
        shadow = type(state)(values=dict(state.values), kwargs_mapping=deepcopy(state.kwargs_mapping))
        self.stream.wait_stream(self.accelerator.current_stream(self.device))
        self.future = self.pool.submit(self._run_reference, components, shadow, torch.is_inference_mode_enabled())
        self.metrics['submitted'] += 1
        started = time.perf_counter()
        try:
            result = original(components, state)
            if any(ref.has_audio for ref in refs):
                # Audio keeps its native collective order, after all text ranks finish.
                self.accelerator.current_stream(self.device).synchronize()
                if self.audio_gate is not None:
                    dist.barrier(group=self.audio_gate)
            self.text_done.set()
            return result
        except BaseException:
            self.cancelled.set()
            self.text_done.set()
            raise
        finally:
            self.metrics['text_wall_s'] = time.perf_counter() - started

    def reference(self, original, components, state):
        if not self.active or self.future is None:
            return original(components, state)
        started = time.perf_counter()
        shadow = self.future.result(timeout=self.timeout)
        self.future = None
        block = original.get_block_state(state)
        block.condition_latents = shadow.get('condition_latents')
        block.audio_condition_latents = shadow.get('audio_condition_latents')
        original.set_block_state(state, block)
        self.metrics['joined'] += 1
        self.metrics['join_wait_s'] = time.perf_counter() - started
        return components, state

    @contextmanager
    def request(self):
        if self.failed:
            raise RuntimeError('Condition-overlap worker failed; restart the manager/service')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('One Ref2VA pipeline cannot prepare concurrent requests')
        self.metrics = dict(backend=self.device.type, submitted=0, joined=0, text_wall_s=0.,
                            reference_wall_s=0., join_wait_s=0., vae_cpu_gathers=0, vae_payload_bytes=0)
        self.cancelled.clear()
        self.text_done.clear()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='h3-reference')
        for parent, name, original, wrapped in self.replacements:
            parent.sub_blocks[name] = wrapped
        self.pipe.audio_vae.encode = self.audio_encode
        self.active = True
        error = None
        try:
            yield self
        except BaseException as exc:
            error = exc
            self.failed = True
            raise
        finally:
            self.cancelled.set()
            self.text_done.set()
            drain_error = None
            try:
                if isinstance(error, FutureTimeout):
                    drain_error = error
                elif self.future is not None:
                    self.future.result(timeout=self.timeout)
            except BaseException as exc:
                drain_error = exc
                self.failed = True
            finally:
                # A timed-out worker must be retired, never reused or retried here.
                self.pool.shutdown(wait=not isinstance(drain_error, FutureTimeout), cancel_futures=True)
                self.pool = self.future = None
                self.active = False
                for parent, name, original, wrapped in self.replacements:
                    parent.sub_blocks[name] = original
                self.pipe.audio_vae.encode = self.original_audio_encode
                self.metrics['failed'] = self.failed
                self.pipe._h3_condition_overlap_metrics = dict(self.metrics)
                self.lock.release()
                logger.info('H3 condition overlap: %s', self.metrics)
            if drain_error is not None and error is None:
                raise drain_error


def request_context(pipe, args):
    if not option_key(args)[0]:
        return nullcontext()
    owner = getattr(pipe, '_h3_condition_overlap', None)
    if owner is None:
        owner = ConditionOverlap(pipe, args)
        pipe._h3_condition_overlap = owner
    return owner.request()

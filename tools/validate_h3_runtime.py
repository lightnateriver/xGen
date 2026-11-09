"""Run explicit H3 functional cases through the existing resident Manager.

Pass the normal xGen inference arguments after --case-plan and --result-dir.
One step checks functionality/capacity only; it is not a quality benchmark.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist

from x_base.device import device_type
from x_diffusers import InferenceManager, init_env, parse_args


class CountedManager(InferenceManager):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.completions = 0
        self.builds = 0

    def _prepare_pipeline(self, settings):
        previous = self.pipe
        super()._prepare_pipeline(settings)
        if previous is self.pipe:
            return
        self.builds += 1
        self.pipe.transformer.register_forward_pre_hook(self._before)
        self.pipe.transformer.register_forward_hook(self._after)

    def _before(self, module, values):
        self.calls += 1

    def _after(self, module, values, output):
        self.completions += 1


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--case-plan', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    local, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    settings = parse_args()
    if settings.model != 'MiniMax-H3':
        raise ValueError('This runtime validator is specific to MiniMax-H3')
    cases = json.loads(local.case_plan.read_text(encoding='utf-8'))
    allowed = {'name', 'prompt', 'frames', 'task_type', 'reference_paths', 'keyframe_image',
               'num_inference_steps', 'expect_reuse', 'expected_frames'}
    if not cases or any(set(case) - allowed for case in cases):
        raise ValueError('Plan must contain explicit cases and per-request input fields only')
    init_env()
    rank = dist.get_rank()
    device = torch.get_device_module(device_type())
    manager = CountedManager()
    for index, case in enumerate(cases):
        name = case['name']
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Case name must be a single directory name')
        folder = local.result_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder / f'rank-{rank}.json'
        if record_path.exists():
            raise FileExistsError('Refuse to overwrite a previous case')
        args = deepcopy(settings)
        for key, value in case.items():
            if key not in ('name', 'expect_reuse', 'expected_frames'):
                setattr(args, key, value)
        args.save_path = str(folder / 'video.mp4')
        before_pipe = manager.pipe
        manager.calls = manager.completions = 0
        device.reset_peak_memory_stats()
        started = time.perf_counter()
        record = dict(case=name, rank=rank, steps=args.num_inference_steps, status='running')
        try:
            seconds = manager.infer(args)
            device.synchronize()
            if case.get('expect_reuse'):
                assert before_pipe is not None and manager.pipe is before_pipe, 'Model unexpectedly rebuilt'
            assert manager.calls == manager.completions == args.num_inference_steps
            record.update(status='generated', inference_save_s=seconds, dit_calls=manager.calls,
                dit_completed=manager.completions, model_builds=manager.builds,
                model_identity=[id(manager.pipe), id(manager.pipe.transformer)],
                effective_cuda_options={k: v for k, v in vars(args).items() if k.startswith('h3_cuda_')},
                condition_metrics=dict(getattr(manager.pipe.transformer, '_h3_conditioning_metrics', {})),
                audio_metrics=dict(getattr(manager.pipe.audio_vae, '_h3_parallel_metrics', {})))
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
            raise
        finally:
            record.update(wall_s=time.perf_counter()-started,
                allocated_after=device.memory_allocated(), reserved_after=device.memory_reserved(),
                peak_allocated=device.max_memory_allocated(), peak_reserved=device.max_memory_reserved())
            record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        dist.barrier()
        gate = [None]
        if rank == 0:
            try:
                video = folder / 'video.mp4'
                probe = json.loads(subprocess.check_output([local.ffprobe, '-v', 'error', '-show_streams',
                    '-of', 'json', str(video)], text=True))
                stream = next(v for v in probe['streams'] if v['codec_type'] == 'video')
                assert (stream['width'], stream['height']) == (args.width, args.height)
                expected_frames = case.get('expected_frames', args.frames)
                assert int(stream['nb_frames']) == expected_frames
                assert any(v['codec_type'] == 'audio' for v in probe['streams'])
                subprocess.run([local.ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', str(video),
                    '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
                gate[0] = dict(status='technical_pass', frames=int(stream['nb_frames']),
                    sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    human_full_video_audio_review='pending')
            except Exception as exc:
                gate[0] = dict(status='failed', error=repr(exc))
            (folder/'result.json').write_text(json.dumps(gate[0], indent=2))
        dist.broadcast_object_list(gate, src=0)
        assert gate[0]['status'] == 'technical_pass', gate[0]
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

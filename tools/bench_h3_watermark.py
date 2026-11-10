"""CPU-only watermark A/B: reuse xGen's actual converter and FFmpeg writer.

Example (from the repository root):
  python tools/bench_h3_watermark.py --input samples/video.mp4 --output outputs/watermark-bench

No torch/model import or GPU use. A bounded bank of real decoded RGB8 frames is
looped to match H3's saved frame counts: 124/243/360 for nominal 5/10/15 seconds.
Decoding, audio preparation and warmup are excluded from the measured interval.
The source must be 1344x768 with at least 24 video frames and 15 seconds of audio.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server/video'))
from maas_artifacts import publish_video, watermark_converter


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def load_encoder():
    """Execute only two unchanged CPU methods; never import the model server."""
    path = ROOT/'server/video/video_server.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VideoGenerationService')
    methods = [node for node in server.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_frame_to_uint8_hwc', '_save_mmh3_video_ffmpeg')]
    if len(methods) != 2:
        raise RuntimeError('Expected the current xGen CPU encoder methods')
    cls = ast.ClassDef(name='CPUEncoder', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = dict(np=np, Image=Image, os=os, subprocess=subprocess,
                     logger=logging.getLogger('watermark-bench'),
                     torch=SimpleNamespace(Tensor=type('UnsupportedTensor', (), {})))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['CPUEncoder']()


def cpu_name():
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--overlay-repeats', type=int, default=7)
    parser.add_argument('--bank-frames', type=int, default=24)
    args = parser.parse_args()
    if min(args.repeats, args.overlay_repeats, args.bank_frames) < 1:
        parser.error('Repeat counts and bank size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = os.getenv('FFMPEG_PATH', 'ffmpeg'), os.getenv('FFPROBE_PATH', 'ffprobe')
    info = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(args.input)]))
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    width, height = video['width'], video['height']
    if (width, height) != (1344, 768):
        parser.error('This minimal protocol requires a 1344x768 input')
    audio_stream = next((stream for stream in info['streams'] if stream['codec_type'] == 'audio'), None)
    if audio_stream is None or float(audio_stream.get('duration') or info['format']['duration']) < 15:
        parser.error('The source must include at least 15 seconds of audio')
    raw = subprocess.check_output([ffmpeg, '-v', 'error', '-threads', '1', '-hwaccel', 'none', '-i', str(args.input),
        '-an', '-frames:v', str(args.bank_frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    expected = args.bank_frames*width*height*3
    if len(raw) != expected:
        raise RuntimeError('Source video has too few frames')
    bank = np.frombuffer(raw, dtype=np.uint8).reshape(args.bank_frames, height, width, 3)
    audio = args.output/'audio.wav'
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-threads', '1', '-hwaccel', 'none', '-i', str(args.input),
                    '-map', '0:a:0', '-vn', '-t', '15', '-c:a', 'pcm_s16le', str(audio)], check=True)
    encoder = load_encoder()
    frame_counts = {5: 124, 10: 243, 15: 360}
    records = []
    protocol = dict(cpu=cpu_name(), logical_cpus=os.cpu_count(), os=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, pillow=pillow_version, ffmpeg=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
        frame_bank_bytes=len(raw), frame_bank_frames=args.bank_frames, frame_bank_sha256=hashlib.sha256(raw).hexdigest(),
        input_sha256=digest(args.input), source_video=dict(width=width, height=height, fps=video['r_frame_rate']),
        encoder=dict(video='libx264', preset='veryfast', crf=10, pixel_format='yuv420p', audio='aac', threads='x264 default'),
        saved_frames=frame_counts, paired_encode_repeats=args.repeats, paired_overlay_repeats=args.overlay_repeats,
        measured_scope='CPU RGB8 input -> optional watermark -> bytes/pipe -> H264/AAC/mux -> ffprobe -> atomic publish',
        excluded=['model/VAE/D2H', 'source decoding', 'audio WAV preparation', 'OBS/network upload', 'warmup'],
        bounded_loop=True, gpu_used=False,
        source_hashes={name:digest(ROOT/name) for name in ['server/video/maas_artifacts.py','server/video/video_server.py']})

    def encode(seconds, watermark, destination):
        frames = [bank[i % len(bank)] for i in range(frame_counts[seconds])]
        options = SimpleNamespace(save_fps=24, h3_cuda_precision='fp8-bf16', _maas_watermark=watermark)
        start = time.perf_counter_ns()
        publish_video(destination, lambda path: encoder._save_mmh3_video_ffmpeg(frames, str(audio), path, options),
                      require_audio=True, expected=(width, height, len(frames)))
        return (time.perf_counter_ns()-start)/1e9

    print(json.dumps(dict(stage='prepared', cpu=protocol['cpu'], bank_mib=round(len(raw)/1024**2, 1)), ensure_ascii=False), flush=True)
    for watermark in (False, True):
        encode(5, watermark, args.output/f'warmup-{int(watermark)}.mp4')
    print('Encoder warmup complete', flush=True)
    for seconds in frame_counts:
        frames = [bank[i % len(bank)] for i in range(frame_counts[seconds])]
        for repeat in range(args.overlay_repeats):
            for watermark in ((False, True) if repeat % 2 == 0 else (True, False)):
                start = time.perf_counter_ns()
                convert = watermark_converter(encoder._frame_to_uint8_hwc, width, height) if watermark else encoder._frame_to_uint8_hwc
                for frame in frames:
                    output = convert(frame)
                elapsed = (time.perf_counter_ns()-start)/1e9
                records.append(dict(kind='overlay', seconds=seconds, repeat=repeat, watermark=watermark, elapsed_s=elapsed))
        for repeat in range(args.repeats):
            order = (False, True) if (repeat+seconds//5) % 2 == 0 else (True, False)
            for watermark in order:
                destination = args.output/f'{seconds}s-r{repeat+1:02d}-{"on" if watermark else "off"}.mp4'
                elapsed = encode(seconds, watermark, destination)
                records.append(dict(kind='encode_publish', seconds=seconds, repeat=repeat, watermark=watermark,
                                    elapsed_s=elapsed, video=destination.name, bytes=destination.stat().st_size,
                                    sha256=digest(destination), technical_check='passed'))
                (args.output/'progress.json').write_text(json.dumps(dict(protocol=protocol, records=records), indent=2), encoding='utf-8')
                print(json.dumps(records[-1]), flush=True)

    summary = []
    for seconds, count in frame_counts.items():
        row = dict(seconds=seconds, frames=count, actual_video_seconds=count/24)
        for kind, prefix in [('overlay', 'overlay'), ('encode_publish', 'encode')]:
            selected = [r for r in records if r['seconds'] == seconds and r['kind'] == kind]
            off = statistics.median(r['elapsed_s'] for r in selected if not r['watermark'])
            on = statistics.median(r['elapsed_s'] for r in selected if r['watermark'])
            paired = [next(r['elapsed_s'] for r in selected if r['repeat']==repeat and r['watermark'])-
                      next(r['elapsed_s'] for r in selected if r['repeat']==repeat and not r['watermark'])
                      for repeat in sorted({r['repeat'] for r in selected})]
            row.update({prefix+'_off_s':off, prefix+'_on_s':on, prefix+'_median_delta_s':on-off,
                        prefix+'_paired_delta_s':statistics.median(paired), prefix+'_paired_range_s':[min(paired),max(paired)]})
        summary.append(row)
    result = dict(protocol=protocol, records=records, summary=summary, checked_at_utc=datetime.now(timezone.utc).isoformat())
    (args.output/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ['# H3 水印 CPU 增量实测', '', f"CPU：{protocol['cpu']}；{protocol['os']}。不加载模型、不使用 GPU。", '',
        '固定 1344×768 RGB8、24 fps；采用真实成片解码的 24 帧缓冲循环输入（约 71 MiB），不将循环样本用于画面质量评估。',
        '沿用当前 xGen 的两个真实 CPU 编码方法和 watermark_converter；H264 veryfast/CRF10、AAC。',
        '5/10 秒按 xGen 原生 padding 分别保存 124/243 帧，15 秒保存 360 帧，实际时长分别为 5.167/10.125/15 秒。',
        f'编码先预热两次，再交错配对测 {args.repeats} 轮；纯转换测 {args.overlay_repeats} 轮。', '',
        '| 请求时长 | 帧数 | 纯水印额外 CPU / ms | 无水印编码保存 / s | 有水印编码保存 / s | 配对净增量 / s | 配对增量范围 / s |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in summary:
        low, high = row['encode_paired_range_s']
        lines.append(f"| {row['seconds']}s | {row['frames']} | {1000*row['overlay_paired_delta_s']:.2f} | {row['encode_off_s']:.3f} | {row['encode_on_s']:.3f} | {row['encode_paired_delta_s']:+.3f} | {low:+.3f} ～ {high:+.3f} |")
    lines += ['', '纯水印增量含图块构造、每帧完整 RGB 复制及覆盖，不含 bytes 序列化和编码。',
        '编码保存包含 bytes/pipe、H264/AAC、封装、ffprobe 技术校验及原子发布，不含视频生成、VAE、D2H、素材解码和音频 WAV 准备。',
        '各列为配对或各组中位数；配对净增量不一定等于两个独立中位数之差。',
        '负的配对编码增量可能来自调度/编码波动及局部图像复杂度变化，不能据此宣称水印加速。',
        '本机 CPU 与服务器的编码性能不同，不能把这些秒数当作 5090 服务的实际整网 E2E。', '',
        '每条计时视频均经真实 ffprobe 检查尺寸、帧数、24 fps、视频包数和音轨；原始耗时、配置、实现/输入哈希见 results.json。',
        '输出目录保留 5/10/15 秒各轮开/关水印视频。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert 'torch' not in sys.modules, 'Benchmark must not import a GPU/model runtime'
    print(json.dumps(dict(stage='complete', summary=summary, report=str(args.output/'report.md')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

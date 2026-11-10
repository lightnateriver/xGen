"""Run the NPU H3 offline matrix in resident eight-card groups."""
import argparse
from copy import copy
import csv
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "example/diffusers/scripts/run_h3.sh"
SMOKE_CASES = {"t2va-5s-768p", "i2va-15s-768p", "t2va-5s-2k",
               "i2va-5s-2k", "max-mixed", "mixed-2k"}


def manifest_inputs(path, matrix):
    root = path.parent

    def asset(name):
        relative = Path(name)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"Manifest asset must be inside the package: {name}")
        return resolved

    entries = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        case_id = row["case_id"]
        if case_id in entries:
            raise ValueError(f"Duplicate manifest case: {case_id}")
        entries[case_id] = row
    if entries.keys() != {case[0] for case in matrix}:
        raise ValueError("Manifest IDs must match the benchmark matrix")
    for name, task, seconds, resolution, expected in matrix:
        row = entries[name]
        media = [asset(item) for item in row["inputs"]]
        if (row["task"], row["duration_s"], row["resolution"], row["status"]) != (task, seconds, resolution, "ready"):
            raise ValueError(f"Unexpected metadata or readiness for {name}")
        if [file.name for file in media] != [file.name for file in expected]:
            raise ValueError(f"Unexpected ordered input list for {name}")
        yield name, task, seconds, resolution, media, asset(row["prompt"])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_matrix(inputs):
    for task, counts in (("t2va", (0,)), ("i2va", (1,)), ("r2va", (1, 5, 9))):
        for count in counts:
            media = ([inputs / "first-frame.png"] if task == "i2va" else
                     [inputs / f"ref{i:02}.png" for i in range(1, count + 1)])
            name = f"r2va-{count}img" if task == "r2va" else task
            for seconds in (5, 10, 15):
                for resolution in ("768p", "2k"):
                    yield f"{name}-{seconds}s-{resolution}", task, seconds, resolution, media
    for name, seconds, resolution, files in (
        ("video-silent", 5, "768p", ["video-2s.mp4"]),
        ("video-sound", 5, "768p", ["video-2s-with-audio.mp4"]),
        ("image-audio", 5, "768p", ["ref01.png", "audio-5s.wav"]),
        ("max-mixed", 15, "768p", [f"ref{i:02}.png" for i in range(1, 7)] +
         [f"video-{n}s.mp4" for n in (2, 5, 8)] + [f"audio-{n}s.wav" for n in (2, 5, 8)]),
        ("mixed-2k", 5, "2k", ["ref01.png", "video-2s.mp4", "audio-2s.wav"]),
    ):
        yield name, "r2va", seconds, resolution, [inputs / file for file in files]


def fields(path):
    return dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)


def memory_kib(path, field):
    try:
        return int(fields(path)[field].split()[0])
    except (OSError, KeyError, ValueError):
        return None


def group_pids(root_pid, seen):
    if root_pid is None:
        return []
    table = subprocess.check_output(["ps", "-eo", "pid=,ppid=,sid=,stat="], text=True)
    processes = {}
    for line in table.splitlines():
        pid, parent, session, state = line.split()
        if not state.startswith("Z"):
            processes[int(pid)] = (int(parent), int(session))
    # torchrun starts each rank in a separate session. Also retain observed
    # descendants so a rank reparented on launcher exit remains visible.
    selected = {pid for pid, (_, session) in processes.items()
                if pid == root_pid or session == root_pid or pid in seen}
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in selected}
        if children <= selected:
            break
        selected.update(children)
    seen.update(selected)
    return sorted(selected)


def hbm_devices(raw):
    """Parse the separate device/chip rows in Ascend 910B npu-smi info."""
    devices, device, chip_name = {}, None, None
    for line in raw.splitlines():
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        header = re.fullmatch(r"(\d+)\s+(\S.*)", parts[0])
        if header and re.fullmatch(r"[A-Za-z]+", parts[1]):
            device, chip_name = header.groups()
        if device is None or not re.fullmatch(r"\d+", parts[0]):
            continue
        hbm = re.fullmatch(r"\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", parts[2])
        if hbm:
            devices[f"{device}:{parts[0]}"] = {
                "name": chip_name, "used_mib": int(hbm[1]), "total_mib": int(hbm[2]),
            }
    return devices


def sample(root_pid=None, seen=None):
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    pids = group_pids(root_pid, seen if seen is not None else set())
    info = {}
    for pid in pids:
        rss = memory_kib(Path(f"/proc/{pid}/status"), "VmRSS")
        if rss is not None:
            info[pid] = {"rss_kib": rss, "pss_kib": memory_kib(Path(f"/proc/{pid}/smaps_rollup"), "Pss")}
    try:
        raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=8)
        hbm = hbm_devices(raw)
        idle = len(re.findall(r"No running processes found in NPU \d+", raw))
        error = None if hbm else "No supported HBM rows in npu-smi info"
    except (OSError, subprocess.SubprocessError) as exc:
        hbm, idle, error = {}, 0, str(exc)
    mem = fields(Path("/proc/meminfo"))
    available = int(mem["MemAvailable"].split()[0])
    rss = [p["rss_kib"] for p in info.values()]
    pss = [p["pss_kib"] for p in info.values() if p["pss_kib"] is not None]
    return {
        "time_utc": timestamp, "monotonic_s": started, "sample_duration_s": time.monotonic() - started,
        "hbm": hbm, "hbm_error": error, "idle_cards": idle, "active_pids": pids, "processes": info,
        "process_rss_kib": sum(rss), "process_pss_kib": sum(pss) if len(pss) == len(rss) else None,
        "pss_coverage": [len(pss), len(rss)],
        "host_available_kib": available,
        "host_used_kib": int(mem["MemTotal"].split()[0]) - available,
        "host_cached_kib": int(mem["Cached"].split()[0]),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def check_adaln(path, variants):
    with path.open("rb") as stream:
        size, = struct.unpack("<Q", stream.read(8))
        if not 0 < size <= 100_000_000:
            raise ValueError("Invalid safetensors header length")
        metadata = json.loads(stream.read(size))["__metadata__"]
    if (int(metadata["num_steps"]), int(metadata["num_eval_steps"])) != (21, 20):
        raise ValueError("Expected a 21-point AdaLN table for 20 actual Base DiT steps")
    if metadata["model_variant"] not in variants or (
        float(metadata["video_shift"]), float(metadata["audio_shift"])
    ) != (12.0, 3.0):
        raise ValueError(f"Expected an AdaLN table for {sorted(variants)} with shifts 12/3")
    return metadata


def worker():
    import torch
    import torch.distributed as dist
    from x_diffusers import InferenceManager, init_env, parse_args
    from x_base.utils.stage_metrics import Telemetry
    from x_base.vae_parallelism.save_av_stream import SaveAVStream

    class BenchmarkManager(InferenceManager):
        def __init__(self):
            super().__init__()
            self.telemetry = Telemetry()
            self.dit_calls = 0
            self.instrumented = None

        def _prepare_pipeline(self, args):
            start = time.perf_counter()
            super()._prepare_pipeline(args)
            self.prepare_seconds = time.perf_counter() - start
            if args.h3_refiner or self.pipe is self.instrumented:
                return
            self.instrumented = self.pipe
            original = self.pipe.transformer.forward

            @wraps(original)
            def measured(*values, **kwargs):
                self.dit_calls += 1
                with self.telemetry.span("h3.dit", sync=False):
                    return original(*values, **kwargs)

            self.pipe.transformer.forward = measured
            decode_target = getattr(self.pipe, '_h3_vae_decode_backend', None) or self.pipe.vae
            decode_method = 'decode_video' if decode_target is not self.pipe.vae else 'decode'
            original_decode = getattr(decode_target, decode_method)

            @wraps(original_decode)
            def decode(*values, **kwargs):
                with self.telemetry.host_span("vae.decode"):
                    return original_decode(*values, **kwargs)

            setattr(decode_target, decode_method, decode)

    template = parse_args()
    init_env()
    manager = BenchmarkManager()
    if not template.h3_refiner:
        original_close = SaveAVStream.close

        @wraps(original_close)
        def close(stream, *values, **kwargs):
            with manager.telemetry.host_span("video.encode_and_audio_mux_wait"):
                return original_close(stream, *values, **kwargs)

        SaveAVStream.close = close
    rank = dist.get_rank()
    accelerator = torch.get_device_module("npu")
    jobs = json.loads(Path(os.environ["XGEN_H3_BENCH_PLAN"]).read_text(encoding="utf-8"))
    previous_pipe_id = None
    for job in jobs:
        args = copy(template)
        args.task_type = {"t2va": "t2va", "i2va": "fl2va", "r2va": "ref2va"}[job["task"]]
        args.prompt = Path(job["prompt"]).read_text(encoding="utf-8").rstrip("\n")
        args.keyframe_image = job["media"][0] if job["task"] == "i2va" else ""
        args.reference_paths = job["media"] if job["task"] == "r2va" else None
        args.frames = {5: 124, 10: 243, 15: 362}[job["seconds"]]
        args.h3_refiner_seconds = job["seconds"]
        args.save_path = job["video"]
        manager.dit_calls = 0
        if not args.h3_refiner:
            manager.telemetry.reset()
        accelerator.reset_peak_memory_stats()
        start = time.perf_counter()
        infer_save_s = manager.infer(args)
        torch.npu.synchronize()
        wall = time.perf_counter() - start
        phases = manager.telemetry.resolve() if not args.h3_refiner else []
        reused = previous_pipe_id == id(manager.pipe)
        if previous_pipe_id is not None and not reused:
            raise RuntimeError("The model was rebuilt inside a resident benchmark group")
        boundary = Path(job["boundary"])
        boundary.mkdir(exist_ok=True)
        write_json(boundary / f"rank-{rank}.json", {
            "rank": rank, "pid": os.getpid(), "wall_s": wall,
            "infer_save_s": infer_save_s, "allocated_bytes": accelerator.memory_allocated(),
            "reserved_bytes": accelerator.memory_reserved(),
            "peak_allocated_bytes": accelerator.max_memory_allocated(),
            "peak_reserved_bytes": accelerator.max_memory_reserved(),
            "model_reused": reused, "prepare_seconds": manager.prepare_seconds,
            "base_dit_calls": manager.dit_calls if not args.h3_refiner else None,
            "native_vae": getattr(getattr(manager.pipe, '_h3_vae_decode_backend', None), 'metrics', None),
            "phases": phases,
        })
        previous_pipe_id = id(manager.pipe)
        dist.barrier()
        if rank == 0:
            (boundary / "ready").touch()
            deadline = time.monotonic() + 300
            while not (boundary / "sampled").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Benchmark monitor did not record request boundary")
                time.sleep(0.1)
        dist.barrier()


def media_check(path, resolution, seconds):
    probe = json.loads(subprocess.check_output(
        [os.environ.get("FFPROBE_PATH", "ffprobe"), "-v", "error", "-show_streams",
         "-of", "json", str(path)], text=True, timeout=30,
    ))
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    width, height, frames = ((2560, 1440, seconds * 24) if resolution == "2k" else
                             (1344, 768, {5: 124, 10: 243, 15: 362}[seconds]))
    if (video["width"], video["height"], video["avg_frame_rate"], int(video["nb_frames"])) != (
        width, height, "24/1", frames,
    ):
        raise ValueError(f"Unexpected output geometry or frame count: {path.name}")
    video_seconds = float(video["duration"])
    audio_seconds = float(audio["duration"])
    if abs(video_seconds - audio_seconds) > 0.5:
        raise ValueError(f"Video and audio durations differ by over 0.5 s: {path.name}")
    subprocess.run([os.environ.get("FFMPEG_PATH", "ffmpeg"), "-v", "error", "-xerror", "-nostdin",
                    "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], check=True, timeout=180)
    return {"frames": frames, "width": width, "height": height, "audio_codec": audio["codec_name"],
            "video_seconds": video_seconds, "audio_seconds": audio_seconds, "sha256": digest(path)}


def group_cases(cases):
    buckets = {}
    for case in cases:
        _, task, _, resolution, _, _ = case
        workflow = "t2va-i2va" if resolution == "768p" and task != "r2va" else task
        buckets.setdefault((workflow, resolution), []).append(case)
    order = [("t2va-i2va", "768p"), ("t2va", "2k"), ("i2va", "2k"), ("r2va", "768p"), ("r2va", "2k")]
    return [(f"{task}-{resolution}", buckets[task, resolution])
            for task, resolution in order if (task, resolution) in buckets]


def jobs_for(cases, args):
    for name, task, seconds, resolution, media, prompt in cases:
        directory = args.output / name
        directory.mkdir()
        hashes = [dict(name=file.name, sha256=args.hashes[file]) for file in (prompt, *media)]
        for index in range(0 if args.warmup_per_case else 1, args.repeat + 1):
            if index == 0:
                phase, filename, request = "warmup", "warmup.mp4", "warmup"
            else:
                phase = "measured" if args.warmup_per_case or index > 1 else "first_touch"
                filename = "video.mp4" if args.repeat == 1 else f"video-{index:02}.mp4"
                request = f"request-{index:02}"
            yield dict(case=name, attempt=index, task=task, seconds=seconds, resolution=resolution,
                       phase=phase,
                       media=list(map(str, media)), prompt=str(prompt), inputs_sha256=hashes,
                       video=str(directory / filename),
                       boundary=str(directory / request))


def phase_seconds(phases, name):
    rows = [row for row in phases if row["name"] == name]
    return sum(row.get("device_envelope_seconds", row["host_seconds"]) for row in rows) if rows else None


def finish_request(job, samples, baseline, group, first, interval):
    boundary, video = Path(job["boundary"]), Path(job["video"])
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(boundary.glob("rank-*.json"))]
    last = samples[-1]
    result = {key: job[key] for key in ("case", "attempt", "phase", "task", "seconds", "resolution", "inputs_sha256")}
    result.update(group=group, measurement_mode="resident_first" if first else "resident_reused",
                  video=str(video.relative_to(boundary.parent.parent)), status="failed",
                  detail=str((boundary / "result.json").relative_to(boundary.parent.parent)),
                  manual_quality="pending", ranks=ranks, end_memory=last,
                  inference_save_s=max((rank["infer_save_s"] for rank in ranks), default=None),
                  request_wall_s=max((rank["wall_s"] for rank in ranks), default=None),
                  allocated_after_bytes=max((rank["allocated_bytes"] for rank in ranks), default=None),
                  reserved_after_bytes=max((rank["reserved_bytes"] for rank in ranks), default=None),
                  hbm_samples=sum(bool(entry["hbm"]) for entry in samples),
                  host_peak_used_kib=max(entry["host_used_kib"] for entry in samples),
                  process_peak_rss_kib=max(entry["process_rss_kib"] for entry in samples),
                  process_peak_pss_kib=max((entry["process_pss_kib"] for entry in samples
                                            if entry["pss_coverage"][0] and entry["process_pss_kib"] is not None),
                                           default=None))
    running = [entry for entry in samples if entry["phase"] != "before_launch"]
    gaps = [end["monotonic_s"] - begin["monotonic_s"] for begin, end in zip(running, running[1:])]
    result["max_sample_gap_s"] = max(gaps, default=None)
    result["max_sample_duration_s"] = max(entry["sample_duration_s"] for entry in running)
    rank_pids = {rank["pid"] for rank in ranks}
    rank_samples = [
        {pid for pid in rank_pids if entry["processes"].get(pid, {}).get("pss_kib") is not None}
        for entry in running
    ]
    first_active = next((index for index, entry in enumerate(running)
                         if rank_pids <= set(entry["active_pids"])), len(running))
    result["rank_pss_coverage_after"] = len(rank_samples[-1])
    result["monitor_complete"] = len(baseline["hbm"]) == 8 and all(
        entry["hbm"].keys() == baseline["hbm"].keys() and entry["active_pids"]
        for entry in running
    ) and result["rank_pss_coverage_after"] == 8 and all(
        covered == rank_pids for covered in rank_samples[first_active:]
    ) and last["process_pss_kib"] is not None and all(gap <= 2 * interval for gap in gaps)
    result["hbm_peak_mib_by_device"] = {
        card: max(entry["hbm"][card]["used_mib"] for entry in samples if card in entry["hbm"])
        for card in sorted({card for entry in samples for card in entry["hbm"]})
    }
    try:
        if sorted(rank["rank"] for rank in ranks) != list(range(8)) or (
            not first and not all(rank["model_reused"] for rank in ranks)
        ):
            raise RuntimeError("Rank snapshots or model reuse verification are incomplete")
        result["media"] = media_check(video, job["resolution"], job["seconds"])
        if job["resolution"] == "2k":
            sidecars = sorted(video.parent.glob(video.stem + ".metrics.rank*.json"))
            if len(sidecars) != 8 or any(
                json.loads(path.read_text(encoding="utf-8")).get("actual_dit") != {"h3": 20, "refiner": 3}
                for path in sidecars
            ):
                raise ValueError("Missing 2K stage metrics or unexpected actual DiT counts")
            result["stage_metrics"] = [path.name for path in sidecars]
            rank_zero = json.loads(video.with_suffix(".metrics.rank0.json").read_text(encoding="utf-8"))
            phases = rank_zero["phases"]
            result["refiner_dit_s"] = phase_seconds(phases, "refiner.dit")
            result["vae_decode_s"] = phase_seconds(phases, "vae.decode")
        else:
            if any(rank["base_dit_calls"] != 20 for rank in ranks):
                raise ValueError("Expected 20 actual Base DiT calls on every rank")
            phases = ranks[0]["phases"]
            result["vae_decode_s"] = phase_seconds(phases, "vae.decode")
        result["base_dit_s"] = phase_seconds(phases, "h3.dit")
        result["encode_mux_wait_s"] = phase_seconds(phases, "video.encode_and_audio_mux_wait")
        result["model_init_s"] = max(rank["prepare_seconds"] for rank in ranks) if first else None
        result["status"] = "passed"
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    write_json(boundary / "result.json", result)
    return result


def run_group(group, cases, args, publish):
    directory = args.output / "groups" / group
    directory.mkdir(parents=True)
    jobs = list(jobs_for(cases, args))
    plan = directory / "plan.json"
    write_json(plan, jobs)
    first = jobs[0]
    task = first["task"]
    table = args.ref_adaln if task == "r2va" else args.base_adaln
    env = os.environ.copy()
    env.update(PROMPT_FILE=first["prompt"], OUTPUT_PATH=first["video"], BASE_ADALN=str(table),
               H3_BASE_STEPS="20", H3_SEED=str(args.seed), XGEN_H3_BENCH_WORKER=str(Path(__file__).resolve()),
               XGEN_H3_BENCH_PLAN=str(plan))
    if task == "r2va":
        env["BASE_TRANSFORMER"] = env.get("BENCH_REF_TRANSFORMER", "model-cache/base-ref2va")
    command = ["bash", str(LAUNCHER), "npu", task, str(first["seconds"]), first["resolution"], *first["media"]]
    config = dict(group=group, cards=8, sp=8, text_tp=8, phaa=2, base_steps=20, base_schedule_points=21,
                  refiner_steps=3 if first["resolution"] == "2k" else 0, attention="laser", dit_linear="int8",
                  vae="Refiner BF16" if first["resolution"] == "2k" else os.environ.get('H3_VAE_BACKEND', 'native-stream'),
                  adaln=table.name, seed=args.seed)
    config["request_memory_release"] = True
    samples = []
    seen = set()
    completed = 0
    error = None

    def record(log, phase, session=None):
        entry = sample(session, seen)
        entry["phase"] = phase
        samples.append(entry)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        return entry

    with (directory / "memory.jsonl").open("w", encoding="utf-8") as memory, (
        directory / "run.log"
    ).open("w", encoding="utf-8") as log:
        baseline = record(memory, "before_launch")
        config["device_names"] = sorted({card["name"] for card in baseline["hbm"].values()})
        write_json(directory / "config.json", config)
        publish("start", config)
        if len(baseline["hbm"]) != 8 or baseline["idle_cards"] != 8:
            result = dict(group=group, process_wall_s=0, exit_code=None, completed=0,
                          remaining=[f'{job["case"]}/{Path(job["boundary"]).name}' for job in jobs],
                          interrupted_request=None, error="eight_idle_cards_required",
                          before=baseline, after=baseline, residual_pids=[])
            write_json(directory / "result.json", result)
            publish("end", result)
            return result
        start = time.monotonic()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            next_sample = start
            deadline = start + args.request_timeout
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    error = "request_timeout"
                    break
                if completed < len(jobs):
                    job = jobs[completed]
                    boundary = Path(job["boundary"])
                    if (boundary / "ready").exists():
                        record(memory, f'{job["case"]}/{boundary.name}', proc.pid)
                        result = finish_request(job, samples, baseline, group, completed == 0, args.sample_interval)
                        publish("request", result)
                        completed += 1
                        if job["phase"] == "warmup" and result["status"] != "passed":
                            break
                        samples.clear()
                        (boundary / "sampled").touch()
                        next_sample = time.monotonic()
                        deadline = next_sample + args.request_timeout
                if time.monotonic() >= next_sample:
                    next_sample = time.monotonic() + args.sample_interval
                    record(memory, "running", proc.pid)
                time.sleep(0.2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        wall = time.monotonic() - start
        time.sleep(args.exit_settle)
        after = record(memory, "after_exit", proc.pid)

    remaining = [f'{job["case"]}/{Path(job["boundary"]).name}' for job in jobs[completed:]]
    if error is None and proc.returncode and remaining:
        with (directory / "run.log").open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 262144))
            tail = log.read().decode(errors="replace").lower()
        error = "oom" if re.search(r"out of memory|acl error.*207001", tail) else "worker_exit"
    result = dict(group=group, process_wall_s=wall, exit_code=proc.returncode, completed=completed,
                  remaining=remaining, interrupted_request=remaining[0] if remaining and error else None,
                  error=error, before=baseline, after=after, residual_pids=after["active_pids"])
    write_json(directory / "result.json", result)
    publish("end", result)
    return result


def format_size(value, divisor):
    return "—" if value is None else f"{value / divisor:.2f}"


def markdown_event(kind, record):
    if kind == "start":
        chips = ", ".join(record["device_names"]) or "芯片采样缺失"
        return (f'\n## {record["group"]}\n\n'
                f'{record["cards"]} 卡 · {chips} · SP{record["sp"]} · 文本 TP{record["text_tp"]} · PHAA{record["phaa"]}'
                f' · Base DiT {record["base_steps"]} 步 · Refiner {record["refiner_steps"]} 步'
                f' · {record["attention"]} · DiT {record["dit_linear"]} · {record["vae"]}'
                f' · seed {record["seed"]} · AdaLN `{record["adaln"]}`'
                f' · 每条成片后显存回收 {"开启" if record["request_memory_release"] else "关闭"}\n\n'
                '| 用例 | 计时口径 | 模型 | 推理及保存 s | 请求 wall s | 首次初始化 s | Base DiT / Refiner DiT / VAE / 封装等待 s | HBM 峰值 GiB/卡 |'
                ' 整机峰值 GiB | RSS / PSS 峰值 GiB | 结束 PSS GiB | 结束 alloc / reserved GiB/卡 |'
                ' 采样 | 技术检查 | 人工画面与听音 |\n'
                '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n')
    if kind == "request":
        peak_hbm = max(record["hbm_peak_mib_by_device"].values(), default=None)
        pss = record["end_memory"]["process_pss_kib"]
        startup = record.get("model_init_s")
        stages = " / ".join(format_size(record.get(stage), 1) for stage in
                            ("base_dit_s", "refiner_dit_s", "vae_decode_s", "encode_mux_wait_s"))
        target = record["video"].replace(" ", "%20")
        detail = record["detail"].replace(" ", "%20")
        model = "首次" if record["measurement_mode"] == "resident_first" else "复用"
        audit = "通过" if record["status"] == "passed" else "失败"
        monitor = ("完整" if record["monitor_complete"] else "部分缺失") + (
            f' · PSS {record["rank_pss_coverage_after"]}/8 rank'
        )
        phase = {"warmup": "预热", "first_touch": "形状首跑", "measured": "热态测量"}[record["phase"]]
        attempt = f' #{record["attempt"]}' if record["attempt"] else ""
        return (f'| [{record["case"]}{attempt}]({target}) | {phase} | {model} |'
                f' {format_size(record["inference_save_s"], 1)} | {format_size(record["request_wall_s"], 1)} |'
                f' {format_size(startup, 1)} | {stages} | {format_size(peak_hbm, 1024)} |'
                f' {format_size(record["host_peak_used_kib"], 1024**2)} |'
                f' {format_size(record["process_peak_rss_kib"], 1024**2)} /'
                f' {format_size(record["process_peak_pss_kib"], 1024**2)} | {format_size(pss, 1024**2)} |'
                f' {format_size(record["allocated_after_bytes"], 1024**3)} /'
                f' {format_size(record["reserved_after_bytes"], 1024**3)} | {monitor} | [{audit}]({detail}) | 待复核 |\n')
    before, after = record["before"], record["after"]
    if before["hbm"] and before["hbm"].keys() == after["hbm"].keys():
        delta = ", ".join(
            f'{card} {after["hbm"][card]["used_mib"] - before["hbm"][card]["used_mib"]:+d}'
            for card in sorted(before["hbm"])
        )
    else:
        delta = "采样不完整"
    group = record["group"]
    remaining = ", ".join(record["remaining"]) or "无"
    return (f'\n组退出码 {record["exit_code"]}，进程 wall {record["process_wall_s"]:.2f} 秒，'
            f' 诊断 {record["error"] or "无"}，中断请求 {record["interrupted_request"] or "无"}。'
            f'残留 PID {record["residual_pids"] or "无"}。退出后相对启动前 HBM 变化（MiB） {delta}。'
            f' 未产出请求 {remaining}。 [组日志](groups/{group}/run.log) ·'
            f' [内存采样](groups/{group}/memory.jsonl) · [退出快照](groups/{group}/result.json)\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", type=lambda value: Path(value).resolve())
    inputs.add_argument("--manifest", type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve(),
                        help="Parent directory for timestamped runs")
    parser.add_argument("--base-adaln", type=lambda value: Path(value).resolve())
    parser.add_argument("--ref-adaln", type=lambda value: Path(value).resolve())
    parser.add_argument("--suite", choices=("smoke", "acceptance", "images", "multimodal", "all"), default="smoke")
    parser.add_argument("--case", help="One case ID, overrides --suite")
    parser.add_argument("--plan-only", action="store_true", help="Validate and hash inputs without starting workers")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat one case in the same worker group")
    parser.add_argument("--warmup-per-case", action="store_true",
                        help="Run one matching warmup before measuring each case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=10)
    parser.add_argument("--exit-settle", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=1800, help="Seconds per request including first startup")
    args = parser.parse_args()
    if not args.plan_only and not Path("/proc/meminfo").is_file():
        parser.error("Run the benchmark on Linux with /proc and Ascend npu-smi available")
    matrix = list(case_matrix(args.inputs or args.manifest.parent / "inputs"))
    try:
        cases = (list(manifest_inputs(args.manifest, matrix)) if args.manifest else
                 [(*case, args.inputs / f"{case[1]}.txt") for case in matrix])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    selected = ([case for case in cases if case[0] == args.case] if args.case else
                {"smoke": cases[:1], "acceptance": [case for case in cases if case[0] in SMOKE_CASES],
                 "images": cases[:30], "multimodal": cases[30:], "all": cases}[args.suite])
    if not selected or args.repeat < 1 or (args.repeat > 1 and len(selected) != 1):
        parser.error("Select a valid case and use --repeat with one case only")
    if not (args.sample_interval > 0 and args.exit_settle >= 0 and args.request_timeout > 0):
        parser.error("Sampling interval and request timeout must be positive and exit settle nonnegative")
    assets = {file for _, _, _, _, media, prompt in selected for file in (prompt, *media)}
    tables = {}
    try:
        args.hashes = {file: digest(file) for file in sorted(assets)}
        for workflow, table, variants in (
            ("fl2va", args.base_adaln, {"T2VA", "FL2VA"}),
            ("r2va", args.ref_adaln, {"Ref2VA"}),
        ):
            if not any((case[1] == "r2va") == (workflow == "r2va") for case in selected):
                continue
            if table is None and args.plan_only:
                continue
            if table is None:
                raise ValueError(f"Missing AdaLN table for {workflow}")
            tables[workflow] = dict(file=table.name, metadata=check_adaln(table, variants), sha256=digest(table))
    except (OSError, ValueError, KeyError, struct.error) as exc:
        parser.error(str(exc))
    groups = group_cases(selected)
    if args.plan_only:
        print(json.dumps({"groups": {name: [case[0] for case in group] for name, group in groups},
                          "unique_inputs": len(args.hashes), "adaln": tables}, indent=2))
        return 0
    started_at = datetime.now(timezone(timedelta(hours=8)))
    run_id = started_at.strftime("%Y%m%d-%H%M%S-%f")
    args.output = args.output / run_id
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / f"report-{run_id}.md"
    print(f"Benchmark report: {report_path}", flush=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    commit = os.environ.get("XGEN_BENCH_SOURCE_REVISION") or revision.stdout.strip() or "unknown"
    write_json(args.output / f"run-{run_id}.json", {
        "started_at": started_at.isoformat(), "commit": commit,
        "groups": {name: [case[0] for case in cases] for name, cases in groups},
        "sample_interval_s": args.sample_interval, "warmup_per_case": args.warmup_per_case,
        "request_timeout_s": args.request_timeout, "adaln": tables,
        "source_archive_sha256": os.environ.get("XGEN_BENCH_SOURCE_SHA256"),
    })
    columns = ("group", "case", "attempt", "phase", "task", "seconds", "resolution", "measurement_mode", "status",
               "inference_save_s", "request_wall_s", "model_init_s", "base_dit_s", "refiner_dit_s", "vae_decode_s", "encode_mux_wait_s",
               "monitor_complete", "rank_pss_coverage_after", "max_sample_gap_s", "max_sample_duration_s", "hbm_samples",
               "hbm_peak_mib_by_device", "process_peak_rss_kib",
               "process_peak_pss_kib", "host_peak_used_kib", "allocated_after_bytes", "reserved_after_bytes",
               "manual_quality", "video", "error")
    totals = dict(passed=0, failed=0, warmup_passed=0, warmup_failed=0, monitor_gaps=0)
    incomplete = []
    interrupted = {}
    failed = False
    with (args.output / f"summary-{run_id}.csv").open("x", newline="", encoding="utf-8") as summary, (
        report_path.open("x", encoding="utf-8")
    ) as report:
        writer = csv.DictWriter(summary, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        report.write(f"# xGen NPU 离线 benchmark\n\n"
                     f"启动时间 {started_at.isoformat(timespec='seconds')} · "
                     f"提交 `{commit}` · 计划 {len(selected) * args.repeat} 条测量"
                     f" + {len(selected) if args.warmup_per_case else 0} 条预热"
                     f" · HBM/RSS/PSS 间隔 {args.sample_interval:g} 秒。\n\n"
                     "HBM 是物理卡采样，峰值为最忙卡。allocated/reserved 为请求排空后各 rank 最大值。"
                     "首条 wall 含模型初始化，推理及保存时间按现有推理入口统计。技术检查为媒体规格与完整解码。\n")
        report.flush()

        def publish(kind, result):
            nonlocal failed
            report.write(markdown_event(kind, result))
            report.flush()
            if kind == "request":
                writer.writerow(result)
                summary.flush()
                print(result["case"], result["attempt"], result["status"], flush=True)
                failed |= result["status"] != "passed"
                totals[("warmup_" if result["phase"] == "warmup" else "") + result["status"]] += 1
                totals["monitor_gaps"] += not result["monitor_complete"]

        for index, (group, group_selection) in enumerate(groups):
            result = run_group(group, group_selection, args, publish)
            incomplete.extend(result["remaining"])
            if result["interrupted_request"]:
                interrupted[result["interrupted_request"]] = result["error"]
            group_failed = bool(result["exit_code"] or result["remaining"] or result["residual_pids"])
            failed |= group_failed
            if (result["residual_pids"] or result["error"] == "eight_idle_cards_required"
                    or (args.suite == "acceptance" and (failed or totals["monitor_gaps"]))):
                incomplete.extend(case[0] for _, unstarted in groups[index + 1:] for case in unstarted)
                break
        outcome = dict(totals, incomplete=incomplete, interrupted_requests=interrupted,
                       expected_measurements=len(selected) * args.repeat,
                       expected_warmups=len(selected) if args.warmup_per_case else 0)
        write_json(args.output / "final-status.json", outcome)
        report.write(f'\n## 本轮汇总\n\n测量技术通过 {totals["passed"]}/{outcome["expected_measurements"]}，'
                     f'失败 {totals["failed"]}。预热通过 {totals["warmup_passed"]}/{outcome["expected_warmups"]}，'
                     f'失败 {totals["warmup_failed"]}。采样不完整 {totals["monitor_gaps"]} 条。\n\n'
                     f'缺项或中断请求 {", ".join(incomplete) or "无"}。\n')
        if interrupted:
            report.write("\n中断原因 " + ", ".join(f"{case} ({reason})" for case, reason in interrupted.items()) + "。\n")
    return int(failed or totals["monitor_gaps"])


if __name__ == "__main__":
    if os.environ.get("XGEN_H3_BENCH_PLAN") and "LOCAL_RANK" in os.environ:
        worker()
    else:
        raise SystemExit(main())

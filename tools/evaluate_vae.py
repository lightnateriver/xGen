"""Evaluate Wan VAE acceleration through the normal batch inference API."""

import argparse
import csv
from copy import deepcopy
from itertools import product
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
TARGETS = ("none", "encoder", "decoder", "encoder_and_decoder")
SPEED_COLUMNS = (
    "Case_ID", "Resolution", "Frames", "VAE_Lighting_Target",
    "Encode_Time(ms)", "Decode_Time(ms)",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Pass normal xGen options such as --model, --batch_input_dir, "
            "--batch_sample_size and --seed alongside these evaluation options. "
            "Run with torchrun in an installed xGen runtime."
        ),
    )
    parser.add_argument("--eval_mode", choices=("speed", "accuracy"), required=True)
    parser.add_argument("--eval_report_dir", type=Path, required=True)
    parser.add_argument("--eval_resolutions", default="", help="Example: 832x480,1280x720")
    parser.add_argument("--eval_frames", default="", help="Example: 81,49,17")
    parser.add_argument("--eval_targets", default="", help=",".join(TARGETS))
    return parser


def parse_test_matrix(options):
    """Parse the matrix before loading weights; malformed input is an error."""
    if options.eval_mode == "speed":
        resolutions = [(832, 480), (1280, 720)]
        frames = [81, 49, 17]
        targets = list(TARGETS)
    else:
        resolutions = [(832, 480)]
        frames = [81]
        targets = ["none", "encoder_and_decoder"]

    if options.eval_resolutions:
        resolutions = []
        try:
            for value in options.eval_resolutions.split(","):
                width, height = value.lower().strip().split("x")
                resolutions.append((int(width), int(height)))
        except ValueError as error:
            raise ValueError("--eval_resolutions requires WIDTHxHEIGHT pairs") from error
    if any(width <= 0 or height <= 0 for width, height in resolutions):
        raise ValueError("--eval_resolutions dimensions must be positive")

    if options.eval_frames:
        try:
            frames = [int(value.strip()) for value in options.eval_frames.split(",")]
        except ValueError as error:
            raise ValueError("--eval_frames requires integer frame counts") from error
    if any(count <= 0 for count in frames):
        raise ValueError("--eval_frames counts must be positive")

    if options.eval_targets:
        targets = [value.strip() for value in options.eval_targets.split(",")]
    if any(target not in TARGETS for target in targets):
        raise ValueError("--eval_targets must be selected from " + ",".join(TARGETS))
    return resolutions, frames, targets


def evaluation_output_path(report_dir, mode, args, case_dir):
    combination = Path(report_dir) / f"{args.width}x{args.height}_{args.frames}f"
    if mode == "accuracy":
        folder = "original" if args.vae_lightning == "none" else f"target-{args.vae_lightning}"
    else:
        folder = f"lighting-{args.vae_lightning}"
    return combination / folder / f"{Path(case_dir).name}.mp4"


def prepare_reports(report_dir, mode, case_dirs):
    report_dir.mkdir(parents=True, exist_ok=True)
    if mode == "speed":
        with (report_dir / "report_speed.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(SPEED_COLUMNS)
        return

    with (report_dir / "prompt.txt").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for case_dir in case_dirs:
            case_dir = Path(case_dir)
            prompt = (case_dir / "prompt.txt").read_text(encoding="utf-8").strip().replace("\n", "")
            writer.writerow([f"{case_dir.name}.mp4", prompt])
    (report_dir / "任务描述.txt").write_text(
        "任务背景：VAE 并行加速输出对照\n"
        "每个分辨率和帧数组合单独保存；original 为未加速输出，target-* 为加速输出。\n"
        "本工具生成对照视频和提示词清单，不自动计算精度指标。\n",
        encoding="utf-8",
    )


def append_speed_report(report_dir, args, case_id, timing_info):
    with (report_dir / "report_speed.csv").open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow([
            case_id, f"{args.width}x{args.height}", args.frames, args.vae_lightning,
            timing_info.vae_last_encode_time, timing_info.vae_last_decode_time,
        ])


def run_evaluation(manager, args, options, case_dirs, *, rank=0, timing_info=None):
    """Keep matrix scheduling, comparison layouts and reports outside the Manager."""
    resolutions, frames, targets = parse_test_matrix(options)
    case_dirs = list(case_dirs)
    if not case_dirs:
        raise ValueError("No batch cases found (each case needs image.jpg and prompt.txt)")
    if options.eval_mode == "speed" and timing_info is None:
        raise ValueError("Speed evaluation requires VAE timing information")
    report_dir = Path(options.eval_report_dir)
    if rank == 0:
        prepare_reports(report_dir, options.eval_mode, case_dirs)

    def output_path(case_args, case_dir):
        return evaluation_output_path(report_dir, options.eval_mode, case_args, case_dir)

    def record_result(case_args, case_id):
        append_speed_report(report_dir, case_args, case_id, timing_info)

    failures = []
    for (width, height), count, target in product(resolutions, frames, targets):
        batch_args = deepcopy(args)
        batch_args.width, batch_args.height = width, height
        batch_args.frames = count
        batch_args.vae_lightning = target
        batch_args.enable_vae_time_count = options.eval_mode == "speed"
        if rank == 0:
            logger.info("VAE %s: %sx%s, %s frames, %s", options.eval_mode, width, height, count, target)
        failed = manager.infer_batch(
            batch_args,
            case_dirs=case_dirs,
            output_path_fn=output_path,
            on_result=record_result if options.eval_mode == "speed" else None,
        )
        failures.extend(
            dict(failure, width=width, height=height, frames=count, vae_lightning=target)
            for failure in failed
        )
    if rank == 0:
        logger.info("VAE evaluation finished: %s; %s failures", report_dir, len(failures))
        for failure in failures:
            logger.error("Failed evaluation case: %s", failure)
    return failures


def main(argv=None):
    parser = build_parser()
    options, inference_argv = parser.parse_known_args(argv)
    try:
        parse_test_matrix(options)
    except ValueError as error:
        parser.error(str(error))

    # Help and evaluation argument validation do not initialize the model runtime.
    from x_diffusers import InferenceManager, init_env, parse_args
    from x_base import infer_info, list_cases
    import torch.distributed as dist

    args = parse_args(argv=inference_argv)
    if not args.batch_input_dir:
        parser.error("--batch_input_dir is required")
    if not args.model.startswith("Wan2."):
        parser.error("This tool evaluates Wan2.x VAE acceleration")
    case_dirs = list_cases(args.batch_input_dir, sample_size=args.batch_sample_size, seed=args.seed)
    if not case_dirs:
        parser.error("No batch cases found (each case needs image.jpg and prompt.txt)")
    init_env()
    rank = dist.get_rank() if dist.is_initialized() else 0
    failures = run_evaluation(
        InferenceManager(), args, options, case_dirs, rank=rank, timing_info=infer_info,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The vLLM team.
# Environment collection adapted for xGen from:
# https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py
# https://github.com/vllm-project/vllm-ascend/blob/1e3c1e76bf9e129bbb49878fa8614886965e418d/collect_env.py
# License text: x_base/x_base/operator/h3_attention/LICENSE.upstream

"""Collect xGen CUDA or Ascend environment information without loading a model."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess


COMMON_ENV = (
    "XGEN_DEVICE", "RANK", "LOCAL_RANK", "WORLD_SIZE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)
CUDA_ENV = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_MAX_CTAS", "NCCL_LAUNCH_ORDER_IMPLICIT",
    "XGEN_FASTVIDEO_RECIPE", "XGEN_SPARGE_TOPK", "XGEN_SPARGE_HEAD_CHUNK",
    "XGEN_PHAA_FP8", "XGEN_PHAA_GATHER_PRIORITY", "XGEN_PHAA_DEVICE_JOIN",
    "XGEN_PHAA_ASYNC_EXCHANGE", "XGEN_PHAA_DIRECT_RETURN", "XGEN_PHAA_FP8_ROW_TILING",
    "XGEN_PHAA_CHUNK_ORDER", "XGEN_PHAA_DUAL_COMM", "XGEN_PHAA_FUSED_RETURN_PACK",
    "XGEN_PHAA_DIRECT_GATHER", "XGEN_H3_CUDA_VAE_TILES", "XGEN_H3_FUSED_FF_AMAX",
    "XGEN_H3_FUSED_INDEXED_AFFINE", "XGEN_H3_FUSED_PARTIAL_ROPE", "XGEN_H3_FUSED_QK_RMS",
    "XGEN_H3_FUSED_SWIGLU", "XGEN_H3_NVFP4_FUSED_AMAX",
)
NPU_ENV = (
    "ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF", "HCCL_BUFFSIZE",
    "HCCL_CONNECT_TIMEOUT", "HCCL_EXEC_TIMEOUT", "HCCL_SOCKET_IFNAME",
    "TASK_QUEUE_ENABLE", "COMBINED_ENABLE", "MEMORY_FRAGMENTATION", "INF_NAN_MODE_FORCE_DISABLE",
)
ENV_PRESENCE_ONLY = (
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "XGEN_PYTHONPATH", "FFMPEG_PATH", "FFPROBE_PATH",
    "CUDA_HOME", "CUDA_PATH", "TRITON_CACHE_DIR", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "MASTER_ADDR",
)


def print_environment(backend):
    print("\nEnvironment variables")
    backend_vars = {"cuda": CUDA_ENV, "npu": NPU_ENV}
    variables = COMMON_ENV + backend_vars.get(backend, CUDA_ENV + NPU_ENV)
    visibility = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
    for name in variables:
        value = os.environ.get(name)
        if value is None:
            if name == "XGEN_DEVICE" or name in visibility:
                print(f"{name}=<unset>")
            continue
        if name == "CUDA_VISIBLE_DEVICES":
            value = ",".join(
                "<device-id>" if device.strip().startswith(("GPU-", "MIG-")) else device
                for device in value.split(",")
            )
        print(f"{name}={ascii(value) if value and not value.isprintable() else value or '<empty>'}")

    print("Paths and endpoint (presence only)")
    for name in ENV_PRESENCE_ONLY:
        if name in os.environ:
            state = "<set>" if os.environ[name] else "<empty>"
            print(f"{name}={state}")


def print_cpu_topology():
    print("\nCPU and NUMA")
    if platform.system() != "Linux":
        print("Process CPU affinity and NUMA topology: unavailable on this OS")
        return

    fields = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(":")
            if name in ("Cpus_allowed_list", "Mems_allowed_list"):
                fields[name] = value.strip()
    except OSError:
        pass
    print(f"Process CPU affinity: {fields.get('Cpus_allowed_list', 'unavailable')}")
    print(f"Process allowed NUMA memory nodes: {fields.get('Mems_allowed_list', 'unavailable')}")

    nodes = Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")
    nodes = sorted(nodes, key=lambda path: int(path.parent.name[4:]))
    print(f"System NUMA nodes visible in sysfs: {len(nodes) if nodes else 'unavailable'}")
    for path in nodes:
        try:
            cpus = path.read_text(encoding="ascii").strip()
        except OSError:
            cpus = "unavailable"
        print(f"NUMA {path.parent.name[4:]} CPUs: {cpus or '(none)'}")


def command_output(*command, first_line=False):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=15,
        )
    except OSError as error:
        return f"Unavailable ({error})"
    except subprocess.TimeoutExpired:
        return "Timed out after 15 seconds"
    if result.returncode:
        return f"Exit code {result.returncode}\n{(result.stderr or result.stdout).strip()}"
    lines = result.stdout.strip().splitlines()
    if not lines:
        return "No output"
    return lines[0] if first_line else "\n".join(lines)


def get_cann_info():
    roots = [Path(value) for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
             if (value := os.environ.get(name))]
    if atc := shutil.which("atc"):
        roots.extend(Path(atc).resolve().parents[:3])
    arch = platform.machine()
    for root in dict.fromkeys(roots):
        for path in (root / f"{arch}-linux" / "ascend_toolkit_install.info",
                     root / "ascend_toolkit_install.info", root / "version.info"):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            fields = [line.strip() for line in lines
                      if line.partition("=")[0].strip().lower() in {"version", "innerversion", "compatible_version"}]
            if fields:
                return "\n".join(fields)
    return "Not found via ASCEND_HOME_PATH, ASCEND_TOOLKIT_HOME or atc on PATH"


def probe_device(torch, backend):
    try:
        if backend == "npu":
            import torch_npu  # noqa: F401  # Registers torch.npu.
        elif getattr(torch.version, "hip", None):
            return 0, "PyTorch is built for ROCm"
        device = getattr(torch, backend)
        return (device.device_count() if device.is_available() else 0), None
    except Exception as error:
        return 0, f"{type(error).__name__}: {error}"


def select_backend(torch, requested):
    npu_hint = (os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
                or shutil.which("npu-smi") or shutil.which("atc"))
    cuda_hint = shutil.which("nvidia-smi") or (torch is not None and getattr(torch.version, "cuda", None))
    order = (requested,) if requested else ("npu", "cuda") if npu_hint else ("cuda", "npu")
    probes = {}
    for backend in order:
        probes[backend] = probe_device(torch, backend) if torch is not None else (0, None)
        if probes[backend][0]:
            return backend, probes
    return requested or ("npu" if npu_hint else "cuda" if cuda_hint else None), probes


def print_packages(backend):
    common = ("x_base", "x_diffusers", "torch", "torchvision", "diffusers", "transformers", "numpy", "pillow")
    extras = {
        "cuda": ("triton", "sageattention", "comfy-kitchen", "spargeattn"),
        "npu": ("torch-npu", "triton-ascend", "mindiesd"),
    }
    for package in common + extras.get(backend, ()):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        print(f"{package}: {installed}")


def print_cuda_info(torch, count):
    compiled = getattr(torch.version, "cuda", None) if torch is not None else None
    print(f"PyTorch CUDA build: {compiled or 'unavailable'}")
    if count:
        libraries = (("cuDNN", lambda: torch.backends.cudnn.version()),
                     ("NCCL", lambda: torch.cuda.nccl.version()))
        for label, get_version in libraries:
            try:
                library_version = get_version()
            except (AttributeError, OSError, RuntimeError):
                library_version = None
            print(f"{label}: {library_version or 'unavailable'}")
    gpu_fields = "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu"
    print(f"\nNVIDIA devices and driver\n{command_output('nvidia-smi', gpu_fields, '--format=csv')}")
    print(f"\nNVIDIA topology\n{command_output('nvidia-smi', 'topo', '-m')}")
    print(f"\nCUDA toolkit (nvcc)\n{command_output('nvcc', '--version')}")


def main():
    print(f"OS: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Logical CPUs: {os.cpu_count()}")
    print(f"Python: {platform.python_version()}")
    print_cpu_topology()
    requested = os.environ.get("XGEN_DEVICE")
    if requested is not None and requested not in ("cuda", "npu"):
        print("XGEN_DEVICE must be npu or cuda")
        return 1
    try:
        import torch
    except Exception as error:
        torch = None
        print(f"PyTorch import failed ({type(error).__name__}): {error}")

    backend, probes = select_backend(torch, requested)
    print(f"Backend: {backend or 'undetected'} ({'XGEN_DEVICE' if requested else 'auto'})")
    print_environment(backend)
    print("\nPython packages")
    print_packages(backend)
    for tool, variable in (("ffmpeg", "FFMPEG_PATH"), ("ffprobe", "FFPROBE_PATH")):
        print(f"{tool}: {command_output(os.environ.get(variable, tool), '-version', first_line=True)}")

    if backend is None:
        for candidate, (_, error) in probes.items():
            if error:
                print(f"{candidate.upper()} probe: {error}")
        return 1
    count, error = probes[backend]
    print(f"\nPyTorch {backend.upper()} available: {count > 0}")
    print(f"Visible {backend.upper()} count: {count}")
    if error:
        print(f"Runtime check failed: {error}")
    if backend == "npu":
        print(f"\nCANN\n{get_cann_info()}")
        print(f"\nnpu-smi info\n{command_output('npu-smi', 'info')}")
    else:
        print_cuda_info(torch, count)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

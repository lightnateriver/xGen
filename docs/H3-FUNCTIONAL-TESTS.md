# H3 功能回归

功能回归和性能、画质测试分开。1次实际DiT适合检查参数、模型复用、并行调用、编码和显存容量；输出通常不具备可评审的生成质量。20步性能表不能用1步结果代替。

## 不加载权重的检查

在准备好依赖的Linux环境、仓库根目录执行。保留Git历史，旧路径AST检查需要上一发布版的commit；源码压缩包缺少Git历史时会明确跳过这些检查。

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q x_diffusers/tests/test_h3_cuda_options.py x_diffusers/tests/test_h3_cuda_precision_legacy.py x_diffusers/tests/test_h3_launch_profiles.py server/video/test_cuda_precision.py
```

检查包括：旧CUDA/NPU/Wan配置与模型重建语义、显式开关优先级、JSON false、非法组合、启动状态不可被请求覆盖、HTTP 5/15秒有效请求入队以及越界请求拒绝。Flask测试客户端真实运行路由和队列代码，不加载模型，不能称为真实GPU HTTP成片测试。

在CPU上检查八rank的数学和通信协议：

```bash
CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=8 tools/validate_h3_io_options.py
CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=8 tools/validate_h3_pair_parallel.py
```

前者覆盖本地输入和SAP输出的四种组合。后者检查TP2块和BF16/FP8协议；量化协议oracle逐字节相同不等于与BF16模型逐字节相同。

## 真实设备上的连续请求

`tools/validate_h3_runtime.py` 复用原InferenceManager、推理及编码链路。计划只允许修改请求输入，不能在请求里修改启动精度或并行配置。它逐rank记录实际DiT次数、峰值和结束内存、模型构建次数，并用FFprobe和FFmpeg检查视频尺寸、帧数、音轨和完整解码。不会自动重试，也不会覆盖已有case。

创建 `smoke-plan.json`。参考图由操作者提供，路径相对于启动目录：

```json
[
  {
    "name": "reference-request-a",
    "task_type": "ref2va",
    "prompt": "Two coworkers talk in a workshop.",
    "reference_paths": ["inputs/ref01.png"],
    "frames": 124,
    "expected_frames": 124,
    "num_inference_steps": 1
  },
  {
    "name": "reference-request-b",
    "task_type": "ref2va",
    "prompt": "The coworkers stand and walk together.",
    "reference_paths": ["inputs/ref02.png"],
    "frames": 243,
    "expected_frames": 243,
    "num_inference_steps": 1,
    "expect_reuse": true
  }
]
```

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 tools/validate_h3_runtime.py --case-plan smoke-plan.json --result-dir outputs/smoke-run --model MiniMax-H3 --task_type ref2va --pretrained_model_name_or_path weights/h3-ref2va --dtype bf16 --sp 8 --text_encoder_tp 8 --phaa_num 2 --h3_cuda_vae_sp --h3_cuda_preset sage-fp8-full-parallel
```

每次测试使用新的结果目录。NPU使用原NPU推理参数，不传CUDA预设；T2VA/I2VA使用对应原模型与启动配置。`expected_frames`按输出规范显式填写：已测离线5/10/15秒对应124/243/361帧；在线保存帧数有独立契约。

真实HTTP测试应使用 [在线教程](H3-ONLINE.md) 启动独立实例，提交连续请求并下载完整文件，记录模型只加载一次、请求条件重新计算，以及关闭实例后全部worker退出。CPU路由测试、离线Manager连续请求与真实HTTP测试应分别报告。

[发布验证结果](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda5090-full-parallel.md)

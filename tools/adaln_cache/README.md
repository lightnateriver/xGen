# AdaLN 预计算缓存工具

## 概述

MiniMax-H3 模型中 AdaLN（Adaptive Layer Norm）相关参数占 13.04B（39.4%），
仅在推理时用于根据 timestep 生成调制参数。由于 timestep schedule 是确定性的，
这些参数可以离线预计算并缓存，从而在推理时跳过 AdaLN 计算、卸载对应权重。

本工具提供两个功能，可独立或组合使用：

| 功能 | CLI 参数 | 说明 |
|------|---------|------|
| 预计算缓存 | `--output` | 生成 safetensors 缓存文件，推理时跳过 adaln_proj 计算 |
| 剥离权重 | `--strip_output` | 从 checkpoint 中删除 AdaLN 相关权重，减小模型体积 |

## 优化效果

| 阶段 | 模型体积 | 缓存体积 | 说明 |
|------|---------|---------|------|
| 原始 | ~62 GB | — | 完整权重 |
| Step 1: 仅缓存 | ~62 GB | ~1.36 GB | 推理时跳过计算，权重仍在 |
| Step 2: 缓存 + 剥离 | ~36 GB | ~1.36 GB | 卸载 AdaLN 权重，节省显存 |

## 前置条件

- Python 3.8+
- PyTorch
- safetensors
- diffusers（需支持 `ModularPipeline`、`MiniMaxH3Scheduler`、`enable_group_offload`）

> **资源需求**：单卡即可运行。脚本使用 `enable_group_offload` 将 transformer blocks 保留在 CPU，
> 每次仅搬 1 个 block（~1.2GB）到 device 计算，峰值显存 ~1.3GB。

## 使用方法

修改 `run_adaln_cache.sh` 顶部的路径变量后运行：

```bash
bash tools/adaln_cache/run_adaln_cache.sh
```

默认执行预计算缓存 + 剥离权重（Step 2）。Ref2VA 改 `TRANSFORMER_PATH` 即可。

### 直接调用 Python

```bash
python tools/adaln_cache/precompute_adaln_cache.py \
    --model_path /data/models/minimax/FL2VA/transformer \
    --steps 50 --video_shift 12.0 --audio_shift 3.0 \
    --output adaln_cache_50step.safetensors \
    --strip_output /data/models/minimax/FL2VA/transformer_adaln_cache
```

## CLI 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | (必填) | transformer checkpoint 目录路径（含 model-*.safetensors） |
| `--steps` | int | 50 | 采样步数 |
| `--video_shift` | float | 12.0 | 视频 scheduler shift |
| `--audio_shift` | float | 3.0 | 音频 scheduler shift |
| `--output` | str | None | 缓存文件输出路径。指定则生成缓存 |
| `--strip_output` | str | None | 精简模型输出目录。指定则剥离权重 |
| `--model_variant` | str | FL2VA | 模型变体（FL2VA / Ref2VA） |

> `--output` 和 `--strip_output` 至少指定一个，可以只指定其中一个。

## 推理集成

预计算完成后，在推理时通过以下 CLI 参数启用：

### Step 1: 仅使用缓存（权重不卸载）

```bash
python run.py \
    --model MiniMax-H3 \
    --pretrained_model_name_or_path /data/models/minimax \
    --task_type fl2va \
    --steps 50 \
    --adaln_cache_file /data/models/minimax/adaln_cache/adaln_cache_fl2va_50step.safetensors
```

### Step 2: 使用缓存 + 剥离权重

```bash
python run.py \
    --model MiniMax-H3 \
    --pretrained_model_name_or_path /data/models/minimax \
    --task_type fl2va \
    --steps 50 \
    --adaln_cache_file /data/models/minimax/adaln_cache/adaln_cache_fl2va_50step.safetensors \
    --adaln_stripped_weights
```

> `--adaln_stripped_weights` 要求同时指定 `--adaln_cache_file`。
> 启用后会校验 checkpoint 确实缺少 AdaLN 权重，并释放对应显存。

### 推理参数说明

| 参数 | 说明 |
|------|------|
| `--adaln_cache_file` | 预计算缓存文件路径。指定后启用 AdaLN 文件缓存模式 |
| `--adaln_stripped_weights` | 标记 checkpoint 已剥离 AdaLN 权重。启用校验 + 显存释放 |
| `--adaln_offload` | （未实现）预留给未来权重 offload 功能 |

## 缓存文件格式

```
metadata:
  num_steps, video_shift, audio_shift, model_variant, num_blocks

tensors per step:
  step{idx}_timesteps  → [M] float32    (该 step 的唯一 timestep 列表)
  step{idx}_block{j}   → [M, 96768] bf16 (adaln_proj 原始输出，pre-chunk)
  step{idx}_final      → [M, 10752] bf16 (norm_out.linear 原始输出，pre-chunk)
```

## 注意事项

1. **采样参数必须匹配**：预计算时使用的 `--steps`、`--video_shift`、`--audio_shift` 必须与推理时一致，否则缓存未命中会报错（权重已剥离时无法回退计算）。
2. **SP 场景**：8 卡 Ulysses SP 下每卡独立加载缓存，无需额外处理。
3. **剥离权重后**：精简模型目录包含完整的非 AdaLN 权重 + config 文件，可直接用于推理（需配合 `--adaln_cache_file`）。

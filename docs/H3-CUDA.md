# MiniMax-H3 CUDA 部署与启动

使用 `XGEN_DEVICE=cuda` 启动现有 CLI 或 HTTP 服务。省略该变量时选择 NPU。设备由启动环境决定，HTTP 请求沿用现有生成参数。

## 环境和模型

在运行 xGen 的环境中，可执行 `XGEN_DEVICE=cuda python tools/collect_env.py` 检查 GPU、驱动和 CUDA 依赖。脚本也支持 [自动识别设备](../tools/README.md#cuda-和-npu-环境采集)。

准备支持 RTX 5090 的 PyTorch CUDA、Triton、兼容 MiniMax-H3 的 Diffusers 和 Transformers。NVFP4 需要 Comfy-Kitchen CUDA 扩展，Sage 和 Sparge 分别需要对应 CUDA 扩展。Triton 首次编译需要与运行时 Python 版本匹配的开发头文件和可写的缓存目录。安装支持 H.264 和 AAC 的 FFmpeg。部署示例使用本地模型组件和权重。

```bash
export PYTHONPATH="./x_base:./x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=./weights/h3
export NVFP4_CHECKPOINT=./weights/nvfp4/minimax_h3_fl2va_pruned_nvfp4.safetensors
export FIRST_FRAME=./inputs/first-frame.png
export PROMPT_FILE=./inputs/prompt.txt
```

`MODEL_PATH` 指向含 `modular_model_index.json` 的 FL2VA 组件目录。NVFP4 checkpoint 需与该目录的 Transformer 配置匹配。CUDA 打包权重已验证的工作流为 FL2VA。

FL2VA 组件来自 [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3) 或 [ModelScope](https://modelscope.cn/models/MiniMax/MiniMax-H3)。保留 `transformer/config.json`、文本编码器、音视频 VAE 和调度组件。下载实测的 NVFP4 权重：

```bash
hf download lilcheaty/MiniMax-H3-NVFP4 minimax_h3_fl2va_pruned_nvfp4.safetensors \
  --revision 8c5abfed61e1b6a170240792b65253fba1a65b7b --local-dir ./weights/nvfp4
```

## 768P 离线

```bash
GPUS=8 ATTENTION_BACKEND=sparge XGEN_SPARGE_TOPK=0.5 XGEN_PHAA_FP8=1 \
  bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh \
  --h3_nvfp4_resident --h3_cuda_vae_sp --phaa_num 2
```

默认请求 124 帧，24 fps，51 个调度点对应 50 次 DiT。可选 `ATTENTION_BACKEND=sage` 使用 5090 上的 INT8 Q/K 和 FP8 P×V。FP8 PHAA 同时控制去程和回程。关闭该环境开关即可恢复常规通信。

`XGEN_PHAA_DEVICE_JOIN=1` 可在 CUDA H3 的单通信流 FP8 PHAA 中试用 GPU 流间等待。它默认关闭，且不能与 `XGEN_PHAA_DUAL_COMM=1` 组合。流间等待可能延后临时张量的回收，显存峰值和时延收益需按实际请求验证。取消设置该变量即可恢复原来的主机同步。

`XGEN_PHAA_ASYNC_EXCHANGE=1` 是独立的实验开关，要求同时设置 `XGEN_PHAA_FP8=1` 和 `XGEN_PHAA_DEVICE_JOIN=1`，并使用提供 `Work.block_current_stream()` 的 PyTorch。它保留分块提交顺序，在计算流消费每块时等待并解包。多个在途 FP8 缓冲区可能增加显存峰值。时延和显存收益需实测。关闭异步及下列依赖选项后，继续使用 v0.3-cuda 的同步 FP8 打包和解包。再关闭 `XGEN_PHAA_DEVICE_JOIN` 即恢复 v0.3-cuda 的主机等待。

`XGEN_PHAA_DIRECT_RETURN=1` 默认关闭，依赖 `XGEN_PHAA_ASYNC_EXCHANGE=1`。它将回程 FP8 分块直接解包到最终 head 位置，省去回程输出拼接。关闭该变量即可恢复原有分块输出布局。

`XGEN_PHAA_FP8_ROW_TILING=1` 默认关闭，依赖 `XGEN_PHAA_ASYNC_EXCHANGE=1`。它在同一个 Triton 程序中处理四行，保留逐行 FP8 scale 和通信格式。

`XGEN_PHAA_FUSED_RETURN_PACK=1` 默认关闭，依赖 `XGEN_PHAA_DIRECT_RETURN=1`。回程打包直接读取 attention 输出，核内完成布局转换和尾部补零，沿用原通信格式。

`XGEN_PHAA_DIRECT_GATHER=1` 默认关闭，依赖 `XGEN_PHAA_ASYNC_EXCHANGE=1`。Q/K/V 解包时直接写入 attention 的连续布局并裁去尾部 padding。

`XGEN_H3_FUSED_INDEXED_AFFINE=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它融合 AdaLN 索引、逐元素调制和门控残差，保留原 BF16 中间舍入。此开关与整块编译互斥。

`XGEN_H3_FUSED_PARTIAL_ROPE=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它融合 Q/K 的 partial RoPE，保留原 BF16 舍入和未旋转通道。此开关与整块编译互斥。

`XGEN_H3_FUSED_QK_RMS=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它直接读取 Q/K 投影视图，省去 RMSNorm 前的连续化拷贝。同时启用 `XGEN_H3_FUSED_PARTIAL_ROPE=1` 时，支持的输入会在同一个核中完成归一化和旋转。此开关与整块编译互斥。

`XGEN_H3_NVFP4_FUSED_AMAX=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它在 Triton 中归约 BF16 activation 的分块绝对值峰值，省去完整绝对值张量。此开关与整块编译互斥。

`XGEN_H3_FUSED_SWIGLU=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它在一个核内完成 gate-first SiLU 和门控乘法，保留 SiLU 输出的 BF16 舍入。此开关与整块编译互斥。

`XGEN_H3_FUSED_FF_AMAX=1` 默认关闭，用于常驻 NVFP4 CUDA H3。它在 SwiGLU 写出 BF16 激活时同时计算分块峰值，供 FFN 下投影的 NVFP4 量化复用。此开关与整块编译互斥。

`XGEN_H3_CUDA_VAE_TILES=1` 默认关闭，依赖 `--h3_cuda_vae_sp`。它按空间 tile 分配解码任务，继续复用 VAE 的时间分块和重叠融合。每个 rank 保留完整输出，显存和精度需按分辨率验证。

C4 预设复用 xGen PHAA。5 秒预设已完成八卡端到端验证，15 秒预设已核对启动参数。

| 预设 | 运行方式 | 帧数 | 分块顺序 | NCCL_MAX_CTAS |
| --- | --- | ---: | --- | ---: |
| 5 秒 | `GPUS=8 XGEN_FASTVIDEO_RECIPE=5s bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh` | 124 | 0,1,2,3 | 2 |
| 15 秒 | `GPUS=8 XGEN_FASTVIDEO_RECIPE=15s bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh` | 345 | 3,0,1,2 | 1 |

预设同时选择文本 TP8、DiT 常驻、Sparge、FP8 PHAA 和 gather-priority。CUDA 文本 TP8 要求八个分布式 rank。没有选择预设时，启动脚本使用文本 TP1。

### v0.3.1 5 秒实验配方

已实测的 5 秒优化组合可以在现有预设上显式开启。它要求单通信流，启动前请确认未设置 `XGEN_PHAA_DUAL_COMM=1`。

```bash
GPUS=8 XGEN_FASTVIDEO_RECIPE=5s XGEN_SPARGE_TOPK=0.5 \
  XGEN_PHAA_DEVICE_JOIN=1 XGEN_PHAA_ASYNC_EXCHANGE=1 \
  XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1 \
  XGEN_H3_FUSED_INDEXED_AFFINE=1 XGEN_H3_FUSED_PARTIAL_ROPE=1 XGEN_H3_FUSED_QK_RMS=1 \
  XGEN_H3_CUDA_VAE_TILES=1 XGEN_H3_FUSED_FF_AMAX=1 \
  bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh --h3_cuda_vae_sp
```

该组合在 8 张 RTX 5090 上生成并保存耗时 54.10 秒，含进程启动和模型加载 84.11 秒。实测使用已有 Triton 磁盘缓存，每个进程只生成一条视频。解码后的画面和音频哈希均与重构前的同输入参考成片一致。历史热测、显存和回归范围见 [性能记录](performance/h3-cuda.md#v031-双设备冒烟)。

## 在线与可选 2K

原 HTTP 服务通过 `/generate`、`/status/<task_id>` 和 `/download/<task_id>` 提供在线推理。BF16 Base 的启动和请求示例见 [在线服务指南](H3-ONLINE.md#cuda-base)。

### MaaS 在线服务

在仓库根目录运行。准备支持 CUDA H3 的运行环境、FFmpeg 和 FFprobe，以及匹配的 FL2VA 模型组件和打包 NVFP4 权重。下面以早期八卡 HTTP 实测的 NVFP4/Sparge 配方为例。MaaS 模式固定 5 秒、768P、16:9 和 50 次实际 DiT。

使用 [MaaS 示例配置](../example/diffusers/configs/h3_maas.example.json)。如果权重放在其他目录，复制配置并修改路径。下面的环境变量开启实测使用的通信、DiT 和 VAE 优化。基础启动方式见 [在线服务指南](H3-ONLINE.md#cuda-maas)。

```bash
export XGEN_DEVICE=cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./x_base:./x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false NCCL_MAX_CTAS=2 XGEN_SPARGE_TOPK=0.5
export XGEN_PHAA_FP8=1 XGEN_PHAA_GATHER_PRIORITY=1 XGEN_PHAA_DEVICE_JOIN=1
export XGEN_PHAA_ASYNC_EXCHANGE=1 XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1
export XGEN_H3_FUSED_INDEXED_AFFINE=1 XGEN_H3_FUSED_PARTIAL_ROPE=1 XGEN_H3_FUSED_QK_RMS=1
export XGEN_H3_FUSED_FF_AMAX=1 XGEN_H3_CUDA_VAE_TILES=1
torchrun --standalone --nproc_per_node=8 --max_restarts=0 server/video/video_server.py \
  --init --maas-fl2va-input --config example/diffusers/configs/h3_maas.example.json \
  --host 127.0.0.1 --port 5001 \
  --maas-queue-capacity 10 --maas-queue-timeout-s 600
```

在另一个终端准备一条预热请求。示例使用自备 PNG，提示词保存在 UTF-8 文本文件。

```bash
BASE_URL=http://127.0.0.1:5001
python - <<'PY' > maas-warmup.json
import base64
import json
from pathlib import Path

image = base64.b64encode(Path("inputs/first-frame.png").read_bytes()).decode("ascii")
prompt = Path("inputs/prompt.txt").read_text(encoding="utf-8").strip()
print(json.dumps({"id": "warmup-001", "input": {
    "prompt": prompt, "first_frame": "data:image/png;base64," + image,
    "seed": 42, "enable_audio": True,
}}))
PY
curl --fail-with-body -sS -H 'Content-Type: application/json' \
  --data-binary @maas-warmup.json "$BASE_URL/v1/tasks"
curl -sS "$BASE_URL/v1/tasks/warmup-001"
```

重复查询任务直到 `status=succeeded`。随后检查 `/ready`，等它返回 HTTP 200 再接入业务流量。正式请求用新的 ID，提交和查询方式相同。成功后可用下列命令下载。

```bash
curl --fail-with-body -sS "$BASE_URL/ready"
curl --fail-with-body -sS -o warmup.mp4 "$BASE_URL/download/warmup-001"
```

关掉 `--maas-fl2va-input` 即回到原服务 API。探针、错误码、客户端计时和网关部署边界见 [MaaS API](../server/video/MAAS-FL2VA-API.md)。

### 实验性 CUDA 2K Refiner

使用现有 H3 Refiner adapter 和视频服务。当前 CUDA 路径要求 FL2VA、SP8、
BF16 配置和 NVFP4 Base 权重，由 Refiner 统一管理阶段驻留。
先安装可执行的 FFmpeg（含 libx264 和 AAC），可通过 `FFMPEG_PATH` 指定。
使用部署教程的 Refiner 源包和预处理命令，在 `XGEN_DEVICE=cuda` 环境下
运行 `h3_regenerate.prepare`。预处理输出单独存放，供后续推理复用。

在线服务在现有 JSON 中加入以下配置。路径示例以启动目录为基准。

```json
{
  "model": "MiniMax-H3",
  "task_type": "fl2va",
  "resolution": "2k",
  "pretrained_model_name_or_path": "weights/h3",
  "h3_nvfp4_checkpoint": "weights/nvfp4/minimax_h3_fl2va_pruned_nvfp4.safetensors",
  "h3_refiner": true,
  "h3_refiner_weights": "weights/refiner",
  "h3_refiner_prepared": "model-cache/refiner-cuda",
  "sp": 8,
  "text_encoder_tp": 1,
  "phaa_num": 2,
  "atten": "none",
  "num_inference_steps": 51,
  "matmul_quant": "none",
  "vae_matmul_quant": "none",
  "vae_lightning": "none",
  "rope_fused": false
}
```

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 server/video/video_server.py \
  --init --config config-cuda-2k.json --host 127.0.0.1 --port 29655
```

通过 `/generate` 提交 `task_type=fl2va`、`resolution=2k`、`duration=5`、
`first_frame`、`prompt` 和 `seed`，用 `/status/<task_id>` 查询结果。
离线使用同一套模型参数，给现有推理入口传入 `--h3_refiner`、
`--width 2560 --height 1440 --frames 124 --save_fps 24` 和首帧。

已验证 8 张 RTX 5090 上的 5 秒 2K 离线推理和两次串行在线请求。
在线原生 attention 也验证了同一服务从 5 秒切换到 10 秒，两条请求复用模型和 CPU 权重。
10 秒请求的单卡 PyTorch 峰值保留显存约 30.34 GiB。两条近同时提交的 5 秒请求也已验证排队与串行执行。15 秒、队列满载和长期内存趋势仍需实测。

[实测性能和验证范围](performance/h3-cuda.md)

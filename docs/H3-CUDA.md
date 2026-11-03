# MiniMax-H3 CUDA 部署与启动

使用 `XGEN_DEVICE=cuda` 启动现有 CLI 或 HTTP 服务。省略该变量时选择 NPU。设备由启动环境决定，HTTP 请求沿用现有生成参数。

## 环境和模型

准备支持 RTX 5090 的 PyTorch CUDA、Triton、兼容 MiniMax-H3 的 Diffusers 和 Transformers。NVFP4 需要 Comfy-Kitchen CUDA 扩展，Sage 和 Sparge 分别需要对应 CUDA 扩展。Triton 首次编译需要与运行时 Python 版本匹配的开发头文件和可写的缓存目录。安装支持 H.264 和 AAC 的 FFmpeg。部署示例使用本地模型组件和权重。

```bash
export PYTHONPATH="./x_base:./x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=./weights/h3
export NVFP4_CHECKPOINT=./weights/h3-fl2va-nvfp4.safetensors
export FIRST_FRAME=./inputs/first-frame.png
export PROMPT_FILE=./inputs/prompt.txt
```

`MODEL_PATH` 指向含 `modular_model_index.json` 的 FL2VA 组件目录。NVFP4 checkpoint 需与该目录的 Transformer 配置匹配。CUDA 打包权重已验证的工作流为 FL2VA。

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

[README 的 CUDA 在线启动示例](../README.md#实验性-cuda-2k-refiner)给出了 JSON 和 torchrun 命令。使用 `/generate` 和 `/status/<task_id>` 提交及查询任务。2K CUDA 需要 SP8、NVFP4 FL2VA Base 权重及预处理的 Refiner 资产。按 [H3 部署教程的模型准备章节](H3-DEPLOYMENT.md#43-一次性预处理)准备 Refiner，在 CUDA 环境执行 `h3_regenerate.prepare`，将结果输出到独立目录。普通 768P 请求可直接使用 Base 配置。

[实测性能和验证范围](performance/h3-cuda.md)

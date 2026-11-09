# CUDA 标准方案二：Sage-FP8

方案二组合 Sage attention、FP8 双向 PHAA 通信、输出 AdaLN 的 indexed affine fusion 和 PyTorch `expandable_segments`。它沿用原来的 FP8 Linear、文本 TP8、PHAA2、50 层 DiT 常驻、VAE SP8 空间 tile、decoder 分阶段加载、音频 FP32 和逐视频空闲显存回收。

八卡 RTX 5090 上，5/10/15 秒、20 步的 T2VA / I2VA 均完成成片和完整音视频解码，实测比方案一更快。**Sage 成片的人工画面、口型和听音审核仍待完成。** 原 [方案一 fp8-bf16](H3-CUDA-PRECISION.md) 的 5 秒固定样例已有人工认可，继续保留为默认；选择方案二不会改变 NPU 的默认路径。

## 范围与依赖

- 已测：8×RTX 5090、SP8、文本 TP8、1344×768、24 FPS、20 次实际 DiT；T2VA / I2VA，5、10、15 秒。
- 本页性能表是原 T2VA/I2VA 配方。R2VA 请使用 v0.4 的 [5090全并行优选方案](H3-CUDA-COMPOSABLE.md)，并查看它独立的性能和验证范围；不要沿用本页的成片结论。2K、四卡和在线端到端仍需分别验证。
- 使用与方案一相同的完整 FL2VA/Base 模块权重，包含 Transformer、文本编码器、音视频 VAE、处理器和调度器。无需 NVFP4、Veda 预测器或额外 LoRA。
- 除原 PyTorch CUDA、Triton、Diffusers、Transformers、FFmpeg / FFprobe 外，需要安装支持 RTX 5090 的 [SageAttention CUDA 扩展](https://github.com/thu-ml/SageAttention)。不增加全局强制依赖；未安装时请选方案一。
- 视频 decoder 的 CPU 锁页副本约 4.52 GiB/rank，八卡约 36.1 GiB；还需考虑模型加载、文本编码器和其他 CPU 权重，不能把此数作为整机内存需求。

## 离线启动

从仓库根目录执行，先准备自己的素材和对应时长的 H3 prompt：

```bash
export BASE_MODEL=./weights/h3
export PROMPT_FILE=./inputs/prompt.txt
```

5 秒文生视频：

```bash
H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

15 秒首帧图生视频：

```bash
H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda i2va 15 768p inputs/first-frame.png
```

时长可换为 `5`、`10` 或 `15`。每条命令生成一个视频，默认保存到 `outputs/`。通过 `OUTPUT_PATH` 可指定独立文件名。`H3_BASE_STEPS=20` 就是实际 20 次 DiT；省略时脚本仍默认 50 步，本文性能和精度审核范围仅为已测的 20 步样例。

选择 `fp8-sage` 后，通用脚本在清理旧 CUDA 实验开关后统一设置：

| 项目 | 标准设置 |
|---|---|
| Precision | `--h3_cuda_precision fp8-sage` |
| attention | 5090 上 Sage INT8 Q/K＋FP8 P×V |
| 去程 / 回程 | `XGEN_PHAA_FP8=1`，Q/K/V 和 attention 输出均走 FP8 |
| 输出调制融合 | `XGEN_H3_FUSED_OUTPUT_ADALN=1` |
| 分配器 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 并行 | SP8、文本 TP8、PHAA2，`NCCL_MAX_CTAS=2` |
| CPU 线程 | 未指定时 `OMP_NUM_THREADS=1` |

脚本固定上述分配器设置；若设置了冲突的 `PYTORCH_ALLOC_CONF`，会在启动前报错。直接使用 `torchrun` 时，precision 参数只选择模型配方，仍需像在线示例一样显式提供通信、输出融合和分配器环境变量。

## 计算精度与通信格式

Sage 的 INT8 Q/K 和 FP8 P×V 是 **attention 激活计算格式**，不会自动量化模型权重。两套方案都保留现有 FP8 W8A8 Linear 和文本 FP8 TP8，没有新增 NVFP4 权重量化。

xGen 当前的低精度 PHAA 是 Q/K/V 全部带 row scale 按 FP8 传输，接收后还原 BF16，Sage 再进行其内部 INT8 Q/K 量化；attention 输出回程也走 FP8。**不是 INT8 Q/K All-to-All。** Sage 和 FP8 通信均为近似计算，需要单独审核精度。

后端复用原 `AttentionManager` 和 H3 模型局部 callback。遇到 mask、形状或 dtype 不满足 Sage 条件时，沿用既有 SDPA 回退。`calls` 与 `kernel_calls` 可区分总调用和实际 Sage kernel 次数；本轮正式请求每 rank 的 2,000 次 DiT attention 调用均实际走 Sage。

输出 indexed affine 只融合索引读取、`1 + scale`、乘法和加 shift，保留原 BF16 逐次舍入。RMSNorm、投影和 SP Gather 不变。分配器仅改变存储分配；不会卸载仍活跃的 50 层 DiT 权重。

## 在线服务

以下为配置示例，**Sage 的 HTTP/MaaS 端到端尚未实测**。precision 是服务启动配置，不接受请求间静默切换；修改后重启。请求、排队、健康检查、超时和下载接口继续使用原实现。

先填写 [Sage 示例配置](../example/diffusers/configs/h3_cuda_sage.example.json) 的模型目录，再启动原 Base HTTP 服务：

```bash
XGEN_DEVICE=cuda XGEN_PHAA_FP8=1 XGEN_H3_FUSED_OUTPUT_ADALN=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 NCCL_MAX_CTAS=2 \
torchrun --standalone --nproc_per_node=8 server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_sage.example.json --host 127.0.0.1 --port 5001
```

该 Base 模式允许 T2VA / I2VA，参数及提交示例见 [在线教程](H3-ONLINE.md#提交原服务请求)。若部署 MaaS，将 `--service-mode t2va` 换为 `--maas-fl2va-input`；MaaS 仍固定 I2VA、5 秒、768P、50 步，不会因切换 Sage 放宽请求约束。在线首次部署必须先用真实请求检查成片和就绪状态，离线数据不能当作 HTTP SLA。

## 性能与验证

2026-11-06，8×RTX 5090、768P、seed42。六项复用同一组模型，每项先完整 1 步预热，再测 20 步。以下取八 rank 最慢的生成至 MP4 保存耗时；初始化和完整媒体解码检查不计入请求时延。阶段 event envelope 可能重叠，不应相加。

| 任务 | 时长 | 生成＋保存 | DiT | 峰值 allocated / reserved（每卡） |
|---|---|---:|---:|---:|
| T2VA | 5s | 42.935 s | 34.518 s | 28.28 / 28.49 GiB |
| I2VA | 5s | 48.089 s | 29.741 s | 28.29 / 28.52 GiB |
| T2VA | 10s | 76.226 s | 63.565 s | 28.32 / 30.14 GiB |
| I2VA | 10s | 97.808 s | 78.950 s | 28.33 / 29.76 GiB |
| T2VA | 15s | 148.148 s | 130.677 s | 29.88 / 30.40 GiB |
| I2VA | 15s | 161.120 s | 138.185 s | 30.00 / 30.41 GiB |

运行源码冻结于 `d428a9b`；本次整理只改命名、错误提示、启动组合和文档，没有改变已测算子的数值路径。六项均完整音视频解码通过，各 rank 完成 20 次实际 DiT，模型 identity 在 12 次预热/正式请求间保持不变。全片六帧抽查未见明显黑屏或整体崩坏，人工完整画面、口型和听音仍待审核。结束后八卡无计算进程、显存回到 0 MiB；这不构成长时间无泄漏保证。

同样开启输出融合和 expandable segments 的方案一 15 秒参考值为 T2VA 293.385 s、I2VA 308.161 s；峰值 allocated 同样为 29.88 / 30.00 GiB。它在更早的独立进程做过 1 步检查，正式组未再预热，首条可能包含首次触达。5/10 秒历史对照及各自预热条件见 [性能记录](performance/h3-cuda.md#cuda-标准方案二-sage-fp8)。这是整套方案的单次参考，不是仅替换 attention 的消融，也不是统计均值。Sage 的显存峰值在 15 秒档基本不变，主要收益是提速。

原离线保存流程包含 FFmpeg `-shortest`：请求生成 124 / 243 / 362 帧，实际 MP4 为 124 / 243 / 361 帧；15 秒档音频为 15.075 秒、视频为 15.041667 秒。没有补帧修改这些记录。Base HTTP 自身还有按请求时长裁剪的保存规则，不能直接沿用离线帧数作为在线验收标准。

## 回退到方案一

在新的终端运行默认命令即可：

```bash
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

若使用直接 torchrun 或在线服务，将 precision 改为 `fp8-bf16`，取消 `XGEN_PHAA_FP8` 和输出融合环境变量后重启。输出融合和 expandable segments 可作为独立选项用于方案一长视频，但不属于原 5 秒人工认可样例的默认设置。第一套默认方案不需要 SageAttention。

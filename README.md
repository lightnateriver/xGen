# xGen

xGen 是面向昇腾 NPU 和 NVIDIA GPU 的音视频生成引擎。你可以用命令行生成视频，也可以启动 HTTP 服务。MiniMax-H3 支持文字、首帧及图片、视频、音频参考输入，输出带声音的视频。

三种任务分别是文生视频（T2VA）、首尾帧图生视频（I2VA，接口名 `fl2va`）和多媒体参考生视频（R2VA，接口名 `ref2va`）。xGen 支持 768P 生成和可选 2K 超分。CUDA MaaS 模式另外提供输入校验、排队、健康检查和成片校验。

[快速开始](#快速开始) · [权重下载](#h3-权重下载) · [离线生成](#离线生成) · [在线服务](#在线服务) · [B2/B3 性能](#b2b3-性能) · [GPU 性能](#gpu-性能) · [Agent 指南](#agent-使用指南) · [版本](#版本)

## 版本

v0.6 新增 [MiniMax-H3-Image 四卡接入](docs/H3-IMAGE.md)与[已验证的NPU常驻优化](docs/H3-IMAGE-NPU-OPTIMIZATIONS.md)：独立图像VAE、六步PDD、SP4/文本TP4、主DiT W8A8、AdaLN预计算和组件并行，覆盖文生图与最多16张参考图编辑。[40项性能与输入Shape](docs/performance/h3-image-npu-40cases.md) · [v0.6发布说明](docs/H3-V06-RELEASE.md)

v0.5 的 GPU **Sage-FP8 全并行 Streaming + AdaLN表** 组合默认包含 **PHAA FP8 Align16**，将同步 FP8 通信的每个 peer 包补齐到16字节，离线与在线启动均自动开启。需要关闭时，在启动前设置 `XGEN_PHAA_FP8_ALIGN16=0`，所有 rank 使用相同值。[配置与回退](docs/H3-PHAA-FP8-ALIGNMENT.md) · [12项在线性能对比](docs/performance/h3-cuda-phaa-align16.md)

v0.5 的八卡 CUDA H3 Base 默认采用 **Sage-FP8 全并行 Streaming + BF16 AdaLN 缓存表**，统一覆盖 T2VA、I2VA／首尾帧和 R2VA：主干 SP8、文本 FP8 TP8、FP8 Linear／PHAA、组件并行、50 层 DiT 常驻与 `native-stream` 解码。需先准备对应工作流的表及剔除 AdaLN 权重后的 checkpoint。[默认配置与差异](docs/H3-V05-CUDA-UPGRADE.md) · [AdaLN 准备与回退](docs/H3-CUDA-ADALN.md)

原 [fp8-bf16 精度参考方案](docs/H3-CUDA-PRECISION.md) 和 [Sage-FP8 基础方案](docs/H3-CUDA-SAGE.md) 保留，可显式选择。新默认的技术解码、抽帧诊断与人工画面／音频验收分别记录，不沿用旧样例的人工认可结论。v0.5 NPU Base 统一采用 [Laser INT8 全并行 Streaming](docs/H3-NPU-DEFAULT.md)，覆盖T2VA/I2VA/R2VA，包含AdaLN查表和通用组件并行。

H3 视频 decoder 支持 CUDA/NPU 共用的 legacy、lightning、native-stream 三后端。CUDA 推荐的 T2VA/I2VA/R2VA 配方默认 native-stream，保留显式回退；v0.5 NPU Base 默认也使用 native-stream。[后端配置与验证范围](docs/H3-OMNI-VAE.md)

v0.4 的 H3 在线接口将提示词上限从 2,000 放宽到 **7,000 个 UTF-16 码元**，按官方 CLI 的 JavaScript `text.length` 计数：常见汉字或英文字母各计 1，`🙂` 计 2。Base、Refiner、MaaS 共用该限制，在入队前拒绝超长文本，保留原文；这不是 tokenizer 的 token 上限。计数规则、错误响应和请求体上限见 [在线服务指南](docs/H3-ONLINE.md) 与 [MaaS API](server/video/MAAS-API.md)。

v0.4 的 `--maas` 提供 MiniMax-H3 V2 格式的生成、查询、列表及取消/删除接口，支持 **768P、4～15 秒整数**、首尾帧、R2VA 和自适应比例；保留 `/generate`、`/v1/tasks` 的旧请求格式。新增可选 AIGC 可见水印、七天任务元信息与状态回调。R2VA 限制为最多 9 图、3 视频、3 音频，总计 12 个文件；视频、音频分别累计不超过 15 秒。**IR 和 2K 目前是默认关闭的模拟后端**，需 `--maas-mock-backends` 显式启用，不代表真实模型能力或性能；再生成接口未实现。新增协议的 CPU 验证与真实 GPU 成片验收分开记录。[API 与启动](server/video/MAAS-API.md) · [对齐边界和替换后端](docs/H3-MAAS-V2.md)。

**MaaS 平台、V2 协议与计算后端是不同层。** 你的 MaaS 平台通过 V2 HTTP 协议调用 xGen；`--maas` 是 xGen 在线服务启用接口和可靠性能力的开关，CUDA/NPU 则负责实际模型计算。V2 字段本身不绑定硬件。当前 `--maas` 启动校验仍限定 CUDA、`--init` 和 SP/进程数匹配；NPU 的旧 HTTP/离线入口继续使用，NPU 上的 MaaS V2 尚未开放或验收。详见 [接口与后端的关系](docs/H3-MAAS-V2.md#接口平台与计算后端)。

水印在 CPU RGB8 帧进入 FFmpeg 前叠加，默认关闭、不增加第二次转码。[5/10/15 秒 CPU 增量实测与复现脚本](docs/performance/h3-watermark-cpu.md)。

| 分支 | 核心变化 | 文档 |
|---|---|---|
| [v0.6](../../tree/v0.6) | 在最新v0.5基础上增加MiniMax-H3-Image六步文生图/参考图编辑与四卡NPU W8A8常驻方案；保留视频默认与GPU Align16。 | [图像入口](docs/H3-IMAGE.md) · [优化](docs/H3-IMAGE-NPU-OPTIMIZATIONS.md) · [40项性能](docs/performance/h3-image-npu-40cases.md) |
| [v0.5](../../tree/v0.5) | CUDA Base 默认 Sage-FP8 全并行 Streaming + AdaLN 表，覆盖 T2VA/I2VA/R2VA；保留旧精度回退。NPU Base 默认 Laser INT8全并行Streaming，覆盖T2VA/I2VA/R2VA、20实际步、SP8，共用组件调度与Stream FP16存储。 | [NPU默认与升级](docs/H3-NPU-DEFAULT.md) · [复测](docs/performance/h3-native-shared-fp16.md) · [通用适配](docs/H3-PORTABLE.md) |
| [v0.4](../../tree/v0.4) | 增加 `cuda-5090 r2va` 优选 Sage-FP8 全并行预设：SAP、本地输入投影、请求内固定条件复用及组件并行，DiT 全程常驻；各项可独立关闭。新增统一 `--maas`、多模态输入校验、首尾帧、768P 自适应比例和公网素材下载。旧 CUDA/NPU 默认保留，TP/CP 为默认关闭的实验选项。 | [启动与开关](docs/H3-CUDA-COMPOSABLE.md) · [实测性能](docs/performance/h3-cuda5090-full-parallel.md) · [MaaS API](server/video/MAAS-API.md) · [在线启动](docs/H3-ONLINE.md) |
| [v0.3.2-maas](../../tree/v0.3.2-maas) | CUDA 提供默认精度参考 fp8-bf16 和更快的 Sage-FP8 两套标准方案；增加输出 AdaLN 融合与长视频实测。保留 MaaS 排队、超时、健康探针、原子发布、环境采集，以及 NPU 离线 benchmark、R2VA 多媒体参考、实际步数语义和逐视频显存回收。 | [CUDA 方案一](docs/H3-CUDA-PRECISION.md) · [CUDA 方案二](docs/H3-CUDA-SAGE.md) · [API](server/video/MAAS-API.md) · [在线启动](docs/H3-ONLINE.md) · [benchmark](docs/H3-NPU-BENCHMARK.md) · [升级说明](docs/H3-UPGRADE.md) |
| [v0.3.1](../../tree/v0.3.1) | CUDA 配置集中到 adaptor。增加可选异步 FP8 PHAA、回程直写、DiT 算子融合和 VAE 空间切片。完成 CUDA、NPU 双设备冒烟。 | [实测](docs/performance/h3-cuda.md#v031-双设备冒烟) |
| [v0.3-cuda](../../tree/v0.3-cuda) | CUDA H3 FL2VA、NVFP4 DiT 常驻、Sage/Sparge、FP8 PHAA 双向通信、文本 TP8、VAE SP、FastVideo C4 预设及可选 FL2VA 2K。 | [CUDA 部署](docs/H3-CUDA.md) |
| [v0.3](../../tree/v0.3) | NPU Base DiT 增加可选 Spark-H3 和 Veda。支持 Veda 稀疏度配置和实验性 LightX2V 8 步配方。 | [启动和回退](docs/H3-ATTENTION.md) · [实测](docs/H3-ATTENTION-VALIDATION.md) |
| [v0.2](../../tree/v0.2) | 增加默认关闭的 H3 → Open-H3-Regenerate-2K 链路，复用原 Manager 和 HTTP 服务。 | [NPU 部署](docs/H3-DEPLOYMENT.md) |
| [v0.1](../../tree/v0.1) | 原始 xGen，包含 Diffusers 适配、SP、FSDP、PHAA、量化和视频服务。 | [基础示例](example/diffusers) |

## 快速开始

准备带 PyTorch、CUDA 或 CANN、FFmpeg 的 Linux 环境。从仓库根目录执行：

```bash
python -m pip install -e ./x_base --no-deps
python -m pip install -e ./x_diffusers --no-deps
python -m pip check
python tools/collect_env.py
mkdir -p outputs
```

确认采集结果显示八张可见设备。然后下载权重并准备自己的提示词和图片。以下示例都从仓库根目录运行。

## H3 权重下载

按工作流下载资产。H3 Base 仓库包含文本编码器、视频 VAE、音频 VAE 和调度组件。T2VA 和首帧 I2VA 使用 FL2VA/Base Transformer，R2VA 使用 Ref2VA Transformer。保留 xGen 所需的目录布局。

| 资产 | Hugging Face | ModelScope | 用途 |
|---|---|---|---|
| MiniMax-H3 Base BF16 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | [MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) | 768P Base。按任务使用 `transformer/` 或 `transformer_ref/`，并保留根目录的 `modular_model_index.json` |
| Open-H3-Regenerate-2K | [权重仓库](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) | [权重仓库](https://modelscope.cn/models/Efficient-Large-Model/Open-H3-Regenerate-2K) | 2K 链路所需的 latent upscaler 和 Refiner 资产，按部署教程补齐 Sharp AdaLN 表并预处理 |
| FL2VA Pruned NVFP4 | [实测版本的单文件](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4/blob/8c5abfed61e1b6a170240792b65253fba1a65b7b/minimax_h3_fl2va_pruned_nvfp4.safetensors) | — | 历史 CUDA 高性能 FL2VA 与 CUDA 2K 的 Base DiT；当前默认 Base / MaaS 配方不需要此文件。另需匹配的 H3 模型组件 |
| LightX2V Turbo 8 步 LoRA | [768P Diffusers BF16 文件](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors) | [LightX2V Turbo](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | 可选的 8 步 FL2VA/T2VA 配方 |
| Veda FP8 预测器 | [实测版本的单文件](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/76f202874608115408d73280be9531c4ab888242/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors) | [Veda 预测器仓库](https://modelscope.cn/models/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) | 仅启用 Veda 时需要 |

下载和预处理步骤见 [Base 和 Refiner 资产准备](docs/H3-DEPLOYMENT.md#4-模型与目录准备)、[CUDA 组件与 NVFP4 下载](docs/H3-CUDA.md#环境和模型) 和 [Veda / Turbo 配方](docs/H3-ATTENTION.md)。AdaLN 表和预处理后的 Refiner 由部署方按配方生成。

## 离线生成

在仓库根目录运行。把提示词放到 `inputs/prompt.txt`。I2VA 默认读取 `inputs/first-frame.png`，R2VA 默认读取 `inputs/ref01.png` 到 `inputs/ref05.png`。视频保存到 `outputs/`。

H3 的 `num_inference_steps` 直接填实际 Base DiT 次数，例如 20 或 50。旧版直接传调度点的普通 H3 配置需将 `51` 改为 `50`、`21` 改为 `20`；Turbo-SLA 等已经按实际步数配置的路径见 [升级说明](docs/H3-UPGRADE.md)。

### 昇腾 NPU 启动

按 [NPU 权重准备](docs/H3-DEPLOYMENT.md#4-模型与目录准备) 下载模型并生成 AdaLN 表。每条命令启动一个八卡任务。时长可以填写 `5`、`10` 或 `15`。

768P Base 可选用 VC ExpCast Attention 和 Ascend H3 融合算子。安装与开关见 [NPU 融合算子](docs/H3-NPU-FUSIONS.md)。

```bash
# 768P：v0.5 默认 Laser INT8 全并行 Streaming / Base 20步
export H3_VAE_MODEL_PATH=weights/native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 10 768p
bash example/diffusers/scripts/run_h3.sh npu r2va 15 768p

# 2K，需先准备 Refiner
bash example/diffusers/scripts/run_h3.sh npu t2va 5 2k
bash example/diffusers/scripts/run_h3.sh npu i2va 10 2k
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k
```

### NVIDIA CUDA 启动

#### 八卡 RTX 5090：Sage-FP8 全并行 Streaming + AdaLN 表

先安装 [CUDA 依赖](docs/H3-CUDA-PRECISION.md#模型与环境)，按 [AdaLN 教程](docs/H3-CUDA-ADALN.md) 为 FL2VA 和 Ref2VA 分别准备 20／50 步表及剔除权重的 checkpoint。缺少资产会明确报错，不会静默退回其他方案。

```bash
export BASE_MODEL=weights/h3-fl2va
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/h3-native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda i2va 10 768p inputs/first-frame.png
```

R2VA 切换完整模型及相应 Ref2VA 表，示例为五张参考图：

```bash
export BASE_MODEL=weights/h3-ref2va
bash example/diffusers/scripts/run_h3.sh cuda r2va 15 768p inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

时长可填 `5`、`10`、`15`，`cuda-5090` 与 `cuda` 在 Base 768P 使用相同默认。仅对可信的原生 `video_vae` 目录开启授权；它不是 Diffusers 的 `vae` 子目录。参考图短边默认仍为 2048，可用 [独立参数](docs/H3-REFERENCE-IMAGE-SIZE.md) 调整。

| 方案 | 选择方式 |
|---|---|
| 新默认：Sage-FP8 全并行 Streaming + AdaLN 表 | 上述命令；自动按实际步数选对应表 |
| 保留全并行，关闭 AdaLN 表 | 命令前加 `H3_CUDA_ADALN_MODE=owner` |
| 原精度参考：FP8 Linear + BF16 attention／通信 | 命令前加 `H3_CUDA_PRECISION=fp8-bf16` |
| 原 Sage-FP8 基础方案 | `H3_CUDA_PRESET=none H3_CUDA_PRECISION=fp8-sage` |
| 原始 CUDA 路径 | `H3_CUDA_PRECISION=none` |

单项开关仍可自由组合，见 [全并行配置](docs/H3-CUDA-COMPOSABLE.md)。本方案为 SP8，不默认开启实验 TP/CP、NVFP4 或稀疏 attention。历史和最新性能的预热、AdaLN 模式及精度范围见 [Streaming 矩阵](docs/performance/h3-cuda-streaming-matrix.md)。

开发中的 [TP2 + Ulysses CP4 备选路线](docs/H3-CUDA-TP-CP.md) 使用独立开关，默认仍为原 SP。它复用现有推理链路，分别配置 TP/CP 的 BF16 或 FP8 传输；合成单层验证与完整视频性能、人工精度验收分开记录，不能沿用上表的成片结论。

实验性 [输入 Token Refiner SP8 / cuDNN / FP8 FFN](docs/H3-INPUT-TOKEN-REFINER.md) 通过独立参数开启，默认关闭。单图 5 秒 R2VA 的 20 步预热后实测中，该模块快 3.35 倍、稳态 DiT 耗时减少约 2.07%，但本次 E2E 没有改善；完整画面、口型和听音审核仍待完成。

CUDA 2K 目前走 I2VA，需要打包 NVFP4 和 [CUDA Refiner 资产](docs/H3-CUDA.md#实验性-cuda-2k-refiner)。接口接受 5、10、15 秒，已有 5 和 10 秒结果。

```bash
bash example/diffusers/scripts/run_h3.sh cuda i2va 5 2k
```

**历史实验配方**。`H3_FAST=1` 显式选择另一套 5 秒 768P 配方，不是当前默认的精度认可方案。该配方已通过完整解码，跨素材画质仍待复核。请在未开启其他实验开关的终端运行。 [详细配置](docs/H3-CUDA.md#v031-5-秒实验配方)

```bash
H3_FAST=1 bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p
```

换素材时，可以在命令末尾直接传首帧或 R2VA 参考文件。`PROMPT_FILE` 和 `OUTPUT_PATH` 分别覆盖提示词和输出路径。[参考媒体限制](docs/H3-DEPLOYMENT.md#r2va-多媒体参考)

```bash
PROMPT_FILE=inputs/my-prompt.txt OUTPUT_PATH=outputs/result.mp4 \
  bash example/diffusers/scripts/run_h3.sh npu i2va 5 768p inputs/my-first-frame.png
```

v0.5 NPU Base 默认使用 Laser attention、INT8 DiT Linear、Stream decoder、full-parallel和20次实际Base DiT；需要21点AdaLN表。它不是FP8配方。详见[NPU默认方案](docs/H3-NPU-DEFAULT.md)。Spark-H3、Veda 和 LightX2V 8 步另见 [可选加速配方](docs/H3-ATTENTION.md)。

H3 在线、离线均在每条视频保存完、临时输出释放后同步设备并回收 PyTorch 空闲显存，保留常驻模型。日志记录回收耗时和前后显存；该策略对单条请求内部的峰值 OOM 不作保证，当前策略的完整设备矩阵待验证。

## B2/B3 性能

最新v0.5 [Laser INT8全并行Streaming性能表](docs/performance/h3-npu-full-parallel-streaming.md)包含T2VA/I2VA 5/10/15秒六项新复测，以及既有R2VA单图/五图六项，保留阶段计数、显存与内存验证边界。

[NPU 离线 benchmark](docs/H3-NPU-BENCHMARK.md) 提供 20 步矩阵、媒体检查和 HBM/RSS/PSS 采样。[2026-11-04 至 05 的完整实测](docs/performance/h3-npu-offline-20261105.md) 包含 35 项正式结果及预热、阶段耗时和内存数据。该批使用了现已移除的阶段性清理，不能当作当前「成片后回收」策略的回归结果。

### NPU 20 步离线矩阵（2026-11-04 至 05）

8 卡昇腾 910B1、SP8、文本 TP8、PHAA2、Laser、DiT INT8；每项一次匹配预热后测一次。单位为秒，包含推理和保存、不含初始化。2K 另加 3 步 BF16 Refiner。以下为旧阶段清理策略的历史值，当前成片后回收策略待复测。

| 时长 | 输出 | T2VA | I2VA | R2VA 1 图 | R2VA 5 图 | R2VA 9 图 |
|---|---|---:|---:|---:|---:|---:|
| 5s | 768P | 64.23 | 65.48 | 96.07 | 300.39 | 623.64 |
| 5s | 2K | 128.03 | 138.57 | 159.49 | 362.48 | 682.01 |
| 10s | 768P | 144.46 | 148.01 | 196.49 | 462.09 | 845.88 |
| 10s | 2K | 336.24 | 334.72 | 386.49 | 644.11 | 1031.13 |
| 15s | 768P | 269.86 | 274.38 | 332.07 | 665.90 | 1120.32 |
| 15s | 2K | 651.42 | 654.23 | 715.40 | 1044.61 | 1499.03 |

35 项技术检查通过；人工画面及听音未全部通过验收，已知问题、5 项多媒体探针及内存表见 [完整报告](docs/performance/h3-npu-offline-20261105.md)。

### 历史 50 步在线／离线对照

B2 是历史测试组，芯片实际识别为 **910B1**。B3 为 **910B3**。两组均是 8 卡、SP8、单卡 64 GiB，Base 50 次实际 DiT，2K 另加 3 次 Refiner DiT。I2VA 输入为 1280×720 首帧，R2VA 为五张 2560×1440 参考图。

单位为秒，每格一条请求。在线从客户端提交开始计时，离线从开始处理请求计时。通常都计到视频导出完成，模型初始化另计。两组执行顺序和计时实现有差异。

| 输入 | 时长 | 输出 | 在线 B2 | 在线 B3 | 离线 B2 | 离线 B3 |
|---|---:|---|---:|---:|---:|---:|
| T2VA | 5s | 768P | 110.66 | 125.75 | 248.32（冷） | 124.25 |
| T2VA | 5s | 2K | 181.55 | 192.49 | 175.09（补） | 197.27 |
| T2VA | 10s | 768P | 297.73 | 331.32 | 297.39（补） | 338.30 |
| T2VA | 10s | 2K | 483.43 | 573.19 | 484.88（补） | 568.97 |
| T2VA | 15s | 768P | 574.34 | 673.40 | 576.75（补） | 664.85 |
| T2VA | 15s | 2K | 969.52 | 1156.55 | — | 1151.69 |
| I2VA | 5s | 768P | 130.59 | 137.32 | 126.28 | 133.73 |
| I2VA | 5s | 2K | 179.64 | 203.06 | 184.00 | 207.74 |
| I2VA | 10s | 768P | 318.74 | 351.44 | 303.25 | 356.70 |
| I2VA | 10s | 2K | 502.48 | 589.34 | 503.57 | 593.25 |
| I2VA | 15s | 768P | 598.86 | 696.48 | 597.41 | 705.86 |
| I2VA | 15s | 2K | 989.27 | 1174.78 | 994.97 | 1175.62 |
| R2VA | 5s | 768P | 659.01 | 774.92 | 658.45 | 771.23 |
| R2VA | 5s | 2K | 716.95 | 844.59 | 719.03 | 837.17 |
| R2VA | 10s | 768P | 1030.40 | 1205.10 | 1037.66 | 1206.30 |
| R2VA | 10s | 2K | 1220.84 | 1441.97 | 1224.21 | 1444.09 |
| R2VA | 15s | 768P | 1509.57 | 1772.10 | 1502.16 | 1777.49 |
| R2VA | 15s | 2K | 1898.22 | 2267.94 | 1904.32 | 2256.44 |

**冷**表示含进程启动、初始化和退出。**补**表示来自独立的常驻补测组。**—**表示该项已生成并补验收，完整 E2E 未保留。72 条均通过技术检查和完整音视频解码，人工画面与听音复核仍待记录。 [逐项数据](docs/performance/historical-pairs.csv) · [阶段拆分](docs/performance/historical-stages.csv)

## GPU 性能

八卡20步、全部 BF16 AdaLN 表模式的[在线 Align16 对照表](docs/performance/h3-cuda-phaa-align16.md)覆盖 T2VA/I2VA/单图及五图 R2VA 的5/10/15秒。每项完整预热一次、正式一次，初始化另计，逐项列出推理、DiT、HTTP耗时和峰值显存，并注明历史节点差异。

H3 VAE Decode 在 CUDA/NPU 均可选择 `--h3_vae_backend legacy|lightning|native-stream`。CUDA 推荐的 `fp8-bf16`、`fp8-sage` 和全并行预设默认 `native-stream`；NPU Base推荐脚本与在线预设默认 `native-stream`，其余低层离线入口保持既有选择逻辑，显式选择优先。H3 旧命令的 `--vae_lightning decoder` 需替换为 `--h3_vae_backend lightning`，其他模型的旧参数保留。实验性 [Native-stream 后端](docs/H3-OMNI-VAE.md)迁移原生 H3 tile 调度、分块输出、异步 D2H 和 MP4 编码，需要原生 `video_vae` 权重目录和显式授权。[组件实测对比](docs/performance/h3-omni-vae.md)包含性能、显存和精度验证范围。

昇腾也可选择 `native-stream`，需追加 `--vae_matmul_quant none`。已完成 [8×910B3 三后端组件对比](docs/performance/h3-vae-npu.md)：768P、5/10/15 秒，预热后新后端比 Lightning 快 1.57–1.95 倍；初始化单列，尚不代表整网加速或成片精度验收。

[默认切换验证](docs/performance/h3-vae-release.md)：134 项回归通过；八卡 5090 的新默认 R2VA 单图 5 秒链路连续两次 1-step 请求完成，第二次八 rank 最大调用耗时 11.48 秒，累计 allocated 峰值 29.14 GiB／卡。这里只验证链路与复用，不是完整步数的性能／精度结论。

除注明外，均为 8×RTX 5090、SP8、50 次实际 Base DiT。下表按用例和计时边界区分，保留各自精度证据。

| 路径 | 请求 | 耗时 | 计时和验证口径 |
|---|---|---:|---|
| vLLM-Omni FP8 线性＋BF16 稠密 attention | T2VA，5s/768P | 139.74 s | 两次热态 HTTP E2E 均值，含 MP4 下载。单例「精度 OK」参考 |
| xGen BF16 稠密 attention | I2VA，5s/768P | 574.61 s | 整进程 wall，含加载。完整音视频解码通过 |
| xGen NVFP4 常驻＋Sage＋FP8 PHAA | I2VA，5s/768P | 102.77 s | 生成并保存，完整音视频解码通过 |
| xGen NVFP4 常驻＋Sparge＋FP8 PHAA＋VAE SP | I2VA，5s/768P | 96.07 s | 生成并保存，完整音视频解码通过 |
| xGen v0.3.1 C4＋异步 PHAA＋DiT 融合＋VAE tile | I2VA，5s/768P | 54.10 s | 生成并保存，含加载为 84.11 s。与重构前同输入参考成片的解码哈希一致 |
| xGen 历史 C4 热测 | I2VA，5s/768P | 51.29 s | 两次生成并保存均值，重构前同步 FP8 配方 |
| xGen NVFP4 Base＋原生 attention＋Refiner | I2VA，5s/2K | 316.17 s | rank 0 请求周期，Base 50＋Refiner 3 次 DiT |
| xGen NVFP4 Base＋Sparge/FP8 PHAA＋Refiner | I2VA，5s/2K | 172.03 s | rank 0 请求周期，Base 50＋Refiner 3 次 DiT |
| xGen 原生 attention＋Refiner | I2VA，10s/2K | 606.24 s | 历史服务推理计时，同实例从 5 秒切换到 10 秒 |
| xGen MaaS 早期八卡 HTTP 测试 | I2VA，5s/768P | 51.655 s | 一条热态成功请求，从受理到发布。推理 50.232 s |
| xGen 历史 fp8-bf16 | T2VA，5s/768P，**20 步** | 56.75 s | 历史常驻生成至保存，取最慢 rank；本轮样例人工精度可接受 |
| xGen 历史 fp8-bf16 | I2VA，5s/768P，**20 步** | 72.67 s | 同组模型复用；首次图像条件请求，样例人工精度可接受 |
| xGen CUDA R2VA Streaming | 五图，5s/768P，20步 | 133.69 s | 常驻、完整20步预热后正式；AdaLN表模式，人工完整视听待审，见下表链接 |

[详细 GPU 性能与对照口径](docs/performance/h3-cuda.md)

最新 **Sage-FP8 全并行 Streaming** 的 12 项 20 步结果见 [完整表格](docs/performance/h3-cuda-streaming-matrix.md)：包含 T2VA/I2VA/单图及五图 R2VA 的 5/10/15 秒。每项均先完整预热 20 步，owner／table 分行标明，不能混算查表收益。Max 多媒体压力样例仍在视觉阶段 OOM，不能承诺满载容量。

**CUDA 两套标准方案的历史 20 步参考（legacy decoder）**：8×RTX 5090、SP8、768P；单位为生成至 MP4 保存的秒数，不含模型初始化。Sage 每项先完整预热 1 步，复用同一组模型。历史基线的预热条件存在差异，比较的是整套配方，不能归因为单个算子收益。方案一仅 5 秒样例已人工认可，Sage 六项均为技术通过、人工待审；这些数据不代表新默认 native-stream 组合。

| 任务 | 时长 | 方案一 BF16 attention / 通信 | 方案二 Sage / FP8 通信 |
|---|---:|---:|---:|
| T2VA | 5s | 56.75 | **42.93** |
| I2VA | 5s | 72.67 | **48.09** |
| T2VA | 10s | 148.76 | **76.23** |
| I2VA | 10s | 164.70 | **97.81** |
| T2VA | 15s | 293.38 | **148.15** |
| I2VA | 15s | 308.16 | **161.12** |

方案一的 15 秒行额外开启输出 AdaLN 融合和 `expandable_segments`，不是未加开关的默认路径。15 秒峰值 allocated 仍约 29.88 / 30.00 GiB/卡，Sage 主要提速，没有明显降低该形状的峰值。详细阶段、显存和预热口径见 [六项实测](docs/H3-CUDA-SAGE.md#性能与验证)。

历史 fp8-bf16 配置另做了八卡 20 步回归：T2VA 57.67 s、I2VA 88.70 s。两条 MP4 均与上述人工认可样例的文件哈希完全一致；新机器首次图像条件请求没有单独预热。这不是本次全并行 Streaming + 表模式的性能。

## 在线服务

在另一个终端提交请求。下面的示例都用本机 5001 端口，启动其中一个即可。

### NPU 在线

v0.5 NPU Base统一默认Laser INT8全并行Streaming，包含20实际步、匹配AdaLN、组件并行与Native-stream，覆盖T2VA/I2VA/R2VA。先在JSON配置可信原生video_vae路径；T2VA/I2VA的5/10/15秒常驻推理已复测，性能和验证边界见[完整表](docs/performance/h3-npu-full-parallel-streaming.md)。要部署 R2VA，按 [NPU 在线配置](docs/H3-ONLINE.md#npu-base) 切换任务类型和 Ref2VA 权重。

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

### CUDA 在线

八卡 CUDA 768P Base 默认使用 Sage-FP8 全并行 Streaming + AdaLN 表。它与 NPU Base 使用同一套原 HTTP 提交、查询和下载接口。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode fl2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001
```

原服务用 `/generate` 提交。Base HTTP 接受 5 至 15 秒的整数时长。 [各任务的请求和下载示例](docs/H3-ONLINE.md#提交原服务请求)

在线 JSON 默认示例已改为全并行 Streaming + AdaLN 表；[表资产与回退](docs/H3-CUDA-ADALN.md) 同离线入口。HTTP/MaaS 的排队、下载及网络耗时不包含在离线性能中；不要将离线常驻 E2E 当作 HTTP SLA。

### 2K 在线

NPU T2VA 可直接用下面的命令启动。服务接受 5、10 或 15 秒。I2VA、R2VA 和 CUDA 2K 使用各自的配置。 [配置方法](docs/H3-ONLINE.md#2k-refiner)

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29558 \
  server/video/video_server.py --init --config example/diffusers/configs/h3_refiner.example.json \
  --host 127.0.0.1 --port 5001
```

### MaaS 在线服务

MaaS 支持 CUDA H3 Base T2VA/I2VA 和独立部署的 R2VA，输出 768P、5～15 秒，可使用自适应比例；步数以启动配置为准。默认采用 Sage-FP8 全并行 Streaming + AdaLN 表；准备好 FL2VA/Base 模块、匹配的表及剔除权重后的 Transformer 后启动：

```bash
pip install -r server/requirements-maas.txt

XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

通过 `POST /v1/tasks` 提交首条预热请求。预热成功后，等 `/ready` 返回 HTTP 200 再接流量。默认可排队 10 条，等待上限 600 秒。 [预热与高性能开关](docs/H3-CUDA.md#maas-在线服务) · [MaaS 接口](server/video/MAAS-API.md)

R2VA 要使用 Ref2VA 权重和独立实例：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode ref2va \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json --host 127.0.0.1 --port 5001
```

请求中的 `ref_path` 接收公网 HTTP(S) URL 或 Base64 data URI。图片、视频、音频单文件分别最多 30、50、15 MiB；请求体最多 64 MiB。数量、时长、格式、尺寸及比例在入队前校验，超限拒绝，不截断、不自动重试生成。部署前还需完成所选输入组合的真实 GPU 生成与成片验收，不能将历史离线性能当作新 MaaS 接口的性能。

## Agent 使用指南

仓库附带 [xgen-h3 skill](.agents/skills/xgen-h3/SKILL.md)。它按设备、任务、分辨率、时长和在线或离线方式选择启动命令，列出权重要求和验收步骤。

## 发布范围

仓库包含源码、示例、测试和脱敏性能数据。权重、内部机器配置、原始诊断报告、测试素材和成片由部署方自行管理。其他模型和界面入口见 [原版安装与模型说明](../../tree/v0.3.1#环境安装)、[Diffusers 示例](example/diffusers)、[ComfyUI 适配](x_comfyui) 和 [工具说明](tools/README.md)。

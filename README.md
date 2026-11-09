# xGen

xGen 是面向昇腾 NPU 和 NVIDIA GPU 的音视频生成引擎。你可以用命令行生成视频，也可以启动 HTTP 服务。MiniMax-H3 支持文字、首帧及图片、视频、音频参考输入，输出带声音的视频。

三种任务分别是文生视频（T2VA）、首帧图生视频（I2VA，接口名 `fl2va`）和多媒体参考生视频（R2VA，接口名 `ref2va`）。xGen 支持 768P 生成和可选 2K 超分。CUDA MaaS 模式另外提供输入校验、排队、健康检查和成片校验。

[快速开始](#快速开始) · [权重下载](#h3-权重下载) · [离线生成](#离线生成) · [在线服务](#在线服务) · [B2/B3 性能](#b2b3-性能) · [GPU 性能](#gpu-性能) · [Agent 指南](#agent-使用指南) · [版本](#版本)

## 版本

CUDA H3 Base 的 T2VA / I2VA 默认采用 [fp8-bf16 配方](docs/H3-CUDA-PRECISION.md)：FP8 Linear 与文本 TP8、BF16 attention / 通信、50 层 DiT 常驻，以及 VAE 和音频内存修复。NPU 默认保持不变；可显式选择旧 CUDA 路径。

CUDA 另提供第二套标准方案 [Sage-FP8](docs/H3-CUDA-SAGE.md)：Sage attention、FP8 双向通信、输出 AdaLN 调制融合和 `expandable_segments`。八卡 5090 的 5/10/15 秒、20 步 T2VA / I2VA 实测更快，六条成片均完整解码通过；人工画面、口型和听音审核仍待完成。原精度认可方案继续保留为默认。

v0.4 提供面向八卡 RTX 5090 R2VA 的优选 [Sage-FP8 全并行配置](docs/H3-CUDA-COMPOSABLE.md)：用 `cuda-5090 r2va` 显式选择，加入 SAP、本地输入、请求内条件复用、组件并行和去 BF16 临时副本，DiT 全程常驻。每项可单独关闭；原 `cuda`／`npu` 命令不自动开启。当前范围为八卡 Ref2VA 768P；组合正确性、显存容量和人工精度以各自实测记录为准。

v0.4 的 H3 在线接口将提示词上限从 2,000 放宽到 **7,000 个 UTF-16 码元**，按官方 CLI 的 JavaScript `text.length` 计数：常见汉字或英文字母各计 1，`🙂` 计 2。Base、Refiner、MaaS 共用该限制，在入队前拒绝超长文本，保留原文；这不是 tokenizer 的 token 上限。计数规则、错误响应和请求体上限见 [在线服务指南](docs/H3-ONLINE.md) 与 [MaaS API](server/video/MAAS-FL2VA-API.md)。

| 分支 | 核心变化 | 文档 |
|---|---|---|
| [v0.4](../../tree/v0.4) | 增加 `cuda-5090 r2va` 优选 Sage-FP8 全并行预设：SAP、本地输入投影、请求内固定条件复用、Token Refiner / 视觉 / VAE / 音频分工及 PHAA 直接打包，DiT 全程常驻；各项可独立关闭。旧 CUDA/NPU 默认保留，TP/CP 为默认关闭的实验选项。 | [启动与开关](docs/H3-CUDA-COMPOSABLE.md) · [实测性能](docs/performance/h3-cuda5090-full-parallel.md) · [在线启动](docs/H3-ONLINE.md) |
| [v0.3.2-maas](../../tree/v0.3.2-maas) | CUDA 提供默认精度参考 fp8-bf16 和更快的 Sage-FP8 两套标准方案；增加输出 AdaLN 融合与长视频实测。保留 MaaS 排队、超时、健康探针、原子发布、环境采集，以及 NPU 离线 benchmark、R2VA 多媒体参考、实际步数语义和逐视频显存回收。 | [CUDA 方案一](docs/H3-CUDA-PRECISION.md) · [CUDA 方案二](docs/H3-CUDA-SAGE.md) · [API](server/video/MAAS-FL2VA-API.md) · [在线启动](docs/H3-ONLINE.md) · [benchmark](docs/H3-NPU-BENCHMARK.md) · [升级说明](docs/H3-UPGRADE.md) |
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

```bash
# 768P
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 10 768p
bash example/diffusers/scripts/run_h3.sh npu r2va 15 768p

# 2K，需先准备 Refiner
bash example/diffusers/scripts/run_h3.sh npu t2va 5 2k
bash example/diffusers/scripts/run_h3.sh npu i2va 10 2k
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k
```

### NVIDIA CUDA 启动

#### RTX 5090 R2VA 优选：Sage-FP8 全并行

适用于八卡 RTX 5090、R2VA、768P。使用完整 Ref2VA 权重；不需要 NVFP4、稀疏权重或离线 AdaLN 表。先设置模型目录和自己的 H3 prompt：

```bash
export BASE_MODEL=weights/h3-ref2va
export PROMPT_FILE=inputs/prompt.txt
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png
```

生成五图参考的 15 秒视频：

```bash
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 15 768p inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

`cuda-5090` 是脚本的位置参数，不是 `--cuda-5090`。R2VA 时自动启用 `sage-fp8-full-parallel`；主干仍为 SP8，DiT 全程常驻。全部单项开关、关闭预设、内存代价和在线 JSON 见 [完整教程](docs/H3-CUDA-COMPOSABLE.md)。原 `cuda`、`npu` 命令不会自动启用新预设。

已测 768P、20 步：单图 5/10 秒 55.97/97.54s，五图 5/10/15 秒 139.67/224.81/316.49s；峰值 allocated 27.93–28.33 GiB/卡。耗时含生成保存，不含初始化和预热；技术检查与抽帧通过，完整动态、口型和人工听音仍待复核。[性能表及口径](docs/performance/h3-cuda5090-full-parallel.md)

全并行预设当前只面向 R2VA 768P。T2VA/I2VA、其他 CUDA 设备及 NPU 请使用下面各自的方案；不能套用 R2VA 的性能与精度结论。

#### 原 CUDA 路线和 T2VA / I2VA

先准备 [完整 FL2VA 权重和依赖](docs/H3-CUDA-PRECISION.md#模型与环境)。CUDA 768P 的 T2VA / I2VA 默认使用 **FP8 Linear、BF16 attention / 通信、文本 TP8、PHAA2、50 层常驻**，八卡启动。5 秒、20 步样例已获人工精度认可；R2VA 保留原路线，精度仍待验证。建议先从 5 秒试跑。

| 标准方案 | 选择方式 | 适用范围与审核状态 |
|---|---|---|
| 一：精度参考 `fp8-bf16` | 默认，不需新增开关 | 八卡 T2VA / I2VA；5 秒、20 步固定样例人工精度认可 |
| 二：更高性能 `fp8-sage` | 命令前加 `H3_CUDA_PRECISION=fp8-sage` | 八卡 T2VA / I2VA 768P，5/10/15 秒、20 步技术通过；人工精度待审 |

```bash
bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda r2va 5 768p
```

用 `H3_BASE_STEPS=20` 可做 20 步复测；用 `H3_CUDA_PRECISION=none` 显式回到旧 CUDA 路径。长视频须单独验证峰值和画质。

**方案二：Sage-FP8。** 脚本自动开启 FP8 通信、输出 AdaLN 融合和 `expandable_segments`，无需再拼接环境变量。准备对应时长的 prompt 后，时长可改为 `5`、`10` 或 `15`。下面按已测的 20 步启动：

```bash
H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
H3_CUDA_PRECISION=fp8-sage H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda i2va 15 768p
```

两套方案均使用 FP8 Linear 和文本 TP8、50 层常驻及 VAE 空间 tile。方案二需安装 SageAttention CUDA 扩展；不是 NVFP4 或稀疏方案。本段是 T2VA/I2VA 的原 Sage 配方；R2VA 全并行另见上文。2K、四卡及在线端到端的验证范围分别记录，不能复用离线结果。范围见 [方案二教程](docs/H3-CUDA-SAGE.md)。

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

NPU 默认使用 Laser、INT8 线性层和 50 次 Base DiT。Spark-H3、Veda 和 LightX2V 8 步另见 [可选加速配方](docs/H3-ATTENTION.md)。

H3 在线、离线均在每条视频保存完、临时输出释放后同步设备并回收 PyTorch 空闲显存，保留常驻模型。日志记录回收耗时和前后显存；该策略对单条请求内部的峰值 OOM 不作保证，当前策略的完整设备矩阵待验证。

## B2/B3 性能

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
| xGen 当前默认 fp8-bf16 | T2VA，5s/768P，**20 步** | 56.75 s | 历史常驻生成至保存，取最慢 rank；本轮样例人工精度可接受 |
| xGen 当前默认 fp8-bf16 | I2VA，5s/768P，**20 步** | 72.67 s | 同组模型复用；首次图像条件请求，样例人工精度可接受 |
| xGen CUDA R2VA | 五图，768P | — | 待完整推理、音视频复核和计时 |

[详细 GPU 性能与对照口径](docs/performance/h3-cuda.md)

**CUDA 两套标准方案的 20 步参考**：8×RTX 5090、SP8、768P；单位为生成至 MP4 保存的秒数，不含模型初始化。Sage 每项先完整预热 1 步，复用同一组模型。历史基线的预热条件存在差异，比较的是整套配方，不能归因为单个算子收益。方案一仅 5 秒样例已人工认可，Sage 六项均为技术通过、人工待审。

| 任务 | 时长 | 方案一 BF16 attention / 通信 | 方案二 Sage / FP8 通信 |
|---|---:|---:|---:|
| T2VA | 5s | 56.75 | **42.93** |
| I2VA | 5s | 72.67 | **48.09** |
| T2VA | 10s | 148.76 | **76.23** |
| I2VA | 10s | 164.70 | **97.81** |
| T2VA | 15s | 293.38 | **148.15** |
| I2VA | 15s | 308.16 | **161.12** |

方案一的 15 秒行额外开启输出 AdaLN 融合和 `expandable_segments`，不是未加开关的默认路径。15 秒峰值 allocated 仍约 29.88 / 30.00 GiB/卡，Sage 主要提速，没有明显降低该形状的峰值。详细阶段、显存和预热口径见 [六项实测](docs/H3-CUDA-SAGE.md#性能与验证)。

整理后的默认配置另做了八卡 20 步回归：T2VA 57.67 s、I2VA 88.70 s。两条 MP4 均与上述人工认可样例的文件哈希完全一致；新机器首次图像条件请求没有单独预热，具体口径见详细记录。

## 在线服务

在另一个终端提交请求。下面的示例都用本机 5001 端口，启动其中一个即可。

### NPU 在线

768P 服务支持 T2VA 和 I2VA。要部署 R2VA，按 [NPU 在线配置](docs/H3-ONLINE.md#npu-base) 切换任务类型和 Ref2VA 权重。

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

### CUDA 在线

普通 768P I2VA 默认使用 FP8 Linear＋BF16 attention。它与 NPU Base 使用同一套提交、查询和下载接口。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode fl2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001
```

原服务用 `/generate` 提交。Base HTTP 接受 5 至 15 秒的整数时长。 [各任务的请求和下载示例](docs/H3-ONLINE.md#提交原服务请求)

Sage-FP8 在线启动配置见 [方案二在线说明](docs/H3-CUDA-SAGE.md#在线服务)。它复用原接口，属于尚未完成在线端到端实测的部署示例；上述六项性能来自离线常驻请求。

### 2K 在线

NPU T2VA 可直接用下面的命令启动。服务接受 5、10 或 15 秒。I2VA、R2VA 和 CUDA 2K 使用各自的配置。 [配置方法](docs/H3-ONLINE.md#2k-refiner)

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29558 \
  server/video/video_server.py --init --config example/diffusers/configs/h3_refiner.example.json \
  --host 127.0.0.1 --port 5001
```

### MaaS 在线服务

MaaS 固定为 CUDA I2VA、5 秒、768P 横屏、50 步。默认采用 fp8-bf16；准备好完整 FL2VA/Base 模块权重后启动：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas-fl2va-input \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

通过 `POST /v1/tasks` 提交首条预热请求。预热成功后，等 `/ready` 返回 HTTP 200 再接流量。默认可排队 10 条，等待上限 600 秒。 [预热与高性能开关](docs/H3-CUDA.md#maas-在线服务) · [MaaS 接口](server/video/MAAS-FL2VA-API.md)

## Agent 使用指南

仓库附带 [xgen-h3 skill](.agents/skills/xgen-h3/SKILL.md)。它按设备、任务、分辨率、时长和在线或离线方式选择启动命令，列出权重要求和验收步骤。

## 发布范围

仓库包含源码、示例、测试和脱敏性能数据。权重、内部机器配置、原始诊断报告、测试素材和成片由部署方自行管理。其他模型和界面入口见 [原版安装与模型说明](../../tree/v0.3.1#环境安装)、[Diffusers 示例](example/diffusers)、[ComfyUI 适配](x_comfyui) 和 [工具说明](tools/README.md)。

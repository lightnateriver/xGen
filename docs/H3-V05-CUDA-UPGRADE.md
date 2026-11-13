# v0.5 CUDA 默认方案与提交范围

GPU 的 Sage-FP8 全并行 Streaming 预设现默认包含同步 PHAA FP8 Align16，沿用现有 transport，设置 `XGEN_PHAA_FP8_ALIGN16=0` 可独立关闭。在线 native-stream 生成的文件通过既有原子发布器交付，在线15秒按360帧交付。该文件交付修复与批测配置一起验证。[Align16配置与回退](H3-PHAA-FP8-ALIGNMENT.md) · [12项在线性能及验证范围](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda-phaa-align16.md)

本次基于 v0.5 的 `1883308` 整理 CUDA 开发代码。保留其 NPU Laser INT8 + Stream 默认、跨设备组件调度和共用 VAE 实现。

| 项目 | 原 v0.5 | 本次 |
|---|---|---|
| CUDA 八卡 H3 Base 默认 | T2VA/I2VA 为 fp8-bf16；R2VA 需显式选择全并行 | T2VA/I2VA/R2VA 统一 Sage-FP8 全并行 Streaming + BF16 AdaLN 表 |
| 条件组件调度 | CUDA 全并行预设主要限定 Ref2VA；已有通用调度 | CUDA FP8 配方复用同一调度，开放到 FL2VA/T2VA |
| AdaLN | CUDA rank-owner 权重与请求内精确缓存 | 新增表工具、配对校验、剔除权重加载和按请求步数选表；owner 回退保留 |
| 参考图尺寸 | 原模型短边 2048 | 新增可配置短边，默认仍 2048；输出画布与输入限制独立 |
| CLI / HTTP | 现有入口、MaaS 协议和队列 | 共用配置解析，新增选项为服务启动参数；接口协议不变 |
| NPU / WAN / Refiner / NVFP4 | 原有配方 | 默认选择不覆盖这些路径；NPU 数值和显存调度保持原 v0.5 |

新默认限定八卡 CUDA H3 Base、原生 SP8 与文本 TP8，主要验证设备为 RTX 5090。其他卡数、设备和 2K 不继承八卡 768P 的性能／容量结论。低层入口显式选定的精度、attention、量化、TP/CP 或旧预处理资产优先。脚本 CUDA Base 默认仍为 50 步，性能复测可设 `H3_BASE_STEPS=20`。

## 目录与职责

| 文件 | 职责 |
|---|---|
| `adaptor/h3_cuda_options.py`、`h3_cuda_precision.py` | 默认配方与启动校验、沿用现有 CUDA 适配器 |
| `adaptor/h3_cuda_adaln.py` | 独立负责表资产、匹配检查、常驻表选择 |
| `x_base/adaln_cache/cache.py` | 原缓存加载器兼容完整条件表；旧表格式继续可用 |
| `adaptor/h3_references.py`、`load_pipe.py` | 参考图短边配置与既有加载接入 |
| `tools/adaln_cache/prepare_cuda_adaln.py` | 一次性准备表与剔除权重的 checkpoint |
| `example/diffusers/scripts/run_h3.sh`、服务参数 | 复用原 CLI / Manager / HTTP 启动方式 |
| `x_diffusers/tests/` | 默认选择、回退、资产匹配、常驻键、尺寸与 NPU 启动契约测试 |

没有新增解码、队列、KV 缓存或批调度框架，没有新重依赖。没有合入 SP4、TP/CP 新实验、Max OOM 临时规避、过程 runner、媒体、权重或机器信息。

## 部署步骤

1. 按 [AdaLN 教程](H3-CUDA-ADALN.md) 分别准备 FL2VA / Ref2VA 的 20／50 步资产。
2. 按 [全并行教程](H3-CUDA-COMPOSABLE.md) 设置完整 Base 模型及可信 native VAE 目录。
3. 用原 `run_h3.sh cuda ...` 或在线 JSON 启动；日志应显示 `fp8-sage`、`native-stream`、`BF16 table, weights absent`。
4. 回退查表用 `--h3_cuda_adaln_mode owner`；回退旧精度用 `--h3_cuda_preset none --h3_cuda_precision fp8-bf16`。改变启动配方需重启服务。

最新 [12 项 Streaming 矩阵](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda-streaming-matrix.md) 逐条标明 owner／table，不能作为统一 table-only 的性能对照。Max 1:4 多图场景在视觉位置插值阶段仍可能 OOM，尚未修复；输入合法不代表显存容量足够。

整理后版本已完成 T2VA、I2VA、单图 R2VA 三条八卡5秒20步冒烟，均技术通过；验证配置、耗时与精度范围见[本次合并回归](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-v05-cuda-release-validation.md)。

# v0.6：八卡 910B3 VC＋融合验证

2026-11-12，在既有 **Laser＋DiT INT8＋full-parallel＋Native-stream** 配方上，仅替换 attention 为 VC 并启用适用的 Ascend 融合。新增能力默认关闭，原有视频启动默认和 H3-Image 独立入口保留。

## 同机性能

单位：秒。每组每种形状先预热一次，再正式测一次；组内模型常驻，5s、10s 不重新构建模型。E2E 是 `InferenceManager.infer` 调用耗时减去 `_prepare_pipeline` 耗时，至完整 MP4 保存返回。DiT 是 20 次 forward 的设备事件跨度之和。每项取八个 rank 中的最大值，初始化和预热不计入正式结果。

| 单图 R2VA | Laser E2E | VC＋融合 E2E | E2E 降幅 | Laser DiT | VC＋融合 DiT | DiT 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| 5s | 85.794 | 77.202 | 10.01% | 70.694 | 62.207 | 12.01% |
| 10s | 193.809 | 170.123 | 12.22% | 174.784 | 151.042 | 13.58% |

本次候选与此前提供的 v0.5_op 参考对比：

| 单图 R2VA | v0.5_op 参考 E2E / DiT | 本次 E2E / DiT | E2E 对参考差异 |
|---|---:|---:|---:|
| 5s | 80.212 / 64.171 | 77.202 / 62.207 | −3.75% |
| 10s | 173.930 / 152.076 | 170.123 / 151.042 | −2.19% |

参考表没有完整配置、输入哈希和计时定义，故不能仅凭秒数接近认定严格同条件复现。迁移收益以上方同机新鲜 Laser 基线为准。每 case 一次正式测量不代表长期时延分布。

## 显存和生命周期

| 单图 R2VA | 后端 | peak allocated (GiB) | peak reserved (GiB) |
|---|---|---:|---:|
| 5s | Laser | 31.121 | 34.121 |
| 5s | VC＋融合 | 31.121 | 34.570 |
| 10s | Laser | 31.183 | 34.426 |
| 10s | VC＋融合 | 31.183 | 36.750 |

allocated 与 reserved 是 PyTorch allocator 指标，不等于驱动 HBM。相同形状从预热到正式请求，八个 rank 的请求结束 allocated 均无增长；reserved 有小幅变化。两组共 16 个 rank PID 全部退出，RSS 为 0，八卡无计算进程，HBM 回到约 3426–3436 MB/卡。没有 OOM。

请求结束快照仍持有用于数值诊断的最终 latent 引用，不是生产服务完全空闲的 allocator 基线。已有约 10s HBM/RSS、约 30s PSS 采样；本轮两个形状不能证明长期无泄漏。

## 输入、形状与环境

| 项目 | 配置 |
|---|---|
| 设备 | 8×Ascend 910B3，64 GiB/卡 |
| 模型 | 原 H3 Ref2VA 权重、对应 20 次评估的 21 点 AdaLN；无 LoRA、无 Refiner |
| 输入 | 单张 2560×1440 参考图；同名 case 的固定 prompt 与 seed 42 |
| 输出 | 1344×768、24fps；5s=124 帧，10s=243 帧 |
| 并行 | SP8、文本 TP8、PHAA2、full-parallel |
| 精度及解码 | DiT INT8；VAE matmul quant none；Native-stream |
| Attention | Laser → VC；不开稀疏、Veda、Refiner 或 2K |
| Python / Torch | 3.12.13 / 2.10.0+cpu |
| torch-npu / Diffusers | 2.10.0.post4 / 0.40.0 |
| Transformers / MindIE-SD | 5.14.1 / 3.1.0 |
| CANN / npu-smi | 9.1 / 25.5.1 |
| 基线源码 | `84c0f03946be5167b4e99513f7e17ff9450a4893` |
| 设备验证的候选源码 | `01975c8865174fa6ac6d6e481ba974dbf7dd7fda` |
| VC 依赖 | `161108e910d86b4db954d5fb18b221ce876e2bed` |

VC 是 dense INT8 QK / FP16 ExpCast / FP16 PV，输入仍为 BF16 BNSD。实际进入 attention 的形状如下；Q/K/V 本轮相同：

| 时长 | 调用位置 | BNSD | 每 rank / 请求调用数 |
|---|---|---|---:|
| 5s | 输入 Token Refiner | `[1,7,7831,128]` | 2 |
| 5s | 主 DiT，PHAA 两块 | `[1,4,52837,128]`、`[1,3,52837,128]` | 各 1000 |
| 10s | 输入 Token Refiner | `[1,7,7951,128]` | 2 |
| 10s | 主 DiT，PHAA 两块 | `[1,4,88633,128]`、`[1,3,88633,128]` | 各 1000 |

## 实际命中与限制

八个 rank 的两个正式请求均确认 VC 实际派发，没有将 SDPA 回退计作 VC 结果。每 rank / 请求的融合计数为：

| 融合 | 命中次数 |
|---|---:|
| shared_qkv | 1000 |
| rope | 1000 |
| gate_residual | 2000 |
| norm_quant | 1000 |
| swiglu | 1000 |

`--h3-opts all` 表示请求开启，不代表九项全部命中。`phaa_layout`、`merge_quant` 的专用 SP4/PHAA7 几何不适用于本轮 SP8/PHAA2，按原路径回退；`laser_preprocess` 交给 VC 自带预处理；`text_cache` 没有新增命中，沿用 full-parallel 已有条件预计算。此轮没有验证 SP4 专用算子的设备性能或 2K / H3-Image 的新增配置。

## 正确性与验收范围

- CPU 回归：226 通过，1 项因需要 CUDA 硬件而跳过；覆盖开关默认值、回退、缓存生命周期和非 H3 兼容性等。
- 目标环境：七个融合库编译成功；6 项 Meta 检查、10 项设备数值 sanity 检查和 VC 安装 smoke 通过。
- 每条正式请求、每个 rank 实际 DiT 调用数为 20；两组内的模型身份及 Native VAE CPU master 复用检查通过。
- 四条正式视频全帧解码及尺寸、帧数、fps、音频流检查通过；20 张时序抽帧未见明显新损坏。
- 2026-11-12，评审者确认两个时长的 VC 画面和声音均可接受，满足本次“精度不退化且性能提升”的合并条件。此结论限定于所评审样例，不等同所有输入的质量保证。

VC 与 Laser 的最终 latent / 音频波形不是逐元素一致。数值 cosine 不能单独替代生成质量判断；本次采用技术检查、抽帧和完整视频/音频评审。

Native-stream 未经过旧 `SaveAVStream.close` 路径，因此没有独立测得编码/封装等待，不能记为 0 秒。Native VAE 的 `decode_output` 包含解码与输出整体，不是纯编码耗时。

## 开启与回退

先按[构建与依赖说明](../H3-NPU-FUSIONS.md)准备可选算子与 VC。模型、AdaLN、原生 VAE 及输入通过既有环境变量配置，在仓库根目录执行：

```bash
export H3_BASE_STEPS=20
export PROMPT_FILE=inputs/prompt.txt
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- \
  --atten vc --h3-opts all
```

这是冷启动功能示例；本表是常驻进程预热后的正式请求。10s 改用对应 prompt 和时长参数。关闭新增能力使用：

```bash
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- \
  --atten laser --h3-opts none
```

原脚本不加新参数时仍使用既有 Laser 配方。原始视频、输入、prompt、监控日志及私有机器路径不随代码发布。

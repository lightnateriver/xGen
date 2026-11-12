# R2VA 条件准备重叠（实验性，默认关闭）

`--h3_condition_overlap` 让参考图片／视频的 VAE 编码与 Qwen3-VL 文本／视觉编码同时进行。两支条件全部准备好后才进入 DiT。每个请求只准备一次，不改变 DiT 步数、Attention、量化精度、输入尺寸或现有 offload 策略。

CUDA 和 NPU 使用同一个参数及实现，不需要设备专用开关。当前只支持 **MiniMax-H3 Base Ref2VA**，原生 SP 布局；不支持 H3-Image、T2VA/I2VA、2K Refiner 协调器或 TP/CP 实验布局。CUDA 的真实验证与 NPU 的待验证范围见[实测记录](performance/h3-condition-overlap.md)。

## 开启与回退

给原有 R2VA 离线启动命令追加：

```bash
--h3_condition_overlap
```

例如沿用现有权重、素材和调度配方：

```bash
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png -- --h3_condition_overlap
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- --h3_condition_overlap
```

`--` 分隔参考文件和额外启动参数。直接使用 `torchrun` 时，把 `--h3_condition_overlap` 追加到原来的 Python 推理参数中即可。开关本身不自动开启视觉 SP、VAE 并行、FP8、INT8 或常驻权重，也不改变参考媒体上限。

在线服务在原启动命令中追加同一参数，或在服务的启动配置 JSON 中设置：

```json
{
  "h3_condition_overlap": true,
  "h3_condition_overlap_timeout": 600.0
}
```

这是**模型服务启动配置**，不是 `/generate` 或 MaaS V2 的逐请求字段。开启／关闭或修改等待时间后需重启服务。未设置时走原串行链路；若启动 JSON 已开启，省略 CLI 开关会保留 JSON 的值，显式传 `--no-h3_condition_overlap` 才覆盖关闭。所有现有 CUDA/NPU 预设默认仍关闭此项。

## 计算与通信

原串行路径：Qwen3-VL 编码并 offload，再进行参考 VAE 编码，随后运行 DiT。

开启后：同一组设备上，后台线程和独立 stream 执行原参考编码 block；主线程执行原 Qwen3-VL block。复用原有 VAE tile 分配、卷积和拼接，不建立另一套编码器。只合并参考 latent 输出，不覆盖文本分支的状态。

多卡时，VAE tile 结果通过 **CPU/Gloo** 汇总，保持原 dtype 和字节，再复制回原设备完成原生拼接，避免与文本 NCCL/HCCL 交错等待。只在开关启用时创建 CPU 通信组。参考音频（含视频内嵌音轨）继续使用原编码器和原精度，并等待所有文本 rank 完成后再开始，避免引入并发设备通信。

Qwen3-VL 原有分组 offload 保留，但其计算期间会与 VAE 权重、激活和工作区同时占用显存，因此**峰值可能上涨**。编码完成后的显存下降不抹去此前的峰值；`allocated` 与 PyTorch 的 `reserved` 缓存也应分开观察。

## 等待、日志和错误

| 参数／指标 | 含义 |
|---|---|
| `h3_condition_overlap` | 默认 `false`，启用条件分支重叠 |
| `h3_condition_overlap_timeout` | 默认 600 秒，正有限数；用于参考 future、音频门控和 CPU 集体通信的单次等待，不限制 DiT 或整个请求的推理时长 |
| `text_wall_s` | 主线程执行原文本 block 的墙钟时间，音频场景包含文本完成门控 |
| `reference_wall_s` | 参考 block 的墙钟时间，包含 VAE 编码、CPU/Gloo 汇总及必要等待 |
| `join_wait_s` | 主线程在参考 block 位置等待后台分支剩余工作的时间 |
| `submitted` / `joined` | 本请求后台参考任务提交／合并次数，正常有参考输入时各为 1 |
| `vae_cpu_gathers` / `vae_payload_bytes` | 本 rank 的 VAE CPU 汇总次数及发送 payload 字节数 |
| `failed` | 本请求是否因计算、等待或通信错误使该重叠 worker 不可继续复用 |

每请求结束记录 `H3 condition overlap` 日志，最新指标位于 `pipe._h3_condition_overlap_metrics`。墙钟区间可以重叠，不能相加作为端到端耗时。禁用 `DEBUG_TIME` 的额外设备 barrier，改用上述指标。

不会自动重试或切回串行路径。错误传播时尽量排空后台任务；超时或失败后，后续请求会明确要求重启该 manager/service，避免复用状态不明的 worker。关闭开关后重启即可回滚。

该功能使用 PyTorch 已有线程、stream 和 Gloo，无新增运行依赖。峰值显存增加的负载、GPU/NPU 都需单独评估；不能用五图结果保证最大满载请求可运行。

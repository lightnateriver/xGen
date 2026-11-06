# H3 步数、参考输入与显存回收升级说明

本次更新继续使用 `v3.2-maas` 分支。对比上一发布提交 `ef1a321`，主要增加 NPU 离线 benchmark、统一的 R2VA 多媒体输入处理和请求结束后的显存回收。

## CUDA Base 默认配方更新

CUDA H3 T2VA / I2VA 的 `--h3_cuda_precision auto` 默认选择 `fp8-bf16`：FP8 Linear、BF16 attention / 通信、文本 TP8、PHAA2、50 层常驻。需要完整原始模块权重，默认示例不再要求 packed NVFP4 文件。当前配方要求八卡；启动配置、CPU 锁页权重成本和验证范围见 [CUDA 默认配方](H3-CUDA-PRECISION.md)。

NPU、其他模型、R2VA，以及显式 packed NVFP4 / Refiner / LoRA / 其他 attention 配置保留既有路线。可用 `--h3_cuda_precision none` 回到旧 CUDA 路径。服务只能在启动时选择配方，修改后需重启；MaaS 请求格式、排队和超时参数不变。本次不合入新的 NVFP4 attention、FP8 通信或 AdaLN gather 实验。

## 步数现在表示实际 DiT 次数

普通 H3 Base、Base + Refiner、HTTP 和 MaaS 的 `num_inference_steps` 现在直接表示 Base DiT 次数。底层 MiniMaxH3Scheduler 需要的末尾 sigma 点由 adaptor 补齐，原有采样器及 Euler 更新公式保留。Wan 等其他模型的参数传递不变。

| 原路径 | 旧配置 | 新配置 | 保留的实际 Base 次数 |
|---|---:|---:|---:|
| 普通 H3 Base / Refiner / Veda | 51 | 50 | 50 |
| 普通 H3 Base / Refiner / Veda I2VA | 21 | 20 | 20 |
| LightX2V Turbo8 | 9 | 8 | 8 |
| Turbo-SLA，自身已补末尾点 | 4 | 4 | 4 |
| 其他模型，例如 Wan | 20 | 20 | 由原采样器决定 |

不要给所有历史 H3 配置机械减一。先确认该路径此前传的是调度点数还是实际执行次数。当前示例、服务参数说明和校验均已同步。MaaS 的固定请求参数从 51 改为 50，仍执行 50 次 Base DiT。

AdaLN 生成工具 `--steps 20` 现在生成 20 次 DiT 对应的表。已有 `num_steps=21, num_eval_steps=20` 的表可以复用，无需改名或重算。校验读取文件元数据，不再根据文件名猜测；表与实际步数不一致时拒绝执行。Refiner 另执行 3 次，不计入 Base 参数。

## R2VA 多媒体参考

Base 和 Refiner 复用 `adaptor/h3_references.py` 的加载、限制与 AdaLN 条件选择。参考媒体按输入顺序传给 H3，视频是否带音轨也参与条件选择。

| 输入 | 上限 |
|---|---|
| 图片 | 9 张 |
| 视频 | 3 段，每段 2–15 秒，视频累计不超过 15 秒 |
| 独立音频 | 3 段，每段 2–15 秒，音频累计不超过 15 秒 |
| 图片＋视频＋独立音频 | 总计 12 个文件 |

有声视频仍算一个视频文件，音轨随该视频加载。音频不能单独作为全部参考，需至少一张图或一段视频。无效、重复、损坏或超限参考会报错，不再静默跳过未知类型。图片像素数上限为 4096×4096。该输入范围不代表所有硬件都能在所有组合下容纳峰值显存，完整 CUDA 多媒体矩阵尚未验证。

## 每条视频保存后回收空闲显存

H3 的离线 Manager、HTTP 主 rank 和其他 worker rank 在各自请求输出使用结束后，释放临时输出引用，执行 `gc.collect()`、设备同步、PyTorch `empty_cache()` 和再次同步。Base 和 Refiner 共用同一回收函数，CUDA/NPU 按现有设备选择分派。常驻模型与 CPU 权重保留，不增加重新读盘、量化或模型构建。

`H3_REQUEST_MEMORY_RELEASE` 日志包含回收耗时及前后 allocated/reserved。离线推理及保存耗时包含回收；在线现有 inference 字段仍只表示生成阶段，回收单独记日志。该操作不卸载仍被引用的张量，也不能解决所有请求内部的峰值 OOM。

NPU 和旧 CUDA 路径中，解码前与 latent 上采样前的实验性清理已经移除；旧环境开关不再生效。新的 CUDA fp8-bf16 配方在视频 decoder 前另外回收空闲缓存，保留全部活跃 DiT 权重。[2026-11-04 至 05 的 35 项性能数据](performance/h3-npu-offline-20261105.md) 使用的是旧实验策略，不能作为本次回收策略的设备验收结果。当前完整矩阵待新一轮验证。

## benchmark 与素材

按 [NPU benchmark 教程](H3-NPU-BENCHMARK.md) 运行。每个工作组沿用原 Manager、Pipeline、权重调度、VAE 和编码器，不新增服务调度器或解码框架；指标包装仅在主动运行 benchmark 时启用。常规矩阵结束后不自动追加独立内存重复专项。

仓库提交用例定义、格式示例及脱敏性能数据。模型、AdaLN 大文件、原始日志、私人素材与完整成片不放进 Git 历史。公开素材包应另行记录生成来源、许可、输入哈希及下载校验值；更换素材后不能把性能当作同输入复现。

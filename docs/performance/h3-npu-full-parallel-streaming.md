# NPU Laser INT8 全并行 Streaming：T2VA / I2VA / R2VA

采集日期2026-11-11。八卡Ascend 910B1、64GiB/卡，原生SP8、文本TP8、PHAA2、Laser attention、INT8 DiT、融合RoPE、50层DiT常驻、21点AdaLN表对应实际20次DiT，输出1344×768/24fps，无2K Refiner。Native-stream decoder Linear以FP16存储并使用FP16 autocast，其余原生VAE权重保留FP32。

本次新增T2VA/I2VA六项，推理源码提交`d5ff2ad`，基于v0.5 `dbac26a`。I2VA使用单张原始1280×720首帧；T2VA无图。每项完整20步预热一次，再正式20步，一组八卡进程连续完成12次请求。首次模型准备八rank最大值106.450s单列，后续请求复用主模型和VAE CPU master；E2E排除模型准备、预热、HTTP排队/网络、进程启动及事后媒体审计。每规格仅一个正式样本，无自动重试或latent恢复。

## 新增六项正式性能

| 用例 | 常驻E2E s | DiT s | VAE H2D s | VAE解码/输出 s | allocated峰值 GiB/卡 | 进程组PSS峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| t2va-5s-768p | 50.518 | 38.840 | 0.473 | 4.559 | 30.859 | 158.051 |
| t2va-10s-768p | 124.129 | 109.011 | 0.478 | 7.773 | 30.906 | 157.924 |
| t2va-15s-768p | 236.735 | 218.987 | 0.473 | 11.278 | 30.953 | 157.968 |
| i2va-5s-768p | 55.313 | 41.750 | 0.378 | 5.240 | 31.052 | 159.955 |
| i2va-10s-768p | 129.510 | 113.022 | 0.488 | 8.115 | 31.099 | 159.983 |
| i2va-15s-768p | 243.695 | 225.184 | 0.481 | 11.366 | 31.146 | 160.161 |

耗时分别取八rank最大值，不能相加还原某一个rank的E2E。VAE解码/输出包含分块RGB8、D2H、编码和音频封装，其异步阶段重叠。DiT仍是主要耗时；不从单次样本声称全场景最优或稳定加速比。

| 用例 | 视觉编码 s | 语言编码 s | 输入图VAE编码 s | 输入Token Refiner s | 音频解码 s |
| --- | --- | --- | --- | --- | --- |
| t2va-5s-768p | 0.000 | 2.610 | 0.000 | 0.068 | 1.603 |
| t2va-10s-768p | 0.000 | 2.458 | 0.000 | 0.122 | 1.534 |
| t2va-15s-768p | 0.000 | 2.678 | 0.000 | 0.193 | 1.225 |
| i2va-5s-768p | 1.305 | 2.431 | 0.194 | 0.176 | 2.109 |
| i2va-10s-768p | 0.762 | 2.328 | 0.196 | 0.114 | 1.567 |
| i2va-15s-768p | 0.637 | 2.366 | 0.199 | 0.203 | 1.289 |

输入Token Refiner时间已包含在DiT中；T2VA无输入图，视觉和VAE编码均不执行。以上为常驻Manager推理保存口径，不是HTTP服务SLA。

## 优化实际执行与内存

所有12次请求的八rank均确认：实际20次DiT、主模型构建次数1、50层驻留、Native VAE CPU master对象及数据指针保持不变；条件投影计算1次/复用19次，VAE前释放、请求结束清空；本地输入和输出投影开启。没有独立的未使用Ref2VA DiT参数，`transformer_ref`为现有NPU管理代码维护的当前模型别名。

I2VA每请求实际执行视觉编码1次、VAE编码1次，输入VAE共28个tile，分给8卡；T2VA对应调用为0。两种任务的Native decoder均使用28个空间tile、8卡分工。5秒音频保留原解码，10/15秒长音频并行实际执行。T2VA/I2VA输入Token Refiner使用SP8、原attention、BF16 FFN、chunk0；R2VA原有chunk2048保留。

| 用例 | reserved峰值 GiB/卡 | 驱动HBM峰值 GiB/卡 | 请求后allocated GiB/卡 | 同形状预热→正式最大rank增量 bytes | 最大采样间隔 s |
| --- | --- | --- | --- | --- | --- |
| t2va-5s-768p | 33.211 | 38.754 | 24.335 | 0 | 18.464 |
| t2va-10s-768p | 33.279 | 40.848 | 24.335 | 0 | 18.406 |
| t2va-15s-768p | 34.359 | 41.485 | 24.335 | 0 | 19.411 |
| i2va-5s-768p | 34.021 | 35.521 | 24.518 | 0 | 19.013 |
| i2va-10s-768p | 34.268 | 36.736 | 24.518 | 0 | 18.840 |
| i2va-15s-768p | 35.307 | 37.928 | 24.518 | 0 | 20.011 |

同形状预热到正式的allocated增量均为0字节。T2VA稳定值24.335GiB/卡，I2VA切换到有图条件后为24.518GiB/卡；AdaLN条件表布局和输入工作区不同，不将模式切换直接视为跨请求泄漏。进程组PSS存在首次形状触达和分配器波动，有限请求不能证明长期无内存泄漏。

驱动HBM、各进程RSS/PSS和主机内存持续采样，目标等待10秒，实际间隔还包含采样开销；周期采样可能漏掉瞬时峰值，不能替代PyTorch高水位。全部本批worker已退出，八卡驱动HBM延后复查回到约3.35GiB基线。当前设备放置仍保留约4.51GiB旧decoder，不重新引入此前回退的CPU放置修改。

## 既有R2VA六项

以下为此前`1883308`的独立复测，相同硬件及20步/预热口径、原始2560×1440参考图；本次没有重跑R2VA。不同源码采集版本分开记录，不混称本次提交的一次全量回归。

| 用例 | 常驻E2E s | DiT s | allocated峰值 GiB/卡 |
| --- | --- | --- | --- |
| r2va-1img-5s-768p | 77.440 | 63.053 | 31.121 |
| r2va-1img-10s-768p | 168.158 | 149.994 | 31.168 |
| r2va-1img-15s-768p | 297.014 | 274.661 | 31.216 |
| r2va-5img-5s-768p | 250.134 | 225.296 | 31.434 |
| r2va-5img-10s-768p | 399.165 | 371.766 | 31.481 |
| r2va-5img-15s-768p | 592.318 | 561.126 | 31.528 |

此前Native Linear存储精度对照见[FP32/FP16匹配复测](h3-native-shared-fp16.md)。本页整套配置与旧Lightning路径的历史性能不用于归因单一组件收益。

## 默认启动与环境

[默认方案](../H3-NPU-DEFAULT.md)适用于T2VA/I2VA/R2VA，推荐NPU脚本和Base在线JSON均默认开启全并行与Streaming；底层CLI仍显式传入完整配置。T2VA/I2VA共用FL2VA权重和匹配表，R2VA使用独立资产。

```bash
export BASE_MODEL=weights/base
export BASE_TRANSFORMER=model-cache/base-t2va
export BASE_ADALN=model-cache/t2va-21step.safetensors
export H3_VAE_MODEL_PATH=weights/native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1  # 仅对确认可信的模型代码
export H3_BASE_STEPS=20
export PROMPT_FILE=inputs/prompt.txt
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 5 768p inputs/first-frame.png
```

10/15秒替换上述时长参数。实测固定环境：Python3.12.13、torch2.10.0+cpu、torch-npu2.10.0.post4、Diffusers0.40.0、Transformers5.14.1、mindiesd3.1.0、CANN9.1.0、driver25.5.1。启动环境含OMP_NUM_THREADS=8、HCCL_BUFFSIZE=512、TASK_QUEUE_ENABLE=2、COMBINED_ENABLE=1、MEMORY_FRAGMENTATION=1、ENCRYPTION_WEIGHT=false、INF_NAN_MODE_FORCE_DISABLE=1、PYTORCH_NPU_ALLOC_CONF=expandable_segments:True。镜像、模型、输入和编码参数在测试中冻结。

## 验证边界

默认配置、原有CUDA/后端选择、通用组件以及新增工作流加载检查：148项CPU回归通过，1项checkpoint设备冒烟跳过。12条矩阵视频均通过分辨率/帧率/帧数/音轨检查和FFmpeg完整音视频解码；5/10/15秒按原入口保存124/243/362帧。完整动态、身份、口型和人工听音验收待完成；单首帧实测不代表首尾帧全部组合已验证。

正式矩阵前另有一次5秒预热已完成推理，但采集器把当前模型别名误判为额外Ref2VA参数而退出。修正对象身份检查后重新启动完整矩阵，推理源码未改；该额外诊断视频和日志保留，不纳入性能表。

原始启动、模型/表/输入哈希、环境、逐rank指标、内存采样与视频保留于本地实验记录；公开仓库仅保留聚合数据与通用复现方法。[CSV](h3-npu-full-parallel-streaming.csv)

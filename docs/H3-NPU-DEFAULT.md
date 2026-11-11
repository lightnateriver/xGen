# v0.5 NPU Base 默认：Laser INT8 全并行 Streaming

v0.5 的推荐 NPU Base 768P 入口统一采用 **Laser INT8 全并行 Streaming**（原称 Laser INT8 + Stream），覆盖T2VA、I2VA和R2VA。DiT Linear 是 INT8；BF16 是主干接口与通信类型，Stream VAE 的 Linear 存储为 FP16，其余权重保留 FP32。三种任务复用同一个`full-parallel`调度及`native-stream`后端，不另建NPU解码组件。

| 配置 | 默认值 |
|---|---|
| 进程、DiT SP、文本 TP | 8 / 8 / 8 |
| Base实际计算步数 | 20，匹配21点AdaLN表 |
| attention / DiT Linear | Laser / INT8 |
| PHAA / RoPE | 2（4+3头）/ fused |
| 调度 | `h3_features=full-parallel`，DiT常驻 |
| 输入Token Refiner | SP8、原attention、BF16 FFN；R2VA chunk2048，T2VA/I2VA不分块 |
| 条件处理 | 本地输入/输出投影、请求内固定条件复用、Qwen视觉SP、VAE tile编码、音频并行 |
| 视频decoder | `native-stream`，`vae_matmul_quant=none` |

固定条件仅在一个请求的多个DiT步骤之间复用；每条请求重新处理素材，结束后释放，不是跨请求内容缓存。不启用未取得E2E收益的新增PHAA双通信流/通信缓冲实验、PHAA7或CUDA专属算子。

| 默认优化 | T2VA | I2VA | R2VA |
|---|---|---|---|
| DiT SP8、文本TP8、PHAA2、AdaLN查表、融合RoPE | 开启 | 开启 | 开启，使用Ref2VA资产 |
| DiT常驻、本地输入/输出投影、请求内条件复用 | 开启 | 开启 | 开启 |
| 输入Token Refiner SP、原attention、BF16 FFN | 开启 | 开启 | 开启，另加chunk2048 |
| Qwen视觉SP、VAE Encoder tile并行 | 无视觉输入时不执行 | 编码首/尾帧 | 编码参考图/视频 |
| Native-stream多卡tile解码、分块RGB8、异步D2H/编码 | 开启 | 开启 | 开启 |
| 长音频解码并行 | 10/15秒形状 | 10/15秒形状 | 10/15秒形状 |
| 参考音频编码并行 | 无参考音频 | 无参考音频 | 有参考音频时执行 |

T2VA/I2VA共用FL2VA权重，可在同一常驻Manager中切换有图/无图请求；条件切换按原接口重选AdaLN条件表，主模型和VAE CPU master继续复用。全并行入口仅加载该工作流的组件，避免加载未使用的Ref2VA DiT。参考图像较小时，VAE Encoder会按tile数回落原路径；Native-stream也保留少tile形状的安全回落。

`--h3_cuda_vae_sp`属于旧CUDA decoder路径；Native-stream直接管理完整SP进程组的tile并行，NPU无需该开关。CUDA的Sage、FP8通信/GEMM、cuDNN/Triton算子保持独立；通用调度在NPU中沿用Laser、INT8和BF16接口。

## 离线

准备原有Diffusers Base与剥离Transformer、匹配工作流的21点AdaLN表，以及原生模型的`video_vae`目录。后者不是转换后的Diffusers `vae`目录。

```bash
export BASE_MODEL=weights/base
export BASE_TRANSFORMER=model-cache/base-ref2va
export BASE_ADALN=model-cache/ref2va-21step.safetensors
export H3_VAE_MODEL_PATH=weights/native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1  # 仅用于你已确认可信的原生模型代码
export PROMPT_FILE=inputs/prompt.txt
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png
```

脚本默认20步，`H3_BASE_STEPS=50`可选择50步，同时须使用匹配的51点表。`TASK_QUEUE_ENABLE=2`、`HCCL_BUFFSIZE=512`、`OMP_NUM_THREADS=8`、`COMBINED_ENABLE=1`、`MEMORY_FRAGMENTATION=1`、`INF_NAN_MODE_FORCE_DISABLE=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`使用本轮启动值，显式环境设置优先。`ENCRYPTION_WEIGHT=false`。

T2VA/I2VA使用对应Base权重与表。本轮已完成八卡910B1、T2VA/单首帧I2VA的5/10/15秒完整复测，既有R2VA单图/五图六项独立记录；视频、阶段计数、内存和源码边界见[完整性能表](performance/h3-npu-full-parallel-streaming.md)。素材上限、首尾帧全部组合、50步及真实HTTP压测仍按各自验证范围报告。

## 在线与回退

在线默认与 `example/diffusers/configs/h3_npu_base.example.json` 对齐。先修改JSON的模型与原生VAE路径；示例中的trust仅在确认模型代码可信后使用。切换后端、存储布局和并行配置需要重启常驻服务。启动前设置上面的环境变量，再按[在线说明](H3-ONLINE.md#npu-base)启动。

R2VA独立实例使用`example/diffusers/configs/h3_npu_ref2va.example.json`：`task_type=ref2va`、`transformer_path=model-cache/base-ref2va`、`adaln_cache_file=model-cache/ref2va-21step.safetensors`及`h3_cuda_refiner_ff_chunk_size=2048`。T2VA/I2VA配置为0，避免误用R2VA专属分块参数。新默认的常驻推理E2E不代表HTTP SLA。

回退可以继续使用现有三选一selector，不删除旧后端：

```bash
unset H3_VAE_MODEL_PATH H3_VAE_TRUST_REMOTE_CODE
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- \
  --h3_vae_backend lightning --vae_matmul_quant int8 \
  --h3_features none --h3_refiner_ff_chunk_size 0
```

上面仍使用20步及对应表；回退到历史50步需同时设置步数与AdaLN资产。2K Refiner使用自己的启动脚本或`h3_refiner.example.json`，保留legacy VAE和原有协调器，不套用Native-stream，也不套用本页Base 20步的默认值。低层Python入口保留显式配置能力，推荐脚本/服务配置是本页默认部署入口。

## 性能与验证边界

| 同机匹配复测 | FP32原生Linear存储 | 共享FP16 Linear存储 |
|---|---:|---:|
| 八rank最大请求E2E | 78.189s | 77.644s |
| 每卡allocated峰值 | 40.137GiB | 31.121GiB |
| 八rank最大VAE H2D | 1.080s | 0.390s |

每组一次预热、一次正式；初始化排除。峰值减少约22.5%，单次E2E仅改善约0.7%，不宣称稳定时延加速。四条视频完整视频/音频技术解码通过，人工画质/音频验收待完成。实际显存已回落：八rank请求后allocated均24.518GiB，进程退出后回到约3.35GiB驱动基线。这两个请求不能证明长期无泄漏。

当前常驻基线仍含约4.51GiB旧decoder，按已回退的设备放置策略保留。该固定占用不属于跨请求泄漏，本版本没有悄悄重新引入旧decoder CPU放置修改。详细版本与环境见[实测记录](performance/h3-native-shared-fp16.md)。GPU配方及真实GPU验证范围保持独立。

## 发布回归

2026-11-11，在上述NPU容器中完成265项回归，2项跳过（CUDA专属音频测试、未配置checkpoint的Native VAE设备冒烟）。覆盖默认启动参数的真实解析、在线JSON、后端选择、FP16 Linear存储、通用调度、常驻启动参数不可热改、原有CUDA与Refiner配置契约。此次发布回归不另跑整网视频；性能与成片证据来自上述匹配实测，不表示所有组合已验收。

```bash
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers"
XGEN_TEST_NATIVE_VAE= python -m pytest -q \
  x_diffusers/tests/test_h3_launch_profiles.py \
  x_diffusers/tests/test_h3_vae_config.py \
  x_diffusers/tests/test_h3_omni_decode.py \
  x_diffusers/tests/test_h3_portable_features.py \
  x_diffusers/tests/test_h3_cuda_options.py \
  x_diffusers/tests/test_h3_condition_parallel.py \
  x_diffusers/tests/test_h3_cuda_conditioning_staging.py \
  x_diffusers/tests/test_h3_cuda_precision_config.py \
  x_diffusers/tests/test_h3_refiner_opt_in.py \
  x_diffusers/tests/test_h3_tp_cp.py \
  tools/test_bench_h3_npu.py tools/test_h3_parallel_config.py
```

兼容性测试以Git中的已发布v0.4提交为基准，运行时需要完整Git历史，浅克隆需先补齐历史。

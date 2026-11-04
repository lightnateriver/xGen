# xGen

xGen 是面向昇腾 NPU 和 NVIDIA CUDA 的音视频生成推理引擎。它复用 Diffusers 模型组件，提供离线命令行和在线 HTTP 服务，并为 MiniMax-H3 提供多卡序列并行、PHAA 通信、文本编码器 TP、模型分层加载和 VAE 并行。

MiniMax-H3 支持文本生成音视频（T2VA）、首帧图生音视频（I2VA，代码中为 `fl2va`）和多参考图生成音视频（R2VA，代码中为 `ref2va`）。昇腾路径支持 768P Base 和 Open-H3-Regenerate-2K 超分。CUDA 已实测 FL2VA Base 及可选 FL2VA Refiner。MaaS 在原在线服务上增加严格输入、异步任务队列、就绪探针和成片校验，由启动开关启用。

[版本](#版本) · [快速开始](#快速开始) · [昇腾启动](#昇腾-npu-启动) · [B2/B3 性能](#b2b3-性能) · [CUDA 启动](#nvidia-cuda-启动) · [GPU 性能](#gpu-性能) · [MaaS 在线](#maas-在线服务)

## 版本

| 分支 | 核心变化 | 文档 |
|---|---|---|
| [v3.2-maas](../../tree/v3.2-maas) | CUDA H3 FL2VA 增加默认关闭的 MaaS 接口。支持输入校验、可配置排队和超时、去重、取消、就绪探针、成片原子发布及分阶段计时。新增双后端环境采集。 | [API](server/video/MAAS-FL2VA-API.md) · [在线启动](docs/H3-CUDA.md#maas-在线服务) · [环境采集](tools/README.md#cuda-和-npu-环境采集) |
| [v0.3.1](../../tree/v0.3.1) | CUDA 配置集中到 adaptor。增加可选异步 FP8 PHAA、回程直写、DiT 算子融合和 VAE 空间切片。完成 CUDA、NPU 双设备冒烟。 | [实测](docs/performance/h3-cuda.md#v031-双设备冒烟) |
| [v0.3-cuda](../../tree/v0.3-cuda) | CUDA H3 FL2VA、NVFP4 DiT 常驻、Sage/Sparge、FP8 PHAA 双向通信、文本 TP8、VAE SP、FastVideo C4 预设及可选 FL2VA 2K。 | [CUDA 部署](docs/H3-CUDA.md) |
| [v0.3](../../tree/v0.3) | NPU Base DiT 增加可选 Spark-H3 和 Veda。支持 Veda 稀疏度配置和实验性 LightX2V 8 步配方。 | [启动和回退](docs/H3-ATTENTION.md) · [实测](docs/H3-ATTENTION-VALIDATION.md) |
| [v0.2](../../tree/v0.2) | 增加默认关闭的 H3 → Open-H3-Regenerate-2K 链路，复用原 Manager 和 HTTP 服务。 | [NPU 部署](docs/H3-DEPLOYMENT.md) |
| [v0.1](../../tree/v0.1) | 原始 xGen，包含 Diffusers 适配、SP、FSDP、PHAA、量化和视频服务。 | [基础示例](example/diffusers) |

## 快速开始

在匹配目标硬件的 Linux 运行环境中，从仓库根目录执行。安装命令适用于已准备好 PyTorch、硬件运行时及其他依赖的环境。

```bash
python -m pip install -e ./x_base --no-deps
python -m pip install -e ./x_diffusers --no-deps
python -m pip check
python tools/collect_env.py
mkdir -p outputs
```

采集脚本显示设备、驱动、软件依赖、CPU 亲和性、NUMA 和白名单环境变量。确认 PyTorch 可见卡数与启动的 SP 一致。权重、首帧、参考图和提示词由部署方准备。命令中的相对路径均以仓库根目录为基准。

## 昇腾 NPU 启动

推荐先跑 Laser + INT8 的 50 次 DiT 路径。8×Ascend 910B、SP8、文本 TP8、PHAA2、融合 RoPE 和匹配的 AdaLN 预计算是已记录的 NPU 性能配方。按 [权重下载与预处理](docs/H3-DEPLOYMENT.md#4-模型与目录准备) 准备 Base、Ref2VA 和可选 Refiner 资产。T2VA 和 I2VA 共用 Base Transformer/AdaLN，R2VA 使用独立的 Ref2VA 资产。

### 768P Base

下面按 1344×768、24 fps、51 个调度点启动。修改 `DURATION` 可选 5、10、15 秒档。传给模型的帧数包含时序对齐所需的帧，导出帧数以实际 MP4 为准。每行命令可单独运行。

```bash
export XGEN_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8 HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True INF_NAN_MODE_FORCE_DISABLE=1
DURATION=5
case "$DURATION" in
  5) FRAMES=124 ;;
  10) FRAMES=243 ;;
  15) FRAMES=362 ;;
  *) echo "DURATION must be 5, 10 or 15" >&2; exit 2 ;;
esac
PROMPT="$(cat inputs/prompt.txt)"

npu_768p() {
  local task="$1" transformer="$2" adaln="$3"
  shift 3
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29530}" \
    example/diffusers/infer.py \
    --model MiniMax-H3 --task_type "$task" --pretrained_model_name_or_path weights/base \
    --transformer_path "$transformer" --adaln_cache_file "$adaln" \
    --dtype bf16 --matmul_quant int8 --vae_matmul_quant int8 --atten laser --rope_fused \
    --vae_lightning decoder --sp 8 --text_encoder_tp 8 --phaa_num 2 \
    --num_inference_steps 51 --width 1344 --height 768 --frames "$FRAMES" --save_fps 24 --seed 42 \
    --prompt="$PROMPT" --save_path "outputs/npu-$task-${DURATION}s-768p.mp4" "$@"
}

npu_768p t2va model-cache/base-t2va model-cache/t2va-51step.safetensors
npu_768p fl2va model-cache/base-t2va model-cache/t2va-51step.safetensors --keyframe_image inputs/first-frame.png
npu_768p ref2va model-cache/base-ref2va model-cache/ref2va-51step.safetensors \
  --reference_paths inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

当前分支的 NPU FL2VA 双设备回归覆盖 5 秒 768P。其他形状的性能数据来自后文历史矩阵。若设备部署配置不同，先用一条请求验证素材处理、显存和导出。

Spark-H3、Veda 和 LightX2V 8 步有独立权重或精度验证边界，按 [可选 attention 与 Turbo 配方](docs/H3-ATTENTION.md) 单独启用。下方 B2/B3 表统一采用 Laser 和 50 次实际 Base DiT。

### 2K Refiner

使用同样的 Base 优化，以 768P latent 接 BF16 Refiner，Refiner 实际运行 3 步。下面的发布版脚本采用分层预取，输出 2560×1440、24 fps，5、10、15 秒分别裁剪到 120、240、360 帧。保留上面的 NPU 环境和 `PROMPT`。

```bash
export BASE_MODEL=weights/base REFINER_WEIGHTS=weights/refiner REFINER_PREPARED=model-cache/refiner
export SECONDS_TO_GENERATE="$DURATION" PROMPT

npu_2k() {
  local task="$1" transformer="$2" adaln="$3"
  shift 3
  BASE_TRANSFORMER="$transformer" BASE_ADALN="$adaln" \
    OUTPUT_VIDEO="outputs/npu-$task-${SECONDS_TO_GENERATE}s-2k.mp4" \
    bash example/diffusers/scripts/infer_minimax_h3_refiner.sh --task_type "$task" "$@"
}

npu_2k t2va model-cache/base-t2va model-cache/t2va-51step.safetensors
npu_2k fl2va model-cache/base-t2va model-cache/t2va-51step.safetensors --keyframe_image inputs/first-frame.png
npu_2k ref2va model-cache/base-ref2va model-cache/ref2va-51step.safetensors \
  --reference_paths inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

原 HTTP 服务可按工作流启动 2K。完整配置见 [NPU 在线 2K](docs/H3-DEPLOYMENT.md#8-原-http-服务的可选-2k-后端)。历史矩阵还使用了异步 RGB8/D2H 和编码流水线，发布脚本的导出与计时口径见 [NPU 性能说明](docs/H3-DEPLOYMENT.md#10-性能数据先看版本与测量边界)。

## B2/B3 性能

B2 是历史测试组，芯片实际识别为 **910B1**。B3 为 **910B3**。两组均是 8 卡、SP8、单卡 64 GiB，Base 50 次实际 DiT，2K 另加 3 次 Refiner DiT。I2VA 输入为 1280×720 首帧，R2VA 为五张 2560×1440 参考图。

单位为秒。常驻 E2E 包含生成和导出，初始化单列。每格一个样本，在线为客户端计时，离线为直接调用循环计时。两组执行顺序、首次形状触达和埋点存在差异。

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

## NVIDIA CUDA 启动

使用匹配 CUDA 的 PyTorch、Diffusers、FFmpeg 和八张可见 GPU。以下先给 BF16 稠密参考配方，再给已测的高性能 FL2VA 配方。 [依赖与完整开关](docs/H3-CUDA.md)

### 画质参考配方

CUDA 上曾标记「精度 OK」的 **T2VA** 是 vLLM-Omni 的 SP8、FP8 线性层、BF16 稠密 attention 和 BF16 All-to-All 单例结果，详见下方 GPU 表。xGen 本身的 BF16 稠密 FL2VA 有完整音视频出片记录，可用于目标输入的画质基线。**xGen T2VA 和 R2VA 的以下 CUDA 命令是候选启动方式，待端到端和人工精度复核。**

```bash
export XGEN_DEVICE=cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
unset XGEN_PHAA_FP8 XGEN_PHAA_DUAL_COMM XGEN_PHAA_GATHER_PRIORITY
unset XGEN_PHAA_DEVICE_JOIN XGEN_PHAA_ASYNC_EXCHANGE XGEN_PHAA_DIRECT_RETURN
unset XGEN_PHAA_FP8_ROW_TILING XGEN_PHAA_FUSED_RETURN_PACK XGEN_PHAA_DIRECT_GATHER XGEN_PHAA_CHUNK_ORDER
unset XGEN_H3_CUDA_VAE_TILES XGEN_H3_FUSED_INDEXED_AFFINE XGEN_H3_FUSED_PARTIAL_ROPE
unset XGEN_H3_FUSED_QK_RMS XGEN_H3_FUSED_FF_AMAX XGEN_H3_FUSED_SWIGLU XGEN_H3_NVFP4_FUSED_AMAX
PROMPT="$(cat inputs/prompt.txt)"

gpu_bf16() {
  local task="$1"
  shift
  torchrun --standalone --nproc_per_node=8 example/diffusers/infer.py \
    --model MiniMax-H3 --task_type "$task" --pretrained_model_name_or_path weights/base \
    --dtype bf16 --atten none --matmul_quant none --vae_matmul_quant none --vae_lightning none \
    --sp 8 --text_encoder_tp 8 --phaa_num 4 --h3_cuda_vae_sp \
    --width 1344 --height 768 --frames 124 --num_inference_steps 51 --save_fps 24 --seed 42 \
    --prompt="$PROMPT" --save_path "outputs/cuda-$task-5s-bf16.mp4" "$@"
}

gpu_bf16 t2va
gpu_bf16 fl2va --keyframe_image inputs/first-frame.png
gpu_bf16 ref2va --reference_paths inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

BF16 走分层 offload，需足够的主机内存。已有 FL2VA 实测使用完整 BF16 Transformer 和对应组件，若组件与 Transformer 分开存放，在该命令追加 `--transformer_path` 指向匹配的完整权重目录。R2VA 需要包含 Ref2VA workflow 和 Transformer 的模型布局。

### 高性能配方，跨用例精度待复核

适用于 FL2VA。使用匹配的打包 NVFP4 checkpoint、Sparge、FP8 PHAA C4、异步交换、回程直写、DiT 融合和空间 VAE。依赖 CUDA 版 Comfy-Kitchen、Sparge 扩展和相应 Triton。使用干净的启动环境，避免继承其他实验开关。

```bash
export MODEL_PATH=weights/h3 NVFP4_CHECKPOINT=weights/h3-fl2va-nvfp4.safetensors
export FIRST_FRAME=inputs/first-frame.png PROMPT_FILE=inputs/prompt.txt
GPUS=8 OUTPUT_PATH=outputs/cuda-fl2va-5s-fast.mp4 \
  XGEN_FASTVIDEO_RECIPE=5s ATTENTION_BACKEND=sparge XGEN_SPARGE_TOPK=0.5 \
  XGEN_PHAA_DEVICE_JOIN=1 XGEN_PHAA_ASYNC_EXCHANGE=1 \
  XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1 \
  XGEN_H3_FUSED_INDEXED_AFFINE=1 XGEN_H3_FUSED_PARTIAL_ROPE=1 XGEN_H3_FUSED_QK_RMS=1 \
  XGEN_H3_CUDA_VAE_TILES=1 XGEN_H3_FUSED_FF_AMAX=1 \
  bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh --h3_cuda_vae_sp
```

5 秒配方通过完整解码，并与同配方重构前的参考成片保持解码后哈希一致。跨输入画质仍需复核。`15s` 预设已有参数核对。CUDA 打包 NVFP4 和 2K Refiner 当前要求 FL2VA，T2VA 和 R2VA 采用上面的 BF16 候选入口。 [CUDA 2K 启动](docs/H3-CUDA.md#实验性-cuda-2k-refiner)

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
| xGen CUDA R2VA | 五图，768P | — | 待完整推理、音视频复核和计时 |

[详细 GPU 性能与对照口径](docs/performance/h3-cuda.md)

## MaaS 在线服务

使用 CUDA H3 Base 的 5 秒 768P FL2VA 入口。启用 `--init --maas-fl2va-input` 后，默认每实例 10 条等待任务，等待满 600 秒过期。使用 `--maas-queue-capacity` 和 `--maas-queue-timeout-s` 可调整两项默认值。任务执行沿用原八卡派发，`/ready` 在真实首单预热和 worker 确认后通过。

- [配置与八卡启动命令](docs/H3-CUDA.md#maas-在线服务)
- [提交、查询、下载、错误码和超时口径](server/video/MAAS-FL2VA-API.md)
- [原在线服务与历史入口](server/video/README.md)

## 参考与发布范围

昇腾章节按硬件与环境、权重准备、分场景启动和实测数据组织，参考 [Ascend-SACT MiniMax-H3-NPU](https://ai.gitcode.com/Ascend-SACT/MiniMax-H3-NPU)。性能表使用各条目所注明的 xGen 或 vLLM-Omni 实测来源。

仓库包含源码、示例、测试和脱敏性能数据。权重、内部机器配置、原始诊断报告、测试素材和成片由部署方自行管理。其他模型和界面入口见 [原版安装与模型说明](../../tree/v0.3.1#环境安装)、[Diffusers 示例](example/diffusers)、[ComfyUI 适配](x_comfyui) 和 [工具说明](tools/README.md)。

# XGEN H3 + Refiner：八卡部署、推理与性能教程

本文面向 `v0.2`。H3 Refiner 是 **experimental、默认关闭** 的可选超分后端；不开启时沿用原 XGEN 推理入口和设备策略。本文中的模型、素材和输出路径均相对于仓库根目录，实际数据不随仓库发布。

本文使用 xGen 的 Manager、HTTP 服务、VAE 和编码器。

## 1. 先选运行路径和数据口径

本页的原有命令使用 NPU。CUDA 环境、768P/2K 启动和 FastVideo 配方见 [CUDA 部署教程](H3-CUDA.md)，实测结果见 [CUDA 性能记录](performance/h3-cuda.md)。

| 路径 | 做什么 | 当前验证边界 |
|---|---|---|
| 原 XGEN / Base-only | 原脚本生成 768P，Refiner 不加载 | 保持原入口和配置行为 |
| v0.2 Refiner | Base 768P latent → latent upscaler → Refiner → 2K VAE → H.264/AAC | 八卡连续两个 T2VA 5 秒 2K 技术回归通过；完整在线服务及其他组合待新版本验收 |
| 历史优化版数据 | T2VA/I2VA/R2VA × 5/10/15 秒 × 768P/2K × 在线/离线 | 两种硬件各 36 条、共 72 条技术检查通过；其独立实验运行器不随 v0.2 发布 |

历史结果与 v0.2 使用不同的导出实现和部分优化配置；它们必须分开报告。本文不会用旧版 72 条结果证明新分支的全矩阵性能。

## 2. 硬件与环境

| 项目 | 记录 / 部署要求 |
|---|---|
| 硬件 | 8×Ascend 910B，SP8；历史设备 A 实际识别为 910B1，设备 B 为 910B3 |
| 单卡 HBM | 两组历史设备均为 64 GiB；其他容量未验证 |
| 驱动 | 历史采集为 25.5.1；部署还需核对驱动、固件与 CANN 的配套关系 |
| PyTorch / torch_npu | 测试记录：2.10.0+cpu / 2.10.0.post4；这里的 `+cpu` 是 Ascend 配套安装包标记，不表示推理在 CPU 上执行 |
| Diffusers | 测试记录：0.40.0 |
| Python / Transformers | 本次回归环境与仓库配置：Python 3.12、Transformers 5.14.1；历史两组环境的所有细项未完整归档 |
| CANN | 官方配套表将 torch_npu 2.10.0.post4 对应到 CANN 9.1.0；不要把配套要求当作历史两组机器均已归档的实测值 |
| 主机 RAM | 源模型、两套 CPU master、页缓存和导出缓冲需同时容纳；历史监控进程 PSS 合计峰值约 714.73 GiB，不能依据其他少步数教程的 512 GB 门槛保证复现 |
| 磁盘 | 需容纳 Base、Refiner 源文件、预处理 transformer/AdaLN、HF 缓存及视频；不要只按 LoRA 文件大小估计 |
| 工具 | Git、匹配运行环境、FFmpeg、FFprobe；HTTP 服务另沿用 Flask 与原存储客户端依赖 |

版本配套以 [Ascend PyTorch 官方表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 为准。先在平台提供的 910B 运行容器中完成驱动设备映射和 CANN 初始化；参考教程的 A3 镜像标签不代表已在本教程的 910B 上验证。仓库不发布内部镜像地址、镜像归档或容器标识。

若需要自行创建容器，使用自己可访问且已验证兼容的镜像，通过 Docker 参数映射指定的八张卡、驱动及项目目录。挂载点与系统组件路径由部署环境配置，不在项目代码中写入个人或机器专用绝对路径。不要将登录密钥、HF token 或云存储凭据写入镜像层或提交到 Git。

容器内先检查：

```bash
python tools/collect_env.py
ffmpeg -version
ffprobe -version
```

环境采集脚本的输出与退出码见 [工具说明](../tools/README.md#cuda-和-npu-环境采集)。

核对八卡当前计算 PID 与任务归属；单看利用率 0 或存在设备节点不能证明卡可用。此检查不是模型正确性验收。

## 3. 安装 XGEN

```bash
git clone --branch v0.2 https://github.com/lightnateriver/xGen.git
cd xGen
python -m pip install -e ./x_base --no-deps
python -m pip install -e ./x_diffusers --no-deps
python -m pip check
```

上述 `--no-deps` 用于已有完整配套环境，避免重装 Torch/CANN 配套包；它不会自动补齐依赖。缺失包应对照 `x_base/requirements.txt` 和 `x_diffusers/requirements.txt` 在隔离环境中补齐，再执行 `pip check`。不要对正在使用的环境执行无约束的全量升级。TorchNPU/内核扩展必须与目标架构和 CANN 相匹配。

原 HTTP 服务需要 `flask` 和 `esdk-obs-python` 等已有服务依赖；`flask-cors` 为可选。部署本地文件服务时不要配置云存储变量；若已有云存储方案，通过外部环境配置，凭据不得出现在配置示例、命令参数或日志中。

CPU 自检：

```bash
python x_diffusers/tests/test_h3_refiner_opt_in.py
python x_diffusers/tests/test_h3_refiner_runtime_cpu.py
```

基线兼容测试需要本地存在 `v0.1` 或 `origin/v0.1` 引用；单分支浅克隆可先 `git fetch origin v0.1:refs/remotes/origin/v0.1`。没有 Torch/FFmpeg 时，对应测试会跳过，跳过不等于通过。

## 4. 模型与目录准备

```text
xGen/
├── weights/
│   ├── base/                    原 H3 模型根，保留 modular_model_index.json
│   └── refiner/                 Regenerate 源包
│       ├── base_model/
│       ├── latent_upsampler/
│       └── lora/
├── model-cache/
│   ├── base-t2va/               剥离后的 Base transformer
│   ├── base-ref2va/
│   ├── t2va-51step.safetensors
│   ├── ref2va-51step.safetensors
│   └── refiner/                 预处理后的 Refiner transformer、表和 manifest
├── inputs/                      使用者自备首帧或参考媒体
├── outputs/                     使用者生成的视频和指标
└── logs/
```

先阅读并满足 [MiniMax-H3 模型许可](https://huggingface.co/MiniMaxAI/MiniMax-H3) 和 [Open-H3-Regenerate-2K 模型许可](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)。需要授权时由使用者在交互终端登录 HF；不要把 token 放进本教程命令或仓库。

### 4.1 Base

[Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [ModelScope](https://modelscope.cn/models/MiniMax/MiniMax-H3)

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 6818f6c32d12b210915e44ad56a4228c2608f160 \
  --local-dir weights/base \
  --include '*.json' 'LICENSE' 'scheduler/*' 'audio_scheduler/*' \
    'processor/*' 'tokenizer/*' 'text_encoder/*' 'transformer/*' \
    'transformer_ref/*' 'vae/*' 'audio_vae/*'
```

XGEN 使用包含根级组件与 modular index 的布局，不要仅下载 `FL2VA/` 子目录后直接套用根级路径。T2VA/首帧 I2VA 使用对应 Base 工作流；R2VA 使用 Ref2VA transformer 和独立匹配的 AdaLN 资产。

### 4.2 Refiner

[Hugging Face](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) · [ModelScope](https://modelscope.cn/models/Efficient-Large-Model/Open-H3-Regenerate-2K)

源包中需要组合 LoRA、Sharp curve 分支和 `adaln_t_table.bin`。仅执行普通 HF 全仓下载可能缺少最后一项。可使用上游固定版本的准备脚本；这里只下载并执行模型准备工具，不导入它的 CUDA 推理引擎：

```bash
mkdir -p model-cache
curl --fail --location \
  'https://raw.githubusercontent.com/NVlabs/Sana/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K/download.py' \
  --output model-cache/download-refiner.py
python model-cache/download-refiner.py --weights weights/refiner
```

上游脚本校验发布文件与 Sharp 表哈希，并处理本地组件定位；其中可能下载本分支不使用的 fast VAE 文件。XGEN 当前用的是原 H3 VAE 的 tile 优化，**没有启用 Light VAE、StreamRender 蒸馏 VAE 或 NVIDIA MXFP8/Sol 引擎**。上游生成到本地模型 index 的环境定位信息属于模型资产，不提交到源码仓库。

### 4.3 一次性预处理

这些步骤需要 NPU；在正式推理前、空闲设备上单独执行。它们是模型级资产准备，不是逐请求条件缓存。

```bash
python -m x_diffusers.framework.pipeline.h3_regenerate.prepare_base_cache \
  --source weights/base/transformer --model-variant T2VA --steps 50 \
  --output model-cache/t2va-51step.safetensors \
  --stripped-output model-cache/base-t2va

python -m x_diffusers.framework.pipeline.h3_regenerate.prepare_base_cache \
  --source weights/base/transformer_ref --model-variant Ref2VA --steps 50 \
  --output model-cache/ref2va-51step.safetensors \
  --stripped-output model-cache/base-ref2va

python -m x_diffusers.framework.pipeline.h3_regenerate.prepare \
  --weights weights/refiner --output model-cache/refiner
```

组合 LoRA 已包含发布的 Refiner/Turbo/Sharp 线性增量，准备工具另核对 Sharp AdaLN 曲线分支；不要在启动时再次重复合并。输入模型目录需与所选模式对应。新目录默认不覆盖；源模型、调度或 LoRA 变化时使用新资产目录，不混用旧表。

H3 的 `num_inference_steps` 和 AdaLN 预计算的 `--steps` 都填实际 Base DiT 次数。50 步填 50，20 步填 20。普通旧 H3 的 `51` 调度点改填 `50` 步，已有同一采样序列的 AdaLN 文件可以直接复用；Turbo-SLA 等特殊路径按 [升级说明](H3-UPGRADE.md) 处理。切换步数时，在线服务需换用匹配的 AdaLN 表并重启。Refiner 固定 3 步，Refiner INT8 和融合 RoPE 均未开启。

## 5. 原链路：不启用超分

不传 `--h3_refiner`、且配置未启用它时，继续使用原脚本：

```bash
# 先按旧示例设置自己的模型、输入和输出位置；Refiner 不会被加载。
bash example/diffusers/scripts/infer_minimax_h3_ref2va.sh
```

该命令展示原入口，不表示示例输入已经随仓库提供。原 768P 路径和历史优化配置见对应原脚本；不要用 experimental 2K 后端的参数覆盖旧服务配置。

### v0.3.1 NPU 首帧图生音视频

在已配置 CANN、TorchNPU、FFmpeg 和 Python 包的环境中，使用现有 FL2VA 脚本。下面的相对路径以脚本目录为基准。模型目录需要根级 modular index，Transformer 和 AdaLN 表须来自对应的 Base，对应 50 次 Base DiT。

```bash
cd example/diffusers/scripts
XGEN_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_NPUS=8 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29507 PHAA_NUM=2 \
  MODEL_PATH=../../../weights/h3 \
  TRANSFORMER_PATH=../../../model-cache/transformer-base \
  ADALN_CACHE_FILE=../../../model-cache/adaln-51step.safetensors \
  FIRST_FRAME_ONLY=1 FIRST_FRAME=../../../inputs/first-frame.png \
  PROMPT_FILE=../../../inputs/prompt.txt OUTPUT_PATH=../../../outputs/h3-npu.mp4 \
  bash infer_minimax_h3_fl2va.sh
```

`FIRST_FRAME_ONLY=1` 只传首帧，`PROMPT_FILE` 读取 UTF-8 提示词。省略这两个变量时沿用原示例的首尾帧和示例提示词。默认使用 Laser、DiT/VAE INT8 线性、融合 RoPE 和优化 VAE decoder；八卡启动启用 SP8 与文本 TP8。确保 Transformer 和 AdaLN 路径都存在，启动日志应显示加载 50 步的预计算表。选择空闲的本地端口，避免使用依赖容器主机名解析的 rendezvous。

[768P、50 次 DiT 实测](performance/h3-cuda.md#v031-双设备冒烟)

## 6. v0.2 离线 2K 推理

以下均在仓库根目录运行，沿用原 `example/diffusers/infer.py`。

```bash
export BASE_MODEL=weights/base
export BASE_TRANSFORMER=model-cache/base-t2va
export BASE_ADALN=model-cache/t2va-51step.safetensors
export REFINER_WEIGHTS=weights/refiner
export REFINER_PREPARED=model-cache/refiner
export OUTPUT_VIDEO=outputs/t2va-5s-2k.mp4
export SECONDS_TO_GENERATE=5
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export INF_NAN_MODE_FORCE_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p outputs logs
bash example/diffusers/scripts/infer_minimax_h3_refiner.sh
```

示例会明确设置 Base INT8、Laser、RoPE、AdaLN、SP8/文本 TP8，以及 Refiner BF16/3 步。`vae_matmul_quant=none`、`vae_lightning=none`、`inf_vram_blocks_num=0` 避免与新后端分阶段驻留叠加 hook。输出位置由环境变量设置，脚本不会写死机器路径。

### 单图 I2VA

```bash
export OUTPUT_VIDEO=outputs/i2va-5s-2k.mp4
bash example/diffusers/scripts/infer_minimax_h3_refiner.sh \
  --task_type fl2va --keyframe_image inputs/first-frame.png
```

### R2VA 多媒体参考

```bash
export BASE_TRANSFORMER=model-cache/base-ref2va
export BASE_ADALN=model-cache/ref2va-51step.safetensors
export OUTPUT_VIDEO=outputs/r2va-5s-2k.mp4
bash example/diffusers/scripts/infer_minimax_h3_refiner.sh \
  --task_type ref2va --reference_paths \
    inputs/ref01.png inputs/ref02.png inputs/ref03.png inputs/ref04.png inputs/ref05.png
```

R2VA 每次最多 9 张图、3 段视频、3 段独立音频，合计最多 12 个文件。视频和音频每段 2–15 秒，两类各自累计最多 15 秒。音频需要配合图或视频。视频内嵌音轨随视频进入 Base，仍按一个视频文件计数。单图像素总量不超过 4096×4096。2K 路径由 Base 消费参考文件，Refiner 沿用 Base latent。媒体时长校验使用 FFprobe。

历史性能使用 1280×720 首帧和五张 2560×1440 参考原图。源图分辨率与最终进入 DiT 的 token 数分别记录。

设置 `SECONDS_TO_GENERATE=10` 或 `15` 可选择时长；这是已实现参数范围，不表示新分支已完成对应 NPU 全矩阵验收。输出固定 2560×1440/24fps。生成帧数 124/243/362，导出裁剪到 120/240/360 帧。

## 7. 连续离线请求：同一 Manager 只初始化一次

同一工作流变化时长不需要每次读盘、量化或构造模型。按下面方式复用原 Manager；每个 rank 执行相同请求顺序，没有新增批调度器：

```python
from copy import deepcopy
from x_diffusers import init_env, parse_args, InferenceManager

args = parse_args()  # 使用上节同一组启动参数
init_env()
manager = InferenceManager()
try:
    for seconds in (5, 10, 15):
        request = deepcopy(args)
        request.h3_refiner_seconds = seconds
        request.save_path = f"outputs/t2va-{seconds}s-2k.mp4"
        manager.infer(request)
finally:
    if manager.sr_pipe is not None:
        manager.sr_pipe.close()
```

将代码保存为自己的启动文件，再用同一组八卡 torchrun 参数运行。不同工作流、模型、并行/量化配置和预计算步数分别启动进程。首版不支持在一个已运行实例中热切换新旧后端。

## 8. 原 HTTP 服务的可选 2K 后端

768P 和 2K 的启动选择、提交和下载示例见 [在线服务指南](H3-ONLINE.md)。

复制示例配置到自己维护的本地配置目录，按需修改相对路径：

```bash
mkdir -p logs
export PYTHONPATH="$(pwd)/x_diffusers:$(pwd)/x_base${PYTHONPATH:+:$PYTHONPATH}"
torchrun --nproc_per_node=8 --max_restarts=0 --master_port=29558 \
  server/video/video_server.py --init \
  --config example/diffusers/configs/h3_refiner.example.json \
  --host localhost --port 5001 > logs/h3-refiner.log 2>&1
```

服务沿用原 `/generate`、任务查询与下载 API。启动参数就绪检查和首次模型加载是不同阶段；`/params` 只能证明当前接口规格，完整请求成功才证明数值链路运行。开启后每个模式单独启动：T2VA 配置 `task_type=t2va`；I2VA 配置 `fl2va`；R2VA 配置 `ref2va` 并设置匹配的 Ref2VA transformer/AdaLN 路径。

```bash
curl --fail --silent --show-error http://localhost:5001/params
curl --fail --silent --show-error http://localhost:5001/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A small robot walks through a sunlit garden.","duration":5,"resolution":"2k","seed":42,"enable_audio":true}'
```

以上返回任务 ID，不是直接返回 MP4；按原任务状态和结果 API 查询。I2VA 追加 `first_frame`，R2VA 追加五元素 `ref_path` 数组，路径由原服务解析。开关、模型路径和加载方式为启动级设置，请求不能覆盖。

### 断开终端后继续运行

普通前台命令会受终端影响。需要断线继续时，在自己的环境用会话管理器或 `nohup` 启动原命令，例如：

```bash
nohup torchrun --nproc_per_node=8 --max_restarts=0 --master_port=29558 \
  server/video/video_server.py --init --model minimax-h3 \
  --config example/diffusers/configs/h3_refiner.example.json \
  --host localhost --port 5001 > logs/h3-refiner.log 2>&1 <&- &
```

记录本次实际启动 PID 与命令，并用真实进程和日志核对。只控制自己启动的任务；不要依据历史 PID 或端口猜测并停止进程。`max_restarts=0` 不会自动重试失败请求。

## 9. 显存、内存、日志与回退

Base、Refiner、VAE、upscaler 的 CPU master 常驻主机内存；HBM 按阶段切换。`prefetch` 为两个层槽，`sync` 用同一计算路径同步加载；不是把整套 Refiner 常驻显存，也不是请求条件缓存。

| 指标 | 含义与注意事项 |
|---|---|
| 初始化 | 一组进程只做一次；不摊进每条常驻请求 E2E |
| Base/Refiner DiT | 各阶段真实调用次数和事件跨度，不能用配置点数代替实际调用数 |
| H2D | 初始加载、分层传输和等待分开记录；预取等待不是全部传输成本 |
| VAE / RGB8 / D2H | 解码、像素归一化、主机传输及队列等待 |
| 编码与音频封装 | 必须完成最终文件后才算技术成功 |
| allocated / reserved / 驱动 HBM | 活跃 tensor、PyTorch 分配池、驱动占用分别观察 |
| RSS / PSS / 页缓存 | 不同主机内存口径；PSS 合计不等于主机总 RAM 使用量 |

每条视频旁的 `.metrics.rankN.json`、`H3_REFINER_READY / COMPLETE / FAILED` 日志可用于定位。事件有嵌套和重叠，不能直接相加得到 E2E。对比请求结束且编码排空后的稳定边界，区分首次形状触达、固定缓冲和持续增长。

历史优化版的固定最大 RGB8 缓冲约 3.71 GiB，采样 HBM 峰值 61.09 GiB、PSS 合计峰值约 714.73 GiB；这些不是当前重构版的固定配置或内存保证。历史 HBM/RSS 每 10 秒、PSS 每 30 秒采样，仍可能漏掉瞬时峰值。

关闭方式：从启动命令移除 `--h3_refiner`，配置改为 `"h3_refiner": false`，重启原服务。不要仅删权重路径而保持开关开启。首次初始化失败或 GPU/collective 失败会要求重启，不继续复用受损实例；没有自动重试、latent 恢复或静默降级成 768P。

## 10. 性能数据：先看版本与测量边界

### 10.1 历史优化版，72 条技术验收

以下数据来自 2026-11-01 的归档测试。只发布匿名数值：设备 A=8×910B1/64GiB，设备 B=8×910B3/64GiB。每台 36 条，每项一个样本，并发 1；不是多次分布或吞吐量。72 条均通过哈希、媒体参数和完整音视频解码检查，人工画面/听音仍待复核。

历史配置：Base BF16 执行＋INT8 matmul、Laser、融合 RoPE、AdaLN、SP8、文本 TP8、PHAA2/workspace1；2K Refiner BF16/3 步、Laser、SP8、PHAA2/workspace0、分层预取；VAE tiles_stream、tile batch1、norm FP32；异步 RGB8/pinned D2H、流水编码、CPU YUV、libx264 veryfast/CRF8/16 threads＋AAC。历史 Base 配置保留 `vae_matmul_quant=int8`，不能说整个 VAE 或 Refiner 都是 INT8。

常驻请求 E2E 包含计算和该请求导出完成等待，不包含组初始化；在线客户端和离线直接循环的计时入口不同。跨设备执行顺序、首次形状触达和指标埋点有差异，B/A 不是严格硬件单因素加速比。

| 模式 | 时长 | 输出 | 路径 | 设备 A：910B1 | 设备 B：910B3 | B/A | 说明 |
|---|---:|---|---|---:|---:|---:|---|
| T2VA | 5s | 768P | 在线 | 110.66 | 125.75 | 1.136× | 常驻 |
| T2VA | 5s | 768P | 离线 | 248.32 | 124.25 | — | 冷启动，不能与常驻比 |
| T2VA | 5s | 2K | 在线 | 181.55 | 192.49 | 1.060× | 常驻 |
| T2VA | 5s | 2K | 离线 | 175.09 | 197.27 | 1.127× | A 为独立补测 |
| T2VA | 10s | 768P | 在线 | 297.73 | 331.32 | 1.113× | 常驻 |
| T2VA | 10s | 768P | 离线 | 297.39 | 338.30 | 1.138× | A 为独立补测 |
| T2VA | 10s | 2K | 在线 | 483.43 | 573.19 | 1.186× | 常驻 |
| T2VA | 10s | 2K | 离线 | 484.88 | 568.97 | 1.173× | A 为独立补测 |
| T2VA | 15s | 768P | 在线 | 574.34 | 673.40 | 1.172× | 常驻 |
| T2VA | 15s | 768P | 离线 | 576.75 | 664.85 | 1.153× | A 为独立补测 |
| T2VA | 15s | 2K | 在线 | 969.52 | 1156.55 | 1.193× | 常驻 |
| T2VA | 15s | 2K | 离线 | — | 1151.69 | — | A 的 E2E 缺失 |
| I2VA | 5s | 768P | 在线 | 130.59 | 137.32 | 1.052× | 常驻 |
| I2VA | 5s | 768P | 离线 | 126.28 | 133.73 | 1.059× | 常驻 |
| I2VA | 5s | 2K | 在线 | 179.64 | 203.06 | 1.130× | 常驻 |
| I2VA | 5s | 2K | 离线 | 184.00 | 207.74 | 1.129× | 常驻 |
| I2VA | 10s | 768P | 在线 | 318.74 | 351.44 | 1.103× | 常驻 |
| I2VA | 10s | 768P | 离线 | 303.25 | 356.70 | 1.176× | 常驻 |
| I2VA | 10s | 2K | 在线 | 502.48 | 589.34 | 1.173× | 常驻 |
| I2VA | 10s | 2K | 离线 | 503.57 | 593.25 | 1.178× | 常驻 |
| I2VA | 15s | 768P | 在线 | 598.86 | 696.48 | 1.163× | 常驻 |
| I2VA | 15s | 768P | 离线 | 597.41 | 705.86 | 1.182× | 常驻 |
| I2VA | 15s | 2K | 在线 | 989.27 | 1174.78 | 1.188× | 常驻 |
| I2VA | 15s | 2K | 离线 | 994.97 | 1175.62 | 1.182× | 常驻 |
| R2VA | 5s | 768P | 在线 | 659.01 | 774.92 | 1.176× | 常驻 |
| R2VA | 5s | 768P | 离线 | 658.45 | 771.23 | 1.171× | 常驻 |
| R2VA | 5s | 2K | 在线 | 716.95 | 844.59 | 1.178× | 常驻 |
| R2VA | 5s | 2K | 离线 | 719.03 | 837.17 | 1.164× | 常驻 |
| R2VA | 10s | 768P | 在线 | 1030.40 | 1205.10 | 1.170× | 常驻 |
| R2VA | 10s | 768P | 离线 | 1037.66 | 1206.30 | 1.163× | 常驻 |
| R2VA | 10s | 2K | 在线 | 1220.84 | 1441.97 | 1.181× | 常驻 |
| R2VA | 10s | 2K | 离线 | 1224.21 | 1444.09 | 1.180× | 常驻 |
| R2VA | 15s | 768P | 在线 | 1509.57 | 1772.10 | 1.174× | 常驻 |
| R2VA | 15s | 768P | 离线 | 1502.16 | 1777.49 | 1.183× | 常驻 |
| R2VA | 15s | 2K | 在线 | 1898.22 | 2267.94 | 1.195× | 常驻 |
| R2VA | 15s | 2K | 离线 | 1904.32 | 2256.44 | 1.185× | 常驻 |

单位为秒。A 的离线 T2VA 5 秒 768P 含冷启动；A 的离线 T2VA 15 秒 2K 完整 E2E 缺失，保持空值；A 的四条独立常驻补测不算成六条连续常驻。只有 34 组可计算同口径 B/A。

完整匿名数据：[36 组请求 E2E](performance/historical-pairs.csv)、[72 条阶段与显存数据](performance/historical-stages.csv)、[验证范围](performance/validation-summary.json)。阶段 CSV 中 DiT/VAE 使用 device envelope，D2H 使用原记录的 host 口径；不能混加。历史实验运行器、内部 plan、原图、prompt、视频和机器定位信息不随源码发布，因此这部分是性能参考，而不是在 v0.2 中可原样运行的旧批测脚本。

### 10.2 本次 v0.2 重构回归，两个连续请求

同一八卡进程、T2VA、5 秒、2560×1440、Base50/Refiner3，初始化约 64.77 秒且只做一次：

| 请求 | rank0 请求周期（秒） | 最大单卡 peak allocated（GiB） | rank0 结束 allocated（GiB） | rank0 结束 reserved（GiB） |
|---|---:|---:|---:|---:|
| 1 | 191.82 | 28.39 | 20.93 | 30.27 |
| 2 | 179.43 | 28.90 | 20.93 | 30.83 |

请求周期含阶段资源切换、解码/导出与 Base 回装，是 worker 埋点口径，不是 HTTP 客户端 E2E；不与上表直接计算性能提升。两条最终文件均为 120 帧、24fps、H.264/AAC，完整解码通过；16 份 rank 指标实际调用数正确，模型/CPU master 身份稳定，所有 worker 退出后八卡无计算进程。

八卡结束 allocated 的第二次增量均为 0；reserved 增加 576 MiB，短测不能判断长期泄漏。新回归没有逐请求边界 PSS 样本，不能宣称主机内存长期稳定。数据见 [v0.2 回归 CSV](performance/v02-smoke.csv)。

当前 v0.2 优先复用原 SaveAVStream、同步 D2H 和原编码默认参数；没有恢复旧独立导出器的全部异步 D2H/跨请求编码重叠。不要将旧性能表作为本版无性能回退的保证。

## 11. 技术检查与常见问题

```bash
ffprobe -v error -show_streams -show_format -of json outputs/t2va-5s-2k.mp4
ffmpeg -v error -i outputs/t2va-5s-2k.mp4 -f null -
```

- **升级后旧的 `51step` AdaLN 文件还能用吗？** 可以。旧文件名里的 51 是包含终点的 sigma 点数。推理配置填 50，文件路径照旧，实际步数由缓存元数据校验。
- **为何不能再加 SeedVR2/旧 offload？** 两套资源 hook 可能同时接管同一模型，启用新后端时会拒绝互斥设置；关闭后走旧逻辑。
- **为什么卷积输入 BF16、bias FP32 报错？** Refiner VAE 加载后需要整体物化 BF16，再将 norm 权重保留 FP32；本分支已补齐该策略和测试。
- **只有模型 HTTP 状态成功就正常吗？** 不够。需要最终视频技术检查，再人工检查人物一致性、画面、音频和音画同步。
- **能直接对已有任意 768P MP4 超分吗？** 当前公开可选链路从 H3 原始 latent 接入；旧独立视频超分实验入口未作为本版正式 API 发布，不应虚构相同命令。
- **能否把 10/15 秒、五图都视为 v0.2 已验收？** 不能。代码有参数支持和 CPU 边界测试，但新架构完整 NPU 矩阵、在线实际请求及长稳验收仍需补齐。

## 12. 模型来源与发布范围
- [MiniMax-H3 官方模型](https://huggingface.co/MiniMaxAI/MiniMax-H3)：基础模型和许可。
- [Open-H3-Regenerate-2K 模型](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)：Refiner 源资产和许可。
- [上游固定版本下载工具](https://github.com/NVlabs/Sana/blob/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K/download.py)：资产准备与校验。
- [Ascend PyTorch 版本配套](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)：部署前核对软件组合。

公开内容包含教程、源代码、测试、匿名性能表与第三方许可证。模型权重、访问凭据、机器地址、个人目录、临时签名素材链接、私人 prompt/媒体和内部实验历史不发布。法律要求的第三方署名与公共项目名称保留，不代表内部人员信息。

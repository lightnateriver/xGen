# X_GEN

## v0.2：可选 H3 Refiner

H3 → latent 超分 → Open-H3-Regenerate-2K 作为 **experimental、默认关闭** 的能力接入原推理 Manager 与视频服务。不加 `--h3_refiner` 且配置不启用它时，沿用 v0.1 的脚本、API、推理、设备调度和缓存路径。

从环境、权重准备到在线/离线推理及匿名性能表，见 [完整部署教程](docs/H3-DEPLOYMENT.md)。参数与限制另见 [H3 Refiner](x_diffusers/x_diffusers/framework/pipeline/h3_regenerate/README.md)。旧优化版性能与当前重构回归分开列示。

## v0.1 版本范围

`v0.1` 发布分支保存接入 H3 Refiner 之前的 XGEN 基线。本分支在其上增加默认关闭的可选能力；以下原有模型适配、算子与并行优化继续保留。

发布内容包含源码、测试和使用示例；机器交接及历史实验记录、模型权重、业务素材不随源码发布。示例中的 `./...` 路径需要按实际环境配置，参考图等输入由使用者提供。

X_GEN 是基于昇腾 NPU 的高性能视频生成推理引擎，支持 Wan、HunyuanVideo、CogVideoX 等主流视频生成模型。

## 支持模型

| 模型 | 任务类型 | 说明 |
|------|----------|------|
| Wan2.1-T2V-1.3B | 文生视频 | 1.3B 轻量版 |
| Wan2.1-T2V-14B | 文生视频 | 14B 标准版 |
| Wan2.1-I2V-14B | 图生视频 | 图像引导生成 |
| Wan2.1-VACE-1.3B | 多任务 | 视频编辑多任务 |
| Wan2.1-VACE-14B | 多任务 | 视频编辑多任务 |
| Wan2.2-T2V-A14B | 文生视频 | Wan2.2 加速版 |
| Wan2.2-I2V-A14B | 图生视频 | Wan2.2 加速版 |
| HunyuanVideo-T2V-13B | 文生视频 | 腾讯混元 |
| CogVideoX-5B | 文生视频 | 智谱 CogVideoX |

## 项目结构

```
x_gen/
├── x_base/              # 基础库 - 加速特性实现
├── x_diffusers/         # Diffusers 适配层 - 推理框架
├── example/             # 使用示例与脚本
├── comfyui/             # ComfyUI 集成
├── tools/               # 模型转换、超分等工具
├── x_vllm_omni/         # vLLM 集成（开发中）
└── build.sh             # 构建脚本
```

## 环境安装

### 1. Docker 环境

使用昇腾云 PyTorch 2.5.1 版本 Docker 镜像创建容器。

### 2. 安装依赖

```shell
# 安装昇腾加速包
(cd x_base && pip install -e .)
# 安装 x_diffusers（默认 torch 2.10.0）
(cd x_diffusers && pip install -e .)

# 如需指定 torch 2.9.0 版本
# cd x_diffusers && X_DIFFUSERS_TORCH_VERSION=29 pip install -e .
```

### 3. 构建 Whl 包（可选）

```shell
bash build.sh
```

## 权重下载

| 模型 | 下载地址 |
|------|----------|
| Wan2.1-T2V-14B | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers |
| Wan2.1-I2V-14B | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers |
| Wan2.1-T2V-1.3B | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| Wan2.1-VACE-1.3B | https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers |
| Wan2.1-VACE-14B | https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers |
| Wan2.2-T2V-A14B | https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers |
| Wan2.2-I2V-A14B | https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers |
| HunyuanVideo | https://huggingface.co/tencent/HunyuanVideo |
| IFRNet_S_Vimeo90K | https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/tag/models |

### 使用 huggingface-cli

```shell
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --local-dir ./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

## 快速开始

```shell
cd example/diffusers/scripts
bash infer_wan2.2_14b_t2v_720p.sh
```

## 详细文档

| 模块 | 说明 | 文档 |
|------|------|------|
| x_base | 加速特性（SP、FSDP、PHAA、量化等） | [x_base/README.md](x_base/README.md) |
| x_diffusers | API 使用、Pipeline 配置 | [x_diffusers/README.md](x_diffusers/README.md) |
| example | 示例脚本说明 | [example/README.md](example/README.md) |
| comfyui | ComfyUI 集成 | [comfyui/README.md](comfyui/README.md) |
| tools | 模型转换、超分工具 | [tools/README.md](tools/README.md) |

## 基线已知问题

`x_base/x_base/infinite_vram/scripts/infer_wan_2.1_14b_t2v_infinite_vram.sh` 在基线中已有引号/编码问题，无法通过 `bash -n`。本分支保留该历史文件作为版本对照；常规启动请使用 `example/diffusers/scripts/` 中对应模型的示例。

部署示例使用相对路径或外部配置。旧版本的绝对占位路径已移植；生产部署请明确配置模型与输出位置。云存储凭据目录由外部 `OBS_CREDENTIAL_DIR` 配置，不能提交凭据。
## v0.3 可选 H3 attention 后端

新增默认关闭的 experimental Spark-H3 / Veda Base attention，通过 `--h3_attention` 选择。
默认 `legacy` 保留原链路；Refiner、VAE、编码及原有调度继续复用。
参见[接入、参数与回退说明](docs/H3-ATTENTION.md)和[50 步实测结果](docs/H3-ATTENTION-VALIDATION.md)。
算子移植对齐、视频抽检和性能收益分别记录；新后端不自动成为默认配置。

5 秒 T2VA 的 LightX2V 8 步离线实验使用独立开关 `--h3_turbo8`。权重准备与 768P/2K 示例见[参数说明](docs/H3-ATTENTION.md#experimental-lightx2v-turbo-8-step-offline)。

10 秒 I2VA 可通过离线 CLI 选择 Veda 0.9 / 0.7。参见[启动命令](docs/H3-ATTENTION.md#veda-i2va-10-second-profile)与[三后端热态对比](docs/H3-ATTENTION-VALIDATION.md#10-second-i2va--20-actual-base-steps)。

## 实验性 CUDA H3 Base

[CUDA 部署与启动教程](docs/H3-CUDA.md) · [CUDA 实测性能](docs/performance/h3-cuda.md)

显式设置 `XGEN_DEVICE=cuda` 选择 CUDA，默认仍使用 NPU。CUDA 上可用
`--atten sage` 或 `--atten sparge` 选择对应算子，需安装对应 CUDA 扩展。

在 RTX 5090 上使用 FastVideo 的 FL2VA pruned NVFP4 权重时，在原 FL2VA
启动参数后加 `--h3_nvfp4_checkpoint ./weights/minimax_h3_fl2va_pruned_nvfp4.safetensors`。
保留 `--pretrained_model_name_or_path` 以加载模型配置、文本编码器和 VAE。
此路径需要支持 NVFP4 的 Comfy-Kitchen CUDA 算子和 BF16 激活。权重以打包格式保存，
每次投影执行激活量化。当前支持 FL2VA Base 和下文的实验性 2K Refiner。暂不组合额外 LoRA、
AdaLN 缓存、FSDP 或其他权重量化。移除该参数即可使用常规 checkpoint。
显存充足时可额外传入 `--h3_nvfp4_resident`，让打包 DiT 常驻每张卡，省去逐层
搬运。文本编码器仍按原策略 offload。移除该参数即可恢复逐层搬运。
常驻模式下可试验 `--h3_nvfp4_compile`，只编译重复的 DiT block。首个请求
包含编译开销，换形状可能再次编译。Attention 通信和 NVFP4 自定义算子继续 eager 执行。
移除参数即可让整个 DiT 恢复 eager。
编译环境需要可用的 C 编译器和与运行时匹配的 Python 开发头文件。
当前验证中局部编译改变了音视频输出，且同配置的 8 步请求比 eager 慢。
多卡可用 `--h3_cuda_vae_sp` 试验 CUDA VAE 序列并行，复用 xGen 的 VAE
分片逻辑和原有音视频保存流程。VAE attention 头数需能整除 SP 卡数。

设置 `MODEL_PATH`、`NVFP4_CHECKPOINT`、`FIRST_FRAME` 和 `PROMPT_FILE` 后，
运行 [CUDA NVFP4 推理示例](example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh)。
模型目录需有 Diffusers 的 `modular_model_index.json` 及对应组件，支持通过
`CUDA_VISIBLE_DEVICES` 选择卡、`GPUS` 设置 SP 卡数、`ATTENTION_BACKEND` 选择 attention。
启动前将 `ffmpeg` 加入 `PATH`，用于封装视频和音轨。
单卡运行 Sparge 时可设置 `XGEN_SPARGE_HEAD_CHUNK=7`，按组执行 attention 以降低
内核临时显存。默认值 `0` 一次处理所有本地头。分块增加内核启动次数，
实际时延取决于序列长度和每卡头数。

实验性多卡通信可用 `XGEN_PHAA_FP8=1` 配合 CUDA、SP 大于 1 和
`--phaa_num` 开启。它复用 PHAA 的分头和流调度，以 E4M3 和逐头 FP32 scale
传输 Q、K、V 和 attention 输出。128 维时每头每 token 的载荷从 256 降至
132 字节，每个头分块的 Q/K/V 合并为一次 all-to-all。量化会改变结果，
接收端将数据还原到原张量 dtype。对 BF16 attention 输出，这一路径就是 BF16 → FP8
回程传输 → BF16。该开关同时控制去程和回程。

RTX 5090 上的 SageAttention CUDA 分派使用 INT8 Q/K 和 FP8 P×V。计算精度由
`ATTENTION_BACKEND=sage` 选择，通信精度由 `XGEN_PHAA_FP8` 选择。
使用上面的 FL2VA 示例时，设置 `GPUS=8 ATTENTION_BACKEND=sage XGEN_PHAA_FP8=1`，
并附加 `--phaa_num 2 --h3_nvfp4_resident`，即可使用已验证的 50 次 DiT 组合。
该配置在 8 张 RTX 5090 上生成并保存约 5 秒的 768P 音视频耗时 102.77 秒，
已通过完整音视频解码。端到端的画质和速度需用目标输入复核。
配合 `XGEN_PHAA_GATHER_PRIORITY=1` 可优先提交全部 Q/K/V 通信。
默认仍交错提交 Q/K/V 与 attention 输出通信。
实验性 `XGEN_PHAA_DUAL_COMM=1` 将输出通信放到独立的 CUDA 流和 NCCL 通信器。
它需要开启 FP8、SP 覆盖全部进程、NCCL 2.26 或更新版本，并在启动前设置
`NCCL_LAUNCH_ORDER_IMPLICIT=1`。

八张 RTX 5090 上已验证的 768P、50 次 DiT 配置如下。使用前先设置上面的
模型、权重和输入变量。该配置每卡保留完整的打包 DiT 权重。

```bash
GPUS=8 ATTENTION_BACKEND=sparge XGEN_SPARGE_TOPK=0.5 XGEN_PHAA_FP8=1 \
  bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh \
  --h3_nvfp4_resident --h3_cuda_vae_sp --phaa_num 2
```

同一输入与 seed 的单轮测试中，Sparge 逐层搬运版生成及保存耗时 250.25 秒，
常驻版为 99.94 秒，加 VAE SP 后为 96.07 秒。常驻版与逐层搬运版的解码后
音视频哈希一致。加 VAE SP 后音频哈希一致，画面 SSIM 为 0.9948。
这些耗时从 pipeline 初始化完成后开始计量。

CUDA 在线推理复用原有视频服务。以双卡 FL2VA 为例，在服务 JSON 中设置
`model=MiniMax-H3`、`sp=2`、`text_encoder_tp=1`、`phaa_num=2`、
`atten=sage`、`vae_lightning=none`、`matmul_quant=none`、
`vae_matmul_quant=none` 和 `rope_fused=false`。设置模型目录和
`h3_nvfp4_checkpoint`，将 `transformer_path` 和 `adaln_cache_file` 设为
`null`，以覆盖服务的 NPU 模型默认值。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=2 \
  server/video/video_server.py --init --config ./h3-cuda.json --host 127.0.0.1
```

通过原有 `/generate` 提交 `task_type=fl2va`、`first_frame` 和 `prompt`，
使用 `/status/<task_id>` 查询结果。测试覆盖同一服务串行处理两条 768P 请求。

### 实验性 CUDA 2K Refiner

使用现有 H3 Refiner adapter 和视频服务。当前 CUDA 路径要求 FL2VA、SP8、
BF16 配置和 NVFP4 Base 权重，由 Refiner 统一管理阶段驻留。
先安装可执行的 FFmpeg（含 libx264 和 AAC），可通过 `FFMPEG_PATH` 指定。
使用部署教程的 Refiner 源包和预处理命令，在 `XGEN_DEVICE=cuda` 环境下
运行 `h3_regenerate.prepare`。预处理输出单独存放，供后续推理复用。

在线服务在现有 JSON 中加入以下配置。路径示例以启动目录为基准。

```json
{
  "model": "MiniMax-H3",
  "task_type": "fl2va",
  "resolution": "2k",
  "pretrained_model_name_or_path": "weights/h3",
  "h3_nvfp4_checkpoint": "weights/h3-fl2va-nvfp4.safetensors",
  "h3_refiner": true,
  "h3_refiner_weights": "weights/refiner",
  "h3_refiner_prepared": "model-cache/refiner-cuda",
  "sp": 8,
  "text_encoder_tp": 1,
  "phaa_num": 2,
  "atten": "none",
  "num_inference_steps": 51,
  "matmul_quant": "none",
  "vae_matmul_quant": "none",
  "vae_lightning": "none",
  "rope_fused": false
}
```

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 server/video/video_server.py \
  --init --config config-cuda-2k.json --host 127.0.0.1 --port 29655
```

通过 `/generate` 提交 `task_type=fl2va`、`resolution=2k`、`duration=5`、
`first_frame`、`prompt` 和 `seed`，用 `/status/<task_id>` 查询结果。
离线使用同一套模型参数，给现有推理入口传入 `--h3_refiner`、
`--width 2560 --height 1440 --frames 124 --save_fps 24` 和首帧。

已验证 8 张 RTX 5090 上的 5 秒 2K 离线推理和两次串行在线请求。
在线原生 attention 也验证了同一服务从 5 秒切换到 10 秒，两条请求复用模型和 CPU 权重。
10 秒请求的单卡 PyTorch 峰值保留显存约 30.34 GiB。两条近同时提交的 5 秒请求也已验证排队与串行执行。15 秒、队列满载和长期内存趋势仍需实测。

### CUDA 文本编码器 TP8 与 FastVideo C4 配方

CUDA H3 可在八个分布式 rank 上显式设置 `--text_encoder_tp 8`。它复用 xGen 的
Qwen3-VL 文本层张量并行和 CPU 分组卸载，默认仍为 TP1。

使用上面的 CUDA FL2VA 脚本时，可设置 `GPUS=8 XGEN_FASTVIDEO_RECIPE=5s`。
配方开启文本 TP8、打包 DiT 常驻、Sparge（可用 `ATTENTION_BACKEND` 覆盖）、
FP8 PHAA、四个头分块及 gather-priority。启动时设置 NCCL CTA 上限。

| 配方 | 请求帧数 | PHAA 分块提交顺序 | NCCL_MAX_CTAS |
| --- | ---: | --- | ---: |
| `5s` | 124 | 0,1,2,3 | 2 |
| `15s` | 345 | 3,0,1,2 | 1 |

尾块优先只改变提交和计算顺序，输出按原头位置拼接。两档使用原有 50 次
DiT 配置。当前 C4 顺序已通过单卡模拟通信验证，TP8 与 CTA 配方的八卡
端到端耗时和显存仍需实测。

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

显式设置 `XGEN_DEVICE=cuda` 选择 CUDA，默认仍使用 NPU。CUDA 上可用
`--atten sage` 或 `--atten sparge` 选择对应算子，需安装对应 CUDA 扩展。

在 RTX 5090 上使用 FastVideo 的 FL2VA pruned NVFP4 权重时，在原 FL2VA
启动参数后加 `--h3_nvfp4_checkpoint ./weights/minimax_h3_fl2va_pruned_nvfp4.safetensors`。
保留 `--pretrained_model_name_or_path` 以加载模型配置、文本编码器和 VAE。
此路径需要支持 NVFP4 的 Comfy-Kitchen CUDA 算子和 BF16 激活。权重以打包格式保存，
每次投影执行激活量化。当前仅支持 FL2VA Base，暂不组合 Refiner、额外 LoRA、
AdaLN 缓存、FSDP 或其他权重量化。移除该参数即可使用常规 checkpoint。

设置 `MODEL_PATH`、`NVFP4_CHECKPOINT`、`FIRST_FRAME` 和 `PROMPT_FILE` 后，
运行 [CUDA NVFP4 推理示例](example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh)。
模型目录需有 Diffusers 的 `modular_model_index.json` 及对应组件，支持通过
`CUDA_VISIBLE_DEVICES` 选择卡、`GPUS` 设置 SP 卡数、`ATTENTION_BACKEND` 选择 attention。
单卡运行 Sparge 时可设置 `XGEN_SPARGE_HEAD_CHUNK=7`，按组执行 attention 以降低
内核临时显存。默认值 `0` 一次处理所有本地头。分块增加内核启动次数，
实际时延取决于序列长度和每卡头数。

实验性多卡通信可用 `XGEN_PHAA_FP8=1` 配合 CUDA、SP 大于 1 和
`--phaa_num` 开启。它复用 PHAA 的分头和流调度，以 E4M3 和逐头 FP32 scale
传输 Q、K、V 和 attention 输出。128 维时每头每 token 的载荷从 256 降至
132 字节，每个头分块的 Q/K/V 合并为一次 all-to-all。量化会改变结果，
端到端的画质和速度需用目标硬件验证。

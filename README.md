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

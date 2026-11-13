# xGen

xGen 是面向 NVIDIA GPU 和昇腾 NPU 的音视频生成引擎，支持命令行生成和 HTTP 服务。MiniMax-H3 可以根据文字、首尾帧或图片／视频／音频参考生成带声音的视频；v0.6 还提供独立的 H3-Image 文生图与参考图编辑入口。

[快速开始](#快速开始) · [推荐配置](#推荐配置) · [离线生成](#离线生成) · [在线服务](#在线服务) · [性能](#性能) · [完整文档](docs/README.md) · [版本记录](docs/H3-VERSIONS.md)

## 推荐配置

| 设备与用途 | 当前推荐方案 | 部署教程 |
|---|---|---|
| 8×RTX 5090，768P 音视频 | **Sage-FP8 全并行＋Streaming＋AdaLN 表**：SP8、文本 FP8 TP8、FP8 Linear／PHAA、Align16、DiT 常驻、native-stream 解码 | [CUDA 配置与回退](docs/H3-CUDA-COMPOSABLE.md) |
| 8×昇腾 910B，768P 音视频 | **VC＋Ascend 融合全并行方案**：SP8、文本 TP8、INT8 DiT、AdaLN 表、native-stream 解码 | [NPU 配置与回退](docs/H3-NPU-DEFAULT.md) |
| 4×昇腾 910B，图像生成／编辑 | **H3-Image W8A8 常驻方案**：SP4、文本 TP4、图像专用 AdaLN 和 VAE 并行 | [图像部署](docs/H3-IMAGE-NPU-OPTIMIZATIONS.md) |

视频示例统一按 **20 次实际 Base DiT**；H3-Image 是独立的 6 步模型。配置的验证设备、输入范围和人工精度状态见各报告，不保证任意满载输入都能运行。

NPU 方案需先[安装 VC 与融合依赖](docs/H3-NPU-FUSIONS.md)。当前整网性能与人工视听验收覆盖八卡 910B3、单图 R2VA、5／10 秒、20 步；其他任务、多图与时长组合需分别验证。

R2VA 条件准备重叠为[独立实验开关](docs/H3-CONDITION-OVERLAP.md)，CUDA/NPU 默认均关闭。旧 FP8 Linear＋BF16 attention 参考方案及其他回退方式保留在 CUDA 教程中。

## 快速开始

在已有匹配的 PyTorch、CUDA 或 CANN、FFmpeg 环境中，从仓库根目录执行：

```bash
python -m pip install -e ./x_base --no-deps
python -m pip install -e ./x_diffusers --no-deps
python -m pip check
python tools/collect_env.py
mkdir -p outputs
```

`--no-deps` 不会补齐依赖；缺少环境时先阅读 [CUDA 环境](docs/H3-CUDA-PRECISION.md#模型与环境)或 [NPU 环境](docs/H3-DEPLOYMENT.md#2-硬件与环境)。H3-Image 使用[独立依赖配置](docs/H3-IMAGE.md#环境与权重)。

### H3 权重下载

[MiniMax-H3：Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [ModelScope](https://modelscope.cn/models/MiniMax/MiniMax-H3)

先按[模型下载与准备](docs/H3-MODELS.md)补齐模型、工作流对应的 **20 步 AdaLN 资产**和原生 `video_vae`。T2VA/I2VA 使用 FL2VA，R2VA 使用 Ref2VA；两套资产不能混用。

## 离线生成

准备 `inputs/prompt.txt`，以及自己的首帧或参考图。以下示例使用完整模型根目录 `weights/base`；先完成[20 步资产准备](docs/H3-VIDEO-QUICKSTART.md#准备模型和输入)，再设置：

```bash
export BASE_MODEL=weights/base
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
```

仅对可信的原生 VAE 代码开启授权。`video_vae` 与 Diffusers 的 `vae` 子目录不同。

**CUDA／RTX 5090 八卡**，分别为文生视频、首帧图生视频和单图参考生视频：

```bash
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh cuda r2va 15 768p inputs/ref01.png
```

**昇腾 NPU 八卡**，采用 VC＋Ascend 融合、INT8 全并行默认方案：

```bash
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh npu r2va 15 768p inputs/ref01.png
```

每条命令独立启动；时长可填 `5`、`10`、`15`，输出写入 `outputs/`。多图、首尾帧、2K 和常驻模型复用见[完整视频教程](docs/H3-VIDEO-QUICKSTART.md)。图像生成请用[H3-Image 入口](docs/H3-IMAGE.md)。

## 在线服务

先修改对应示例 JSON 中的模型和 VAE 路径，准备匹配的 20 步资产。下面两个服务均接受 T2VA/I2VA 请求，每次选择一个启动：

```bash
# CUDA
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001

# NPU
XGEN_DEVICE=npu torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

NPU 启动前还需设置[配套环境变量](docs/H3-VIDEO-QUICKSTART.md#npu)。用 `/generate` 提交、查询并下载视频，R2VA 使用独立的 Ref2VA 实例。[请求示例与下载](docs/H3-ONLINE.md)

对接平台时，可在 CUDA 服务启用 `--maas`，获得 V2 接口、排队、健康检查和原子发布；NPU MaaS V2 尚未开放。真实输出目前为 768P，IR／2K 仅有默认关闭的模拟后端。[MaaS 启动](docs/H3-ONLINE.md#cuda-maas) · [API 文档](server/video/MAAS-API.md)

## 性能

以下均为 **20 步、768P、完整预热后的单次正式样例**，排除初始化。CUDA 为服务端推理计时，NPU 为常驻推理至保存计时，不能直接跨设备比较。

| 设备与方案 | 视频样例 | 耗时 | 完整报告 |
|---|---|---:|---|
| 8×5090，Sage-FP8 全并行 Streaming＋AdaLN＋Align16 | T2VA，5 秒 | 33.517 s | [12 项在线性能与显存](docs/performance/h3-cuda-phaa-align16.md) |
| 同上 | R2VA 单图，5 秒 | 46.672 s | [同一报告](docs/performance/h3-cuda-phaa-align16.md) |
| 8×910B3，VC＋Ascend 融合全并行 | R2VA 单图，5 秒 | 77.202 s | [NPU 性能与验收范围](docs/performance/h3-v06-npu-vc-fusions.md) |
| 同上 | R2VA 单图，10 秒 | 170.123 s | [同一报告](docs/performance/h3-v06-npu-vc-fusions.md) |

CUDA 12 项已通过技术检查，完整人工动态／口型／听音复核待记录；NPU 验收限定上述已评审样例。更多 5／10／15 秒、2K、历史 50 步及 H3-Image 数据见[性能目录](docs/performance/README.md)，历史值保留原始条件。

## 更多文档

[完整文档目录](docs/README.md) · [版本与升级](docs/H3-VERSIONS.md) · [v0.6 发布说明](docs/H3-V06-RELEASE.md) · [Agent 启动指南](.agents/skills/xgen-h3/SKILL.md)

仓库发布源码、示例、测试及脱敏性能记录。模型、输入素材、成片和内部机器配置由部署方管理。其他模型见 [Diffusers 示例](example/diffusers)，界面接入见 [ComfyUI](x_comfyui)。

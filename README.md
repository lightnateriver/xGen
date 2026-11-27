# xGen

xGen 是面向 NVIDIA GPU 和昇腾 NPU 的音视频生成引擎，提供命令行、HTTP 和集成适配。MiniMax-H3 支持文字、首尾帧及图片/视频/音频参考生成；H3-Image 提供独立的图像生成与编辑。

[快速开始](#快速开始) · [视频](docs/H3-VIDEO-QUICKSTART.md) · [图像](docs/H3-IMAGE.md) · [在线服务](docs/H3-ONLINE.md) · [架构](docs/ARCHITECTURE.md) · [文档目录](docs/README.md)

## 快速开始

在已有匹配的 PyTorch、CUDA 或 CANN/torch_npu、FFmpeg 环境中，从仓库根目录执行：

```bash
python -m pip install -e ./x_base --no-deps
python -m pip install -e ./x_diffusers --no-deps
python -m pip check
python tools/collect_env.py
```

`--no-deps` 不补齐环境。模型、工作流、AdaLN 表和原生 VAE 须按 [模型与资产](docs/H3-MODELS.md) 配套准备；图像使用 [独立环境](docs/H3-IMAGE.md#environment)。

## 推荐配置

| 用途 | 入口与配置 |
|---|---|
| 八卡 CUDA Base 视频 | [Sage-FP8 全并行、native-stream、BF16 AdaLN 表](docs/H3-CUDA.md) |
| 八卡 NPU Base 视频 | [VC、Ascend 融合、INT8、full-parallel、native-stream](docs/H3-NPU-DEFAULT.md) |
| 四卡 NPU 图像 | [H3-Image 六步、文本 TP4/W8A8 与图像专用优化](docs/H3-IMAGE.md#npu-profile) |

## H3 权重下载

视频：[Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [ModelScope](https://modelscope.cn/models/MiniMax/MiniMax-H3)。图像：[H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)。下载布局、固定资产版本和离线预处理统一见 [模型文档](docs/H3-MODELS.md)。

## 离线生成

完成相应设备的资产准备后，显式选择实际采样步数：

```bash
export BASE_MODEL=weights/base
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

原生模型代码授权仅用于可信目录。NPU、首尾帧、Ref2VA 多媒体和 2K 的配套资产与命令见 [视频指南](docs/H3-VIDEO-QUICKSTART.md)，图像入口见 [Image](docs/H3-IMAGE.md)。

## 在线服务

按 [在线指南](docs/H3-ONLINE.md) 选择 Base、Refiner 或 CUDA MaaS，先填写对应启动 JSON。普通 HTTP 使用 `/generate`；MaaS V2 的字段和状态以 [API](server/video/MAAS-API.md) 为准。启动精度、并行和模型配置不能由逐请求字段覆盖。

## 架构与开发

通用算子/并行机制在 `x_base`，模型实现按模型收拢到 framework，加载装配和请求编排在 adaptor，文件和服务协议在用户入口。分层职责、H3 文件组织和检查方式见 [架构文档](docs/ARCHITECTURE.md)。

模块公共接口由 [x_base](x_base/README.md)、[Diffusers 集成](x_diffusers/README.md)、[示例](example/README.md) 和 [工具](tools/README.md) 各自维护。

[统一配置](docs/H3-CONFIGURATION.md) · [版本与迁移](docs/H3-VERSIONS.md) · [Agent 启动指南](.agents/skills/xgen-h3/SKILL.md)

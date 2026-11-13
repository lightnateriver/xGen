# 模型下载与资产准备

[文档目录](README.md) · [20 步视频启动教程](H3-VIDEO-QUICKSTART.md)

下载权重后还需准备匹配的 AdaLN 表和 Transformer。CUDA、NPU 使用各自的准备工具，不能混用表或剔除权重后的 checkpoint。当前推荐示例统一为 **20 次实际 Base DiT**；NPU 的对应表包含 21 个调度点，文件名中的 `21step` 不代表执行 21 次 DiT。

## 音视频模型

按工作流下载资产。H3 Base 仓库包含文本编码器、视频 VAE、音频 VAE 和调度组件。T2VA 和首帧 I2VA 使用 FL2VA/Base Transformer，R2VA 使用 Ref2VA Transformer。保留 xGen 所需的目录布局。

| 资产 | Hugging Face | ModelScope | 用途 |
|---|---|---|---|
| MiniMax-H3 Base BF16 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | [MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) | 768P Base。按任务使用 `transformer/` 或 `transformer_ref/`，并保留根目录的 `modular_model_index.json` |
| Open-H3-Regenerate-2K | [权重仓库](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) | [权重仓库](https://modelscope.cn/models/Efficient-Large-Model/Open-H3-Regenerate-2K) | 2K 链路所需的 latent upscaler 和 Refiner 资产，按部署教程补齐 Sharp AdaLN 表并预处理 |
| FL2VA Pruned NVFP4 | [实测版本的单文件](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4/blob/8c5abfed61e1b6a170240792b65253fba1a65b7b/minimax_h3_fl2va_pruned_nvfp4.safetensors) | — | 历史 CUDA 高性能 FL2VA 与 CUDA 2K 的 Base DiT；当前默认 Base / MaaS 配方不需要此文件。另需匹配的 H3 模型组件 |
| LightX2V Turbo 8 步 LoRA | [768P Diffusers BF16 文件](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors) | [LightX2V Turbo](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | 可选的 8 步 FL2VA/T2VA 配方 |
| Veda FP8 预测器 | [实测版本的单文件](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/76f202874608115408d73280be9531c4ab888242/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors) | [Veda 预测器仓库](https://modelscope.cn/models/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) | 仅启用 Veda 时需要 |

下载和预处理步骤见 [Base 和 Refiner 资产准备](H3-DEPLOYMENT.md#4-模型与目录准备)、[CUDA 组件与 NVFP4 下载](H3-CUDA.md#环境和模型) 和 [Veda / Turbo 配方](H3-ATTENTION.md)。AdaLN 表和预处理后的 Refiner 由部署方按配方生成。

## 下载原生视频 VAE

Native-stream 还需要官方原生 [FL2VA/video_vae 完整目录](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/5d9b308a59ab12e67147f191e184baf704185bd1/FL2VA/video_vae)，包括 Python 实现、`config.json` 和 `source/` 内的权重。它不包含在仅下载 Diffusers `vae/` 的选择性下载中。

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 5d9b308a59ab12e67147f191e184baf704185bd1 \
  --include 'FL2VA/video_vae/*' --local-dir weights/native
```

下载后路径为 `weights/native/FL2VA/video_vae`，使用 `H3_VAE_MODEL_PATH` 指向此处；不要指向其父目录或仅指向 `source/`。已部署环境也可继续使用经过验证的自有目录。[加载与授权说明](H3-OMNI-VAE.md)

## 按用途准备

| 用途 | 准备说明 |
|---|---|
| CUDA T2VA / I2VA / 首尾帧 | [FL2VA 的 20 步 CUDA 表和剔除权重 checkpoint](H3-CUDA-ADALN.md#一次性准备) |
| CUDA R2VA | 同一工具选择 Ref2VA，独立生成对应资产；不能复用 FL2VA 表 |
| NPU 768P | [20 步启动与资产路径](H3-VIDEO-QUICKSTART.md#准备模型和输入)；[下载与预处理工具](H3-DEPLOYMENT.md#4-模型与目录准备) |
| 原生视频 VAE | 按上方命令下载；[模型布局与授权](H3-OMNI-VAE.md)，不同于 Diffusers 的 `vae` 子目录 |
| NPU 2K | 另准备 [Regenerate Refiner](H3-DEPLOYMENT.md#42-refiner)，Base 20 步与 Refiner 3 步分开配置 |
| H3-Image | 使用[图像专用模型、VAE 和依赖](H3-IMAGE.md#环境与权重)，不使用视频 AdaLN 表 |

NVFP4、Veda 和 Turbo 是可选／历史配方所需资产，不是当前 Sage-FP8 或 Laser INT8 默认视频方案的前置条件。仅对已确认可信的原生 VAE 模型代码开启授权。

# 环境、模型与资产准备

[文档目录](README.md) · [视频生成](H3-VIDEO-QUICKSTART.md) · [图像生成](H3-IMAGE.md)

先准备匹配的 PyTorch CUDA 或 CANN/torch_npu、Diffusers、Transformers，以及支持 H.264/AAC 的 FFmpeg 和 FFprobe。图像模型使用独立的固定 Diffusers 版本，见 [Image 环境](H3-IMAGE.md#environment)。

```bash
python -m pip install --no-deps -e ./x_base -e ./x_diffusers
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
python -m pip check
python tools/collect_env.py
```

`--no-deps` 使用现有依赖，不自动补齐运行环境。资产准备在目标设备环境单独执行；源模型保留，结果写入指定的新目录。

<a id="environment-cuda"></a>
## CUDA 环境

默认 H3 方案使用八张支持所选 FP8 GEMM 的 GPU，SageAttention 与 PyTorch、CUDA、Triton 需要匹配。保留的 BF16 attention 参考方案使用 cuDNN dense attention。具体精度选择见 [CUDA](H3-CUDA.md)；不能只改 attention 开关就假定其他算子和通信精度同步改变。

<a id="environment-npu"></a>
## NPU 环境

使用与 CANN 匹配的 PyTorch/torch_npu。推荐 VC 与融合配置还需 AscendC、CMake、C++17 工具链及对应设备构建产物，安装命令集中在 [NPU 算子](H3-NPU-DEFAULT.md#operators)。

<a id="base"></a>
## Base 权重与工作流

| 资产 | 来源 | 用途 |
|---|---|---|
| MiniMax-H3 Base | [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [ModelScope](https://modelscope.cn/models/MiniMax/MiniMax-H3) | T2VA/I2VA 使用 FL2VA/Base Transformer，Ref2VA 使用独立 Transformer |
| Open-H3-Regenerate-2K | [Hugging Face](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) · [ModelScope](https://modelscope.cn/models/Efficient-Large-Model/Open-H3-Regenerate-2K) | 可选 latent upscaler 和 2K Refiner |
| MiniMax-H3-Image | [模型仓库](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | 独立的六步图像模型及图像 VAE |

Base 示例保留固定的组件版本：

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 6818f6c32d12b210915e44ad56a4228c2608f160 \
  --local-dir weights/base \
  --include '*.json' 'LICENSE' 'scheduler/*' 'audio_scheduler/*' \
    'processor/*' 'tokenizer/*' 'text_encoder/*' 'transformer/*' \
    'transformer_ref/*' 'vae/*' 'audio_vae/*'
```

保留根级 `modular_model_index.json` 和全部组件。只下载原生 `FL2VA/` 或单个 Transformer 不能直接替代这个 Diffusers 根目录。

```text
weights/base/transformer/           原始 FL2VA Transformer
weights/base/transformer_ref/       原始 Ref2VA Transformer
weights/base/...                    文本、处理器、VAE、音频和调度组件
weights/native/FL2VA/video_vae/      独立原生视频 VAE
model-cache/                       NPU 预处理资产
```

<a id="native-vae"></a>
## 原生视频 VAE

Native-stream 使用原生 `video_vae`，包括 Python 实现、`config.json` 和 `source/` 权重；它不同于上方转换后的 `vae/` 组件。

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 5d9b308a59ab12e67147f191e184baf704185bd1 \
  --include 'FL2VA/video_vae/*' --local-dir weights/native
```

设置 `H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae`。仅对已确认可信的本地模型代码设置 `H3_VAE_TRUST_REMOTE_CODE=1`。后端、量化和编译约束见 [VAE 配置](H3-CONFIGURATION.md#vae)。

<a id="adaln-cuda"></a>
## CUDA AdaLN 资产

20 指实际 DiT 执行次数。工具为 FL2VA 和 Ref2VA 分别生成表，并将已缓存的投影从新 Transformer 中物理移除；原始权重保留用于 owner 回退。

```bash
export XGEN_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0 python tools/adaln_cache/prepare_cuda_adaln.py \
  --model_path weights/base/transformer --model_variant FL2VA --steps 20 \
  --output_dir weights/base/xgen-adaln/fl2va/tables \
  --strip_output weights/base/xgen-adaln/fl2va/transformer

CUDA_VISIBLE_DEVICES=0 python tools/adaln_cache/prepare_cuda_adaln.py \
  --model_path weights/base/transformer_ref --model_variant Ref2VA --steps 20 \
  --output_dir weights/base/xgen-adaln/ref2va/tables \
  --strip_output weights/base/xgen-adaln/ref2va/transformer
```

默认目录包含 `transformer/` 和 `tables/adaln_<variant>_bf16_20step.safetensors`。可显式生成其他步数的表，但须保持权重、工作流、采样点、shift 与 LoRA 一致。表模式的选择和 owner 回退见 [配置](H3-CONFIGURATION.md#adaln)。

<a id="adaln-npu"></a>
## NPU AdaLN 资产

在已加载 CANN 的 NPU 环境执行。该工具使用 NPU 0；20 次计算对应 21 个调度点，文件名里的 `21step` 不表示执行 21 次 DiT。

```bash
export XGEN_DEVICE=npu
python tools/adaln_cache/prepare_h3_base_cache.py \
  --source weights/base/transformer --model-variant T2VA --steps 20 \
  --output model-cache/t2va-21step.safetensors \
  --stripped-output model-cache/base-t2va

python tools/adaln_cache/prepare_h3_base_cache.py \
  --source weights/base/transformer_ref --model-variant Ref2VA --steps 20 \
  --output model-cache/ref2va-21step.safetensors \
  --stripped-output model-cache/base-ref2va
```

第一组用于 T2VA/I2VA，第二组用于 Ref2VA。CUDA 与 NPU 的准备工具、元数据和目录布局分别管理，不混用资产。51 点旧表仍对应 50 次实际计算，不能改文件名后按 20 步使用。

<a id="refiner"></a>
## 2K Refiner 资产

源包需包含 `base_model/`、`latent_upsampler/`、组合 LoRA、Sharp curve 分支与 `adaln_t_table.bin`。上游固定准备脚本可补齐这些文件：

```bash
mkdir -p model-cache
curl --fail --location \
  'https://raw.githubusercontent.com/NVlabs/Sana/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K/download.py' \
  --output model-cache/download-refiner.py
python model-cache/download-refiner.py --weights weights/refiner
XGEN_DEVICE=npu python tools/prepare_h3_refiner.py \
  --weights weights/refiner --output model-cache/refiner
```

xGen 准备工具合并组合 LoRA、核对 Sharp 曲线并生成 AdaLN 表、Transformer 和 manifest；不要在启动时再次重复合并。工具也可按选定 CUDA 后端准备资产，但 CUDA 2K 仍需独立的 packed NVFP4 FL2VA Base，见 [视频 2K](H3-VIDEO-QUICKSTART.md#refiner)。

源模型、LoRA 或调度发生变化时，使用新的配套输出；缓存刷新使用工具的 `--refresh-cache`，它不替代受影响权重的重新合并。资产生成与服务请求分开，加载入口不会自动生成这些大文件。

<a id="optional-assets"></a>
## 可选模型资产

| 能力 | 资产和配套要求 |
|---|---|
| Packed NVFP4 FL2VA | [固定版本权重](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4/blob/8c5abfed61e1b6a170240792b65253fba1a65b7b/minimax_h3_fl2va_pruned_nvfp4.safetensors)；另需完整 H3 组件及兼容 CUDA 算子 |
| Veda | [固定预测器](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/76f202874608115408d73280be9531c4ab888242/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors)；配套编译和参数见 [attention](H3-CONFIGURATION.md#attention) |
| LightX2V Turbo8 | [768P BF16 LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors)；需要预合并 Transformer 与八次计算的配对表 |

Turbo8 资产用现有 NPU 工具准备：

```bash
XGEN_DEVICE=npu python tools/adaln_cache/prepare_h3_base_cache.py \
  --source weights/base/transformer --model-variant T2VA \
  --turbo8-lora weights/turbo8.safetensors \
  --stripped-output model-cache/turbo8-transformer \
  --output model-cache/turbo8-adaln.safetensors
```

启用 Turbo8 时同时选择 `--h3_turbo8 --num_inference_steps 8`、上述 Transformer 和表。适用范围及回退见 [配置](H3-CONFIGURATION.md#turbo)。这些资产均非默认 Base 配方的必要组成部分。

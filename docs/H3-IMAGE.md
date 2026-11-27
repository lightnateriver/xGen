# H3-Image 图像生成与编辑

[文档目录](README.md) · [模型准备](H3-MODELS.md) · [架构](ARCHITECTURE.md#entrypoints)

H3-Image 是独立的六步图像模型，使用图像 VAE 和每步 PDD 输出头，保存 PNG。视频模型的 AdaLN 表、音频组件和 MP4 解码配方不用于此入口。

<a id="environment"></a>
## 环境与权重

在独立图像环境安装正常 xGen 依赖，再安装固定的 Image Diffusers 源码：

```bash
python -m pip install --no-deps -r x_diffusers/requirements-h3-image.txt
X_DIFFUSERS_PROFILE=h3-image python -m pip install --no-deps -e ./x_diffusers
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
```

图像依赖固定到 Diffusers 提交 `0905b667fe3f37f43b6ef9c6e05ad20636206938`，普通视频依赖不包含这些类。模型来源为 [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)。

本地目录包含 `modular_model_index.json`、`transformer_diffusers/`、`vae_diffusers/`、`text_encoder/`、`processor/`、`tokenizer/`、`scheduler/` 和 `pdd_config.json`，保留组件混合精度。

<a id="npu-profile"></a>
## NPU 四卡方案

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export H3_IMAGE_MODEL="$PWD/weights/MiniMax-H3-Image"
bash example/diffusers/scripts/run_h3_image_npu.sh \
  --prompt 'A product photograph of a green ceramic bowl on a neutral background.' \
  --width 4096 --height 4096 --save_path outputs/h3-image/t2i.png
```

此启动器选择以下配置，用户参数在其后可显式覆盖：

| 组件 | 配置 |
|---|---|
| 文本编码 | 保留所需 Qwen decoder 层，文本 TP4、W8A8 |
| Image DiT | SP4、W8A8 常驻、图像 AdaLN 表 |
| 条件与计算 | 请求内条件缓存、局部输入、Token Refiner SP、视觉 SP、FF chunk 2048 |
| 图像 VAE | 原 checkpoint 混合精度与拼接规则，tile 并行 |
| 输出 | RGB8 传输，rank 0 保存 PNG 和运行 JSON |

默认 dense SDPA、六步、无 CFG，不接受负面提示词或任意步数。纯 Laser 不作为此方案的默认后端。量化、常驻和并行是独立设置，SP 只分片序列，不自动分片全部权重。

<a id="editing"></a>
## 参考图编辑

```bash
bash example/diffusers/scripts/run_h3_image_npu.sh \
  --prompt 'Replace the background in <Picture 1> with a plain gray backdrop.' \
  --image_path_list inputs/reference.png \
  --width 2048 --height 2048 --save_path outputs/h3-image/edit.png
```

图片按参数顺序对应 `<Picture 1>`、`<Picture 2>` 等，最多 16 张。视觉编码和图像 VAE 各自使用模型的预处理，保持参考图长宽比。

默认最小生成面积为 4 MP；指定较小输出可能先生成更大画布再缩放。仅需小尺寸功能检查时可显式指定 `--min_generation_area 1024 --width 1024 --height 1024`，并记录该配置。

<a id="text-precision"></a>
## 精度与资源选项

NPU 文本 W8A8 在 CLI 和 Python 加载入口默认开启；`--no-image_text_int8` 回退 BF16，CUDA 文本默认 BF16。文本量化仅作用于保留 decoder 层的 attention/MLP 线性投影，embedding、norm、视觉塔等保持原精度。

| 参数 | 用途与限制 |
|---|---|
| `--sp`、`--text_encoder_tp` | 控制序列和文本并行；匹配实际进程数 |
| `--image_dit_tp` | 可选 DiT TP，和 SP 布局共同校验 |
| `--image_resident`、`--image_int8` | DiT 常驻和 W8A8，分别控制 |
| `--image_text_resident` | 显式文本常驻；文本 TP 配置已有自身驻留策略 |
| `--image_adaln_table` | 当前 checkpoint 六步的图像调制表；不能使用视频表 |
| `--image_condition_cache`、`--image_local_input` | 请求内固定条件复用及局部输入 |
| `--image_refiner_sp`、`--image_vision_sp` | Token Refiner 和视觉塔并行 |
| `--image_vae_parallel` | 原 VAE tile 任务分派，保留拼接规则 |
| `--image_vae_decode_batch_size`、`--image_vae_gather_once` | 解码 tile 批量和收集方式，需考虑额外暂存 |
| `--image_ff_chunk` | FFN 分块；0 保留原计算路径 |
| `--image_rgb8_output` | 已量化像素按 uint8 传回并输出 |
| `--image_swiglu` | `none/native/triton`，可选内核依赖仅在启用时加载 |
| `--warmup`、`--repeat` | 同一 Pipeline 的预热和重复运行 |

需要低层显式配置时使用 `example/diffusers/scripts/run_h3_image.sh` 或其 Python 入口，先检查 `python example/diffusers/infer_minimax_h3_image.py --help`。`--no-image_text_int8` 是文本精度回退，不会自动关闭 NPU 启动器中的 DiT 或其他优化。

<a id="python"></a>
## Python 与服务集成

`x_diffusers.adaptor.load_pipe.load_pipe(args, model="MiniMax-H3-Image")` 负责模型加载和装配。`h3_runtime.generate_h3_images(pipe, ...)` 接收内存参考图与参数，成功或失败后清理条件缓存。示例入口的 `run_image(args, pipe=None)` 负责路径、计时和输出文件。

图像服务入口和协议见 [Image 服务](../server/image/README.md)。视频的 `/generate`、MaaS 视频协议和采样参数不能直接用于此模型。输出尺寸、参考数量与设备容量应按实际配置检查，数学对照不替代图像质量评审。

# MiniMax-H3-Image：NPU 四卡接入

独立的文生图／参考图编辑入口，基于 v0.5 `1476c54`。输出 PNG，使用 H3 Image 的图像 VAE、固定六步采样和每步独立 PDD 输出头。视频模型的 AdaLN 表、音频 VAE、Streaming MP4 配方不用于这个入口。

## 环境与权重

在独立环境中安装 xGen 的正常依赖，再安装图像入口的可选依赖：

```bash
pip install --no-deps -r x_diffusers/requirements-h3-image.txt
X_DIFFUSERS_PROFILE=h3-image pip install --no-deps -e ./x_diffusers
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers:$PYTHONPATH"
```

可选 Diffusers 固定到官方提交 `0905b667fe3f37f43b6ef9c6e05ad20636206938`。常规 v0.5 的 Diffusers 0.40 不包含 H3 Image 类；保留视频环境，单独建立图像环境。

`X_DIFFUSERS_PROFILE=h3-image` 仅替换图像环境的 Diffusers 依赖声明，避免和默认 0.40 固定版本冲突。不设置这个变量时仍沿用视频环境的依赖。

本地权重目录必须包含 `modular_model_index.json`、`transformer_diffusers/`、`vae_diffusers/`、`text_encoder/`、`processor/`、`tokenizer/`、`scheduler/` 和 `pdd_config.json`。加载使用本地文件，保留组件的混合精度，不修改共享权重。

## 四卡方案

| 阶段 | 首版方案 |
|---|---|
| Qwen3-VL 文本编码 | 保留需要的前 50 层及 pre-norm 输出，TP4 |
| 文本 Token Refiner | 完整文本序列，复用 xGen 注意力 |
| 图像 DiT | Ulysses SP4；56 个 attention heads，每卡 14 个；仅主干序列分片 |
| DiT 权重 | BF16，按块 CPU/NPU 卸载；SP 不分片权重 |
| PDD 输出 | 原版 FP32 输出头，汇集完整序列后按官方图像索引选行 |
| 图像 VAE | 官方图像 VAE，保留敏感模块 FP32；首版在各 rank 执行相同解码 |
| 输出 | 仅 rank 0 保存 PNG 与测量 JSON |

四个 rank 需要足够的主机内存保存各自的 CPU 权重。原始 DiT 约 66 GB，不能在 64 GiB 910B 上与文本编码器一起完整常驻。此版优先建立 BF16 正确性基线；INT8、图像专用 AdaLN 表、VAE 并行需要单独校验。

## 文生图

先确认选定的四张卡没有其他任务，再运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export H3_IMAGE_MODEL=/mnt/sfs_turbo/lzp/models/MiniMax-H3-Image
bash example/diffusers/scripts/run_h3_image.sh \
  --prompt 'A studio photograph of a red ceramic teapot beside a white porcelain cup, soft window light, natural shadows, sharp details.' \
  --width 4096 --height 4096 --save_path outputs/h3-image/t2i.png
```

默认用密集 SDPA。可显式传 `--atten laser` 选择 xGen 的 NPU Laser；缺少 MindIE-SD 时直接报错。固定六步、无 CFG，不接受负面提示词或任意采样步数。

## 参考图编辑

```bash
bash example/diffusers/scripts/run_h3_image.sh \
  --prompt 'Keep the composition in <Picture 1>, but change the teapot to blue ceramic.' \
  --image_path_list inputs/reference.png \
  --width 2048 --height 2048 --save_path outputs/h3-image/edit.png
```

参考图片按参数顺序对应 `<Picture 1>`、`<Picture 2>` 等，最多 16 张。官方预处理分别处理视觉编码器图片和 VAE 参考图片，保留参考图的长宽比例。

默认最小生成面积为 4 MP，小尺寸输出仍可能先生成较大画布再缩放。技术冒烟测试可显式使用 `--width 1024 --height 1024 --min_generation_area 1024`，直接生成 1 MP；不能把这种测试作为默认 4 MP／16 MP 的性能结果。

`--warmup 1 --repeat 2` 在同一常驻 pipeline 中测量。初始化和最大 rank 的生成耗时分别记录；`inference_s` 包含图像生成与 VAE 解码，不包含 PNG 写盘，不等于 HTTP 请求延迟。

## 校验

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29673 \
  tools/validate_h3_image_sp.py --device npu
# 实际 checkpoint 的 BF16 主干 + FP32 输入／时间嵌入／PDD 输出头：
torchrun --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29673 \
  tools/validate_h3_image_sp.py --device npu --dtype bf16
```

小模型验证覆盖 13／16／19 个交错 token 的序列与全部六个输出头；检查序列尾 padding、每行 AdaLN 索引、RoPE 和完整图像行顺序。测试使用官方未并行 forward 为参考。

真实权重的出图结果、运行配置及技术检查另行记录。数学测试或 PNG 可解码不代表人工画质验收，也不证明其他尺寸和所有参考图数量都已验证。

### 2026-11-11 首版验证

环境：4× Ascend 910B1，Torch 2.10.0、Torch-NPU 2.10 系列，专用容器；主干 BF16、SDPA、SP4、文本 TP4、按块卸载。使用模型仓库版本 `fbde899fffb5c5225ee8dd7027680113746af42e`。

| 检查 | 结果 |
|---|---|
| FP32 小模型：3 个序列长度 × 6 个 PDD 头 | 18 项通过，与官方 forward 逐元素一致 |
| 混合精度小模型：BF16 主干、FP32 输入／时间嵌入／PDD 头 | 18 项通过，与官方 forward 逐元素一致 |
| 视频采样调度与启动配方 CPU 回归 | 22 项通过 |
| 1024×1024 文生图，显式 1 MP 画布 | PNG 正常；首次初始化 81.90 s，首次生成 176.81 s |
| 1024×1024 单参考图编辑 | PNG 正常；预先加载 pipeline 后单次生成 104.75 s |
| 2048×2048 文生图，复用编辑后的 pipeline | PNG 正常；单次生成 105.81 s，最大 rank 已分配显存约 21.22 GiB |

这些是单样例技术验证；未做多轮统计、HTTP 延迟或同条件性能比较。生成时间含文本编码、DiT 和图像 VAE，不含 PNG 写盘。首次生成包含首次权重触达和预取追踪；另外两项使用同一预先加载的 pipeline，均未额外预热。

图像抽查中，文生图呈现红色茶壶与白杯，编辑图把茶壶改为蓝色并保留主要构图；仍需用户画质验收。4096 画布、多参考图、Laser 实际大形状、CUDA 与 HTTP 接口未在此版验证。

上游卸载预取追踪在第一步仅执行第一个 PDD 头时会提示其他头未执行；日志保留此提示。上述完整六步与 pipeline 复用均通过。此版本仍采用 BF16 卸载基线，未启用视频配方中的 INT8 或 AdaLN 缓存。

官方参考：[Diffusers H3 Image 文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)、[模型仓库](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)。

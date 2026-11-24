# CUDA H3：AdaLN 缓存表

v0.5 的八卡 CUDA H3 Base 默认使用 **Sage-FP8 全并行 + native-stream + BF16 AdaLN 表**。T2VA、I2VA、首尾帧使用 FL2VA 资产，R2VA 使用 Ref2VA 资产。NPU 继续使用自己的预处理工具和默认配方。

表替代时间编码与 AdaLN 投影；对应的 106 个原始参数从新的 Transformer checkpoint 中物理移除。沿用现有 sampler、缓存加载器、调制算子与 DiT 循环，不增加第二套推理调度。主 DiT 的 50 层仍常驻。

## 一次性准备

安装原 CUDA 推理依赖，从仓库根目录执行。以下目录都是示例，可自行替换。工具写入新目录，不覆盖原始模型。

```bash
export XGEN_DEVICE=cuda
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers"
CUDA_VISIBLE_DEVICES=0 python tools/adaln_cache/prepare_cuda_adaln.py \
  --model_path weights/h3-fl2va/transformer \
  --model_variant FL2VA --steps 20 \
  --output_dir weights/h3-fl2va/xgen-adaln/fl2va/tables \
  --strip_output weights/h3-fl2va/xgen-adaln/fl2va/transformer
```

R2VA 需独立准备，不能混用 FL2VA 权重或表：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/adaln_cache/prepare_cuda_adaln.py \
  --model_path weights/h3-ref2va/transformer_ref \
  --model_variant Ref2VA --steps 20 \
  --output_dir weights/h3-ref2va/xgen-adaln/ref2va/tables \
  --strip_output weights/h3-ref2va/xgen-adaln/ref2va/transformer
```

`20` 是本页推荐的实际 DiT 次数；需要兼容 50 步时可显式使用 `--steps 20 50` 同时生成两套表。默认视频／音频 scheduler shift 为 12／3。工具只将时间 MLP 和当前投影放到一张 CUDA 卡，并校验查表输出；仍需足够磁盘保存剔除后的 Transformer。权重、表和测试素材不随 Git 仓库发布。

FL2VA 表保存无视觉／有视觉条件，Ref2VA 表保存无参考／视频类／音频类／两类同时存在的完整时间批次。图片和参考视频共享固定视觉条件时间 0.999；参考音频也使用固定条件时间。这些是模型内部时间值，与文件时长不同。prompt、参考图数量和输出秒数不改变时间表；权重、步数、时间序列、scheduler shift 或相关 LoRA 改变时需重新匹配。

## 默认使用与自定义位置

```bash
export BASE_MODEL=weights/h3-fl2va
export H3_BASE_STEPS=20
export H3_VAE_MODEL_PATH=weights/h3-native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

仅对可信的原生 VAE 模型代码授权。默认从 `BASE_MODEL/xgen-adaln/fl2va` 或 `ref2va` 查找 `transformer/` 及 `tables/adaln_<variant>_bf16_<steps>step.safetensors`。

| 配置项 | 含义 |
|---|---|
| `--h3_cuda_adaln_mode auto` | 全并行 Sage 配方自动使用表；旧精度方案保持原行为 |
| `--h3_cuda_adaln_mode table` | 显式要求表模式 |
| `--h3_cuda_adaln_mode owner` | 使用原始 checkpoint 的 rank-owner AdaLN；不使用表 |
| `--h3_cuda_adaln_dir DIR` | 指定含 `transformer/` 和 `tables/` 的资产目录 |
| `--transformer_path` ＋ `--adaln_cache_file` | 显式指定固定配对，不能再传资产目录 |

脚本环境变量为 `H3_CUDA_ADALN_MODE`、`H3_CUDA_ADALN_DIR`。HTTP JSON 使用同名小写字段。目录模式按每次请求的实际步数选表；相同表和条件复用已加载结果，变更时加载相应表，不重建或重新量化主模型。显式固定表仍只接受匹配的步数。接口不允许请求任意传入权重／表路径。

资产缺失、模型类型／步数／权重指纹／scheduler shift 不一致，会在加载主模型前报错，不自动生成大文件或降级。实际条件表在进入 DiT 前再次核对。表模式和 checkpoint 不能在已加载服务内切换，需重启。仅支持原生 SP，TP/CP 请显式使用 `owner`。

## 回退

保留 Sage 全并行，仅关闭查表：

```bash
H3_CUDA_ADALN_MODE=owner bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

回到原 FP8 Linear、BF16 attention／通信参考方案：

```bash
H3_CUDA_PRECISION=fp8-bf16 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

服务中关闭查表时，还需移除已配置的 `h3_cuda_adaln_dir` 或固定的两个路径字段，再重启使用原始模型。切换旧精度方案时使用 `h3_cuda_bf16.example.json`，其中已设置预设为 none、AdaLN 为 owner。

## 空间与精度边界

本轮 FL2VA 原始 106 个参数合计 24.347 GiB；20／50 步表文件分别约 906.590／2294.228 MiB，包含全部条件。每个 rank 只加载当前条件；文件大小不等于每卡显存。对应对照峰值 allocated 约减少 3 GiB／卡，没有统一的耗时收益。

48 项局部查表对照逐位一致；整网 owner／table 对照抽帧 SSIM 为 0.646～0.934，音频亦有差异。因此局部对照不代表整片逐位等价或人工精度认可。新默认需按业务素材审核画面、口型和声音。[最新矩阵及已知容量限制](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda-streaming-matrix.md)

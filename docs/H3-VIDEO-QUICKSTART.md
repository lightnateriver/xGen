# H3 视频生成

[文档目录](README.md) · [模型与资产](H3-MODELS.md) · [在线服务](H3-ONLINE.md)

MiniMax-H3 支持文字生成带声音的视频、首帧/首尾帧和图片/视频/音频参考。CLI 中首帧与首尾帧均使用 `fl2va` 工作流；启动脚本用 `i2va` 表示首帧入口。Ref2VA 使用独立权重。

| 工作流 | 脚本参数 | 模型资产 |
|---|---|---|
| T2VA | `t2va` | FL2VA/Base Transformer 与表 |
| 首帧、首尾帧 | `i2va` | FL2VA/Base Transformer 与表 |
| 多媒体参考 | `r2va` | Ref2VA Transformer 与表 |

<a id="inputs"></a>
## 准备模型和输入

完成 [环境与资产准备](H3-MODELS.md)，从仓库根目录运行。下面各命令独立启动八个 rank；按实际配置选择已分配的可见设备。

```bash
export BASE_MODEL=weights/base
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
mkdir -p outputs
```

`PROMPT_FILE` 为 UTF-8 提示词。只对可信的原生 VAE 目录开启授权。20 指实际 Base DiT 次数；底层 CUDA/2K 脚本保留 50 步默认，因此本指南显式设置 20。步数变化时需要匹配的 AdaLN 表。

<a id="cuda"></a>
## CUDA 768P

默认选择 Sage-FP8 全并行、native-stream 与 BF16 AdaLN 表。资产位于 `BASE_MODEL/xgen-adaln/fl2va` 或 `ref2va`。

```bash
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh cuda r2va 15 768p inputs/ref01.png
```

`cuda-5090` 与 `cuda` 的 Base 768P 默认相同。每条命令的时长可选 5、10、15；`OUTPUT_PATH` 覆盖输出路径。依赖、精度和 owner 回退见 [CUDA](H3-CUDA.md)。

<a id="npu"></a>
## NPU 768P

先按 [NPU](H3-NPU-DEFAULT.md) 安装 VC 和融合算子，并准备 NPU 的配对资产。推荐启动环境如下；离线脚本为未设置的变量补默认值，在线服务应显式配置。

```bash
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false OMP_NUM_THREADS=8 HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True INF_NAN_MODE_FORCE_DISABLE=1
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh npu r2va 15 768p inputs/ref01.png
```

默认使用 VC、Ascend 融合、INT8 DiT、通用 full-parallel 和 native-stream。T2VA/I2VA 使用 `model-cache/base-t2va` 与 `t2va-21step.safetensors`，Ref2VA 使用对应 `base-ref2va` 与 `ref2va-21step.safetensors`；`BASE_TRANSFORMER`、`BASE_ADALN` 可指定其他配对路径。

<a id="references"></a>
## 首尾帧与有序参考输入

首尾帧沿用 I2VA 资产，在额外参数中传 `--keyframe_last_image`：

```bash
bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p inputs/first-frame.png -- \
  --keyframe_last_image inputs/last-frame.png
```

Ref2VA 多参考文件按需要的素材顺序追加，之后再使用 `--` 分隔额外参数：

```bash
bash example/diffusers/scripts/run_h3.sh cuda r2va 5 768p \
  inputs/ref01.png inputs/clip.mp4 inputs/audio.wav -- \
  --h3_reference_image_resize_mode match
```

参考媒体支持最多 9 图、3 视频、3 音频，总数最多 12；媒体时长与音轨规则由参数检查执行。图片、视频和独立音频的次序及视频是否含音轨都会影响条件。参考图分辨率不同于输出分辨率，缩放规则见 [配置](H3-CONFIGURATION.md#reference-images)。

<a id="refiner"></a>
## 2K Base → Refiner

2K 是独立的 Base、latent upscaler、Refiner 链路，不由 768P 优化开关自动开启。先完成 [Refiner 资产准备](H3-MODELS.md#refiner)。

```bash
export REFINER_WEIGHTS=weights/refiner
export REFINER_PREPARED=model-cache/refiner
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh npu t2va 5 2k
bash example/diffusers/scripts/run_h3.sh npu i2va 10 2k inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k inputs/ref01.png
```

NPU Base 使用 20 次计算及对应 21 点表，Refiner 固定另做 3 步。输出为 2560×1440、24 fps，5/10/15 秒分别交付 120/240/360 帧。

CUDA 2K 保留实验性 I2VA 路线：另需 packed NVFP4 FL2VA Base、兼容算子和 CUDA 准备的 Refiner。不能直接套用 CUDA 768P 表资产；配置入口仍为 `run_h3.sh cuda i2va ... 2k`。Refiner 参数和固定工作流约束见 [配置](H3-CONFIGURATION.md#refiner)。

<a id="lifecycle"></a>
## 复用与输出检查

命令行每次启动一组进程；应用需常驻时使用原 InferenceManager 或 [HTTP 服务](H3-ONLINE.md)。同一模型实例可更换 prompt、素材和支持的时长，启动精度、并行布局和固定资产的变化需要重启。

视频和音频使用原生时序对齐，保存帧数按具体入口裁剪；例如 15 秒的原生生成可为 362 帧，在线交付为 360 帧。不要对所有入口使用同一帧数断言。

输出完成后检查 MP4、完整视频/音频解码和实际帧数。文件可解码、画面质量和音频质量分别评估；初始化、常驻生成及 HTTP 下载也分别计时。

# H3 统一配置与组件边界

[文档目录](README.md) · [架构](ARCHITECTURE.md#configuration-flow) · [CUDA](H3-CUDA.md) · [NPU](H3-NPU-DEFAULT.md)

参数定义在 `adaptor/config.py`，默认值、组合与资产匹配在 `adaptor/validation.py`，加载装配在 `adaptor/load_pipe.py`。本页集中解释配置规则，不重复各入口的完整启动命令。完整参数以对应入口的 `--help` 和源码定义为准。

<a id="priority"></a>
## 配置优先级和生命周期

- 预设补齐未指定的选项，显式正/负参数优先；JSON 布尔值使用 `true/false`，不要使用字符串。
- 服务的显式 CLI 参数覆盖同名启动 JSON。配置文件路径由部署方填写，shell 的资产变量不会自动替换 JSON。
- 精度、并行、组件后端、固定模型和 LoRA 是启动配置，更改后重启。逐请求字段只处理 prompt、素材、支持的时长/尺寸和采样请求，不允许覆盖启动布局或权重路径。
- 不兼容组合在模型加载前或运行前明确拒绝；请求错误不自动重试、裁图、降分辨率或切换精度。

<a id="switches"></a>
## 通用组件开关

`--h3_features full-parallel` 是组件调度集合；CUDA 的 `sage-fp8-full-parallel` 预设还选择其硬件算子和精度。通用参数和兼容的 `h3_cuda_*` 拼写解析到同一能力，冲突别名会报错。

| 能力 | 通用/兼容参数 | 说明 |
|---|---|---|
| 本地输入与输出投影 | `--h3_local_input` / `--h3_cuda_local_input`；对应 `local_output` | 保留完整上下文；输出头精度不因收集顺序改变 |
| 请求内条件复用 | `--h3_precompute_conditioning` / `--h3_cuda_precompute_conditioning` | 同一请求多步复用，结束或异常后释放 |
| CUDA DiT 扩展融合 | `--h3_cuda_dit_fusions` / `--no-h3_cuda_dit_fusions` | H3 Base FP8 原生 SP 的启动开关。5090 Sage-FP8 全并行默认开启；兼容的形状和设备见 [CUDA](H3-CUDA.md#dit-fusions) |
| 视觉 SP | `--h3_vision_sp` / `--h3_cuda_vision_sp` | 保留每图边界和 DeepStack |
| VAE 编码并行 | `--h3_vae_encoder_parallel` / `--h3_cuda_vae_encoder_parallel` | 按原 tile 规则分派，支持形状外使用原编码 |
| 音频编码并行 | `--h3_audio_encoder_parallel` / `--h3_cuda_audio_encoder_parallel` | 遵守卷积 halo 和因果边界 |
| 长音频解码 | `--h3_audio_decoder_parallel` / `--h3_cuda_audio_decoder_parallel` | `none/long`，仅在支持的长形状生效 |
| DiT 驻留 | `--h3_cuda_dit_residency resident/conditioning/reference` | 后两项在相应阶段暂存 CPU，增加搬运 |
| FFN 分块 | `--h3_cuda_refiner_ff_chunk_size` | 0 关闭；非零为 128–32768 间的 128 倍数 |

通用 full-parallel 不自动改变 DiT TP/CP、输出尺寸、主干量化、attention 或 VAE decoder。NPU 和 CUDA 的推荐入口各自选择这些设置，不能由一个功能列表推断所有精度。

<a id="input-refiner"></a>
## 输入 Token Refiner

输入 Token Refiner 处理条件序列，与 2K 超分 Refiner 是不同模块。其 SP、attention 和 FFN 分别控制：

| 轴 | 参数 |
|---|---|
| 序列并行 | `--h3_input_refiner_sp` / `--h3_cuda_input_refiner_sp` |
| Attention | `--h3_input_refiner_attention` / `--h3_cuda_input_refiner_attention` |
| FFN 精度 | `--h3_input_refiner_ffn bf16/fp8` / 兼容 CUDA 拼写 |

CUDA 可选 cuDNN attention 和 FP8 FFN；NPU 使用其兼容 attention 与 BF16 FFN。旧 `--h3_cuda_input_refiner none/cudnn-sp/cudnn-sp-fp8` 是组合简写，显式轴设置优先。FFN chunk 只分块逐 token 前馈计算，不缩短 attention 上下文。

<a id="parallel"></a>
## DiT 并行布局

| `h3_parallel_mode` | 布局 |
|---|---|
| `sp` | 原生序列并行，默认布局 |
| `tp_cp` | 常规 TP×CP；CP 使用 Ulysses Token/Context 轴，不是 CFG batch 轴 |
| `tp_sp` | SP8 中的 TP2 线性桥接；不是额外 TP2×SP8 的 16 卡网格 |
| `tp_cp_shared` | 同组八卡线性 TP8 / attention CP8，卡数不按 8×8 相乘 |

`--h3_dit_tp` 选择分片数，`--h3_tp_comm bf16/fp8` 与 `--h3_cp_comm auto/bf16/fp8` 选择传输。共享布局要求两种传输精度一致，不能叠加独立 PHAA FP8 打包。FP8 包大小使用 `--h3_tp_comm_chunk_elements`，默认 262144。

旧 `--h3_cuda_linear_tp 2` 解析为 `tp_sp`。`--h3_parallel_warmup_mib` 默认为 0，非零资源预热用于支持的实验布局并计入初始化。这些实验布局不由默认 SP8 预设开启；owner/table、驻留、任务和设备限制由参数检查共同确认。

<a id="phaa"></a>
## CUDA PHAA 与包对齐

`--h3_cuda_phaa_fp8` 控制 FP8 传输；`--h3_cuda_phaa_direct_pack` 控制直接打包。直接打包移除中间连续副本，不移除原 Q/K/V、权重或发送/接收缓冲，也不等于零拷贝。

默认 CUDA 全并行预设为同步 FP8 peer 包选择 Align16。设置 `XGEN_PHAA_FP8_ALIGN16=0` 并重启可单独关闭；所有 rank 必须一致。NPU、BF16 传输和独立异步 FP8 实现不使用此布局。`fp8-bf16` 参考配方保持 BF16 通信，不应额外开启 PHAA FP8。

<a id="adaln"></a>
## AdaLN 表和 owner

| 参数 | 含义 |
|---|---|
| `--h3_cuda_adaln_mode auto` | CUDA 全并行预设选择配对表；其他配方按自身规则解析 |
| `--h3_cuda_adaln_mode table` | 显式使用表 |
| `--h3_cuda_adaln_mode owner` | 原始权重实时计算，不使用 stripped checkpoint |
| `--h3_cuda_adaln_dir` | 含 `transformer/` 和 `tables/` 的 CUDA 资产目录 |
| `--transformer_path` + `--adaln_cache_file` | 显式固定配对，不同时使用资产目录模式 |

权重、工作流、步数、scheduler shift 与相关 LoRA 必须匹配。CUDA 目录模式按请求的实际步数和条件选择对应表；固定表只接受其执行次数，路径不能成为任意请求字段。表模式限定原生 SP；实验 TP/CP 使用其要求的 owner 路径。

局部查表的数值一致性不代表整个生成链路逐位一致。准备工具和路径集中在 [模型资产](H3-MODELS.md#adaln-cuda)。

<a id="vae"></a>
## 视频和音频 VAE

`--h3_vae_backend legacy/lightning/native-stream` 选择视频后端。CUDA 推荐 FP8 和 NPU Base 推荐入口默认 native-stream；其他低层入口按自身配置，显式设置优先。

Native-stream 要求 `--h3_vae_model_path` 指向可信的原生 `video_vae`，并显式设置 `--h3_vae_trust_remote_code`、`--vae_matmul_quant none`。shell 对应 `H3_VAE_MODEL_PATH`、`H3_VAE_TRUST_REMOTE_CODE` 和可选 `H3_VAE_BACKEND`。

原生 decoder 管理 tile、时序边界、RGB8 与异步输出，不叠加第二套旧 VAE SP 调度。切换 legacy/lightning 时同步移除原生路径/授权并按对应量化参数重启。`--h3_cuda_vae_sp` 只用于其支持的 CUDA 旧解码路径，不是 NPU native-stream 的开关。

音频 VAE 始终按 FP32 加载和计算，并关闭外层 autocast。视频分块、DiT 量化和参考素材规模不会自动降低音频精度。

<a id="compiler"></a>
## 可选 VAE Torch Compiler

`--h3_vae_torch_compile` 默认关闭，仅支持 CUDA native-stream 的 decoder tile 子图。DiT、Sage、PHAA、文本 TP 和请求收集流程不编译。NPU、legacy/lightning 会拒绝此选项。

初次编译和新 shape 的编译成本应与常驻执行分开。服务启动 JSON 或 CLI 可设置，变更后重启；默认关闭时保持原执行路径。

<a id="reference-images"></a>
## Ref2VA 参考图片缩放

`--h3_reference_image_resize_mode` 与 `--h3_reference_image_short_edge` 二选一，仅用于支持的 Ref2VA Base。两种编码器读取同一归一化图片，缩放保持比例并对齐到 32 倍数。

| 模式 | 行为 |
|---|---|
| `diffusers` | 短边缩到/放大到 2048 |
| `max` | 只在短边超过 2048 时缩小 |
| `match` | 只在面积超过输出画布时缩小 |

Ref2VA Base HTTP 未设置两项时使用 `max`；离线两项都未设置时保留 checkpoint 规则。固定短边是 32–2048 的整数、32 的倍数，改变参考细节，不是等价数值优化。

这些是启动选项，不是生成请求字段。T2VA/I2VA 和 Base+2K Refiner 不接受当前显式参考图尺寸覆盖。

<a id="condition-overlap"></a>
## 条件准备重叠

`--h3_condition_overlap` 默认关闭，可让参考 VAE 编码与 Qwen 文本/视觉准备重叠。仅适用于 CUDA/NPU 的 H3 Base Ref2VA 原生 SP，排除 Image、T2VA/I2VA、2K 协调器和 TP/CP 实验布局。

两支条件完成后才进入 DiT，不增加采样、不裁图、不改现有精度/offload。`--h3_condition_overlap_timeout` 默认 600 秒，只约束条件等待，不是总请求超时。重叠可能增加同时驻留的条件和权重峰值。

离线追加 `-- --h3_condition_overlap`；服务在启动 CLI 或 JSON 设置。JSON 已开启时，显式 `--no-h3_condition_overlap` 才覆盖关闭，所有预设都不会自动启用它。

<a id="refiner"></a>
## 2K Refiner

`--h3_refiner` 默认关闭。`--h3_refiner_weights` 指向源包，`--h3_refiner_prepared` 指向已合并权重、表和 manifest；`--h3_refiner_seconds` 接受 5/10/15，`--h3_refiner_loading prefetch/sync` 选择权重加载方式。

NPU 支持各自配对的 T2VA、I2VA、Ref2VA；CUDA 2K 保留独立 I2VA packed NVFP4 路线。一个实例固定工作流、后端和模型资产。初始化失败后需重启，不复用半初始化状态。

应用编排在 adaptor，Base → Refiner 的模型计算顺序在 Pipeline；不开启时不加载 Refiner。输入 Token Refiner 的 chunk 或 SP 开关不会启用 2K。启动和输出规格见 [视频](H3-VIDEO-QUICKSTART.md#refiner)。

<a id="attention"></a>
## NPU 可选 Base attention

`--h3_attention legacy/spark/veda` 选择 Base 局部适配器，默认 `legacy` 保留普通 `--atten` 路径。Spark/Veda 需要兼容的 Ascend Triton、CANN、CMake 和固定的可选源码：

```bash
git clone https://github.com/Huage001/Ascend-H3.git optional/Ascend-H3
git -C optional/Ascend-H3 checkout e02a57325029e67eafaf67d64b3d62a330f1e8d4
export PYTHONPATH="$PWD/optional/Ascend-H3/src:$PWD/x_diffusers:$PWD/x_base${PYTHONPATH:+:$PYTHONPATH}"
export H3_ASCEND_CACHE="$PWD/model-cache/h3-attention-native"
python -c "from h3_ascend.build import build; build('spark',soc='Ascend910B1'); build('veda',soc='Ascend910B1')"
```

先加载实际部署环境的 CANN，按真实 SoC 调整构建参数，保留源码及内核许可。构建产物依赖 ABI，不跨环境直接复用。

Spark 需要在启动 Python 前显式选择兼容的 `TASK_QUEUE_ENABLE=1` 或 `0`，不兼容模式会在加载前拒绝。Veda 需要 `--h3_veda_predictor`；离线 `--h3_veda_sparsity` 接受 0.7 或 0.9，具体任务、时长、分辨率和组合由 validation 限定，不能任意替换默认 attention。

适配器只改变受支持的 Base attention，不自动改变 Refiner、VAE、LoRA 和采样。算子/预测器配置改变或执行失败后重启，回退使用 `--h3_attention legacy` 并恢复相应队列环境。

<a id="turbo"></a>
## Turbo 与 LoRA

Turbo8 要求预合并的 LightX2V Transformer、匹配 AdaLN 表和 `--num_inference_steps 8`；当前受限于支持的 NPU 5 秒横屏 T2VA 组合，具体 attention 和 Refiner 组合单独校验。准备命令见 [可选资产](H3-MODELS.md#optional-assets)。移除开关时同时恢复原 Transformer、表和步数。

`--h3_turbo_sla_lora` 使用模型对应的 LoRA 与训练 Euler 调度；其步数和 scheduler 契约由专门校验处理，不能套用普通表文件名来推断。LoRA 不允许在常驻请求中悄悄换用；Turbo8、其他 LoRA、缓存、packed 和精度组合的限制以 validation 为准。

<a id="observability"></a>
## 观测与故障处理

检查实际启用能力、fallback、条件与组件计数、采样步数及设备状态。H3 请求结束后释放临时输出并回收空闲 allocator 缓存，`H3_REQUEST_MEMORY_RELEASE` 记录同步/回收耗时和前后 allocated/reserved；它不会卸载仍被引用的常驻权重，也不能消除请求内部的所有 OOM。

功能检查应覆盖输出文件、完整解码、输入顺序、时序边界和异常后状态。设备容量、生成内容质量和服务端到端表现需要各自验证，不能从开关名称或局部算子通过推断。

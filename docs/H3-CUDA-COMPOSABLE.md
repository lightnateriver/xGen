# RTX 5090：Sage-FP8 全并行 Streaming

v0.5 八卡 CUDA H3 Base 默认采用 `sage-fp8-full-parallel`、`native-stream` 和 BF16 AdaLN 表，覆盖 T2VA、I2VA／首尾帧、R2VA。复用原 CLI、InferenceManager、HTTP 服务及同一个 H3 DiT 循环。NPU 保留 v0.5 的 Laser INT8 + Stream 配方；其他模型、2K、NVFP4 和显式旧精度配方不自动套用。

先完成 [AdaLN 资产准备](H3-CUDA-ADALN.md)。表替代对应的时间编码／投影权重，主 DiT 全部常驻；每项组件优化仍可单独关闭。最大输入限制不等于显存容量保证，技术生成也不等于人工精度认可。

## 最简单的启动方式

先按根 README 准备 Ref2VA 完整模型、对应 AdaLN 表、参考图和 H3 prompt。路径由环境变量指定，下面的路径相对于仓库根目录。

```bash
export BASE_MODEL=weights/h3-ref2va
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/h3-native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png
```

五图输入直接追加五个文件，不改变输入顺序。10 / 15 秒把 `5` 改为 `10` / `15`。初始化只计入模型启动；常驻 Manager 连续请求可以切换时长和参考图，启动配置不能在复用模型时悄悄改变。

新预设默认使用：主干 SP8、Sage attention、FP8 W8A8 Linear、文本 FP8 TP8、PHAA2 双向 FP8、直接 stride 打包、SAP 本地 FP32 输出、本地输入投影、请求内固定条件复用、Token Refiner SP8/cuDNN BF16/FP8 FFN、视觉 SP8、VAE Encoder tile 分工、Audio Encoder 卷积分工和长音频 Decoder 分工。DiT 全部常驻 GPU，包括条件编码和 VAE 阶段。视频 decoder 默认改为 `native-stream`：原生 decoder、FP16 autocast、全 rank tile 分派、分块 RGB8 和异步 D2H/编码；输出头及音频计算的既有精度不变。Audio Decoder 的 `long` 在 5 秒形状走原实现；Audio Encoder 在没有参考音频时不会执行。

`video_vae` 是独立的原生模型目录，不是 Diffusers `vae` 子目录。授权变量只应用于可信模型代码。新后端初始化更慢，CPU master 和显存代价见 [VAE 教程](H3-OMNI-VAE.md)。[最新 Streaming 矩阵](performance/h3-cuda-streaming-matrix.md) 区分 owner 与 table；历史 legacy decoder 表单独保留。

CUDA 推荐配方仅把 native decoder 的 Linear 预存为 FP16，其他参数保持 FP32，计算仍是 FP16 autocast；复用旧解码器同一策略以减少静态占用。解码前释放请求内条件结果并回收空闲缓存，DiT 仍全常驻。NPU 沿用 v0.5 原配方；原生 FP32 组件测量路径不变。

## 覆盖单项配置

`--` 后传原 Python 参数。预设只填未指定的默认值，显式的正／负参数优先。例如保留预设其余能力，关闭本地输入投影，并把输入 Refiner FFN 改回 BF16：

```bash
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png -- \
  --no-h3_cuda_local_input --h3_cuda_input_refiner_ffn bf16
```

关闭整个新增预设、使用原 Sage-FP8 配方：

```bash
H3_CUDA_PRESET=none H3_CUDA_PRECISION=fp8-sage bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png
```

`cuda` 与 `cuda-5090` 的 Base 768P 默认相同。关闭新预设不代表关闭原先的 FP8 Linear 或 Sage；精度选择仍由 `H3_CUDA_PRECISION`／`--h3_cuda_precision` 控制。

## 独立控制项

表中“原路径”指未选择新预设、未显式开启新功能。旧 TP/CP 实验布局保留原先本地输入／输出的默认值；可以显式关闭它们。所有布尔参数均支持对应的 `--no-名称`，JSON 则使用真正的 `false`，不要写字符串 `"false"`。

| 能力 | 参数 | 原路径默认 | 全并行预设 | 关闭后的行为／限制 |
|---|---|---|---|---|
| AdaLN 资产 | `--h3_cuda_adaln_mode auto\|table\|owner` | owner | table | owner 使用原始权重，需重启；[准备与匹配](H3-CUDA-ADALN.md) |
| 视频解码后端 | `--h3_vae_backend legacy\|lightning\|native-stream` | CUDA 推荐 FP8 配方为 native-stream；其他入口见 VAE 教程 | native-stream | 可显式切换；legacy/lightning 需取消原生模型路径及授权 |
| 本地输入投影 | `--h3_cuda_local_input` | SP 下关 | 开 | 全模态投影后再分片；文本上下文不会被截短 |
| SAP 本地 FP32 输出 | `--h3_cuda_local_output` | SP 下关 | 开 | 先 gather 完整 hidden，再执行输出头 |
| 请求内条件预计算 | `--h3_cuda_precompute_conditioning` | 关 | 开 | 每步计算；请求结束／异常或条件改变即释放／失效 |
| 输入 Token Refiner SP | `--h3_cuda_input_refiner_sp` | 关 | 开 | 完整上下文单卡副本计算 |
| 输入 Refiner Attention | `--h3_cuda_input_refiner_attention legacy\|cudnn` | legacy | cudnn | BF16；与主 DiT Sage 后端独立 |
| 输入 Refiner FFN | `--h3_cuda_input_refiner_ffn bf16\|fp8` | bf16 | fp8 | 不改变 Refiner Attention 的 BF16 输入输出 |
| Qwen 视觉塔 SP8 | `--h3_cuda_vision_sp` | 关 | 开 | 原视觉 forward；保留每图边界及 DeepStack |
| VAE Encoder tile 并行 | `--h3_cuda_vae_encoder_parallel` | 关 | 开 | 原 encode；小／不齐 tile 记录原生回退 |
| Audio Encoder 并行 | `--h3_cuda_audio_encoder_parallel` | 关 | 开 | 原 encode；halo 和完整因果投影随适配器生效 |
| Audio Decoder 并行 | `--h3_cuda_audio_decoder_parallel none\|long` | none | long | none 为原 decode；long 仅在支持的长音频形状生效 |
| FP8 PHAA 传输 | `--h3_cuda_phaa_fp8` | 继承原环境 | 开 | BF16 传输；`fp8-bf16` 主配方要求此项关 |
| 去 BF16 临时连续副本 | `--h3_cuda_phaa_direct_pack` | 继承原环境，通常关 | 开 | 原 contiguous 后再打包；需要 FP8 PHAA，同组 TP8/CP8 不适用 |
| DiT 驻留 | `--h3_cuda_dit_residency resident\|conditioning\|reference` | 继承原暂存开关 | resident | 后两项在条件／参考阶段暂存 CPU，有 D2H/H2D 时间代价 |
| 输入 Refiner FFN 分块 | `--h3_cuda_refiner_ff_chunk_size` | 0 | 0 | 0 关闭；128..32768，128 的倍数；不是 2K Refiner |
| 通信资源预热 | `--h3_parallel_warmup_mib` | 0 | 0 | 0 关闭；适用实验 TP/CP 布局，计入初始化 |

旧 `--h3_cuda_input_refiner none|cudnn-sp|cudnn-sp-fp8` 保留为三轴的简写；显式轴参数优先。旧 `conditioning_offload`／`reference_offload` 保留，和显式驻留策略冲突时会报错。

直接打包的“去副本”仅移除中间 BF16 `.contiguous()` 张量。原始 Q/K/V、权重、CPU master、FP8 send/recv、接收端 BF16 张量仍存在。它与 DiT 驻留完全独立，不等于零拷贝通信。大 shape 单层历史微基准约省 558 MiB/卡、整层约快 0.4%；不能外推为所有整网输入都有相同收益。

## 并行布局和组合校验

`--h3_parallel_mode sp` 默认不变。`tp_cp` 为常规 TP×CP；`tp_cp_shared` 为同八卡线性 TP8／attention CP8；`tp_sp` 为四组 TP2 线性桥接原 SP8 attention。后两种不能按 TP×CP 乘积计算卡数。TP2 的旧参数 `--h3_cuda_linear_tp 2` 解析到 `tp_sp`，避免维护第二套配置。

TP／CP 传输由 `--h3_tp_comm bf16|fp8` 和 `--h3_cp_comm auto|bf16|fp8` 控制；FP8 会带来数值误差，不宣称等价于 BF16。TP/CP 与 TP2-SP8 仍为 experimental，完整成片精度和速度按具体测试记录判断，不包含在全并行 SP8 的默认选择中。共享 TP8/CP8 的融合重排要求两种传输精度一致，不能叠加独立 PHAA 打包选项。FP8 TP 分包大小使用 `--h3_tp_comm_chunk_elements`，默认 262144。

配置在模型加载前检查：硬件、模型、world size、任务、精度、布局、驻留、FP8 打包依赖以及已有 packed/offload/LoRA 配方冲突。无效组合明确拒绝。OOM 不会自动裁图、降分辨率、换精度或增添 offload。全程常驻使用更多静态显存，可显式选 conditioning/reference 暂存；不会自动降级。

## 在线服务

原 HTTP 服务支持同一套启动参数和 JSON 字段。配置、路由、队列和请求作用域已有CPU回归，真实八卡HTTP连续请求仍待部署验证，不能沿用离线成片结论。由启动配置决定开关，生成请求不能覆盖权重／并行布局／精度；如需改变，重启 worker。MaaS 复用既有 R2VA 协议和输入限制，独立部署对应工作流。

修改仓库提供的 [在线示例 JSON](../example/diffusers/configs/h3_cuda5090_ref2va.example.json) 中的模型目录。示例已配置 Ref2VA、SP8、文本 TP8、全并行预设、DiT 常驻和20步；无需手动拼接优化开关。启动：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 server/video/video_server.py \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json --init \
  --service-mode ref2va --host 127.0.0.1 --port 5001
```

独立开关也可写进同一 JSON。显式 CLI 参数覆盖 JSON；JSON 中的 false 必须转成对应关闭参数。新增选项属于 worker 启动状态，在线／离线常驻 Manager 都检查同一个配置 key，时长、prompt 和参考素材可以按请求更换。日志打印最终有效选项，既有组件 metrics 记录计算／复用次数、通信精度、分工和 fallback。参数解析失败或常驻配置冲突不会先修改运行中进程的通信环境。

## 验证范围

CPU 配置契约、八 rank Gloo 输入／输出四组合和 BF16／FP8 桥接协议测试，与真实八卡成片测试分开。FP8 桥接协议逐字节匹配量化 oracle，只证明实现符合该协议，不能证明与未量化模型精度相同。旧源码 Git 历史不在源码包时，AST 历史比较会跳过，不能计作通过。NPU／Wan 本轮主要做默认配置和分派回归，不把 CUDA 实测当成它们的成片验证。

五项真实20步结果及追加兼容性测试见 [公开性能与验证记录](performance/h3-cuda5090-full-parallel.md)。初始化、预热、正式请求、技术解码和人工画面／音频复核分别记录。五图15秒峰值allocated约28.33GiB/卡、reserved约30.39GiB/卡，接近32GiB设备容量；这不是9图或音视频满载的容量承诺。CPU master及VAE staging仍占用主机内存；全常驻不会消除临时张量与通信缓冲。

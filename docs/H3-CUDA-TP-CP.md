# CUDA H3 TP + CP（experimental）

默认仍是原 SP。新路线使用同一套 H3 模型、PHAA、Attention、VAE 和推理管理器，只替换 DiT 的权重布局以及输入/输出边界。不安装 vLLM。

## 参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--h3_parallel_mode` | `sp` | `tp_cp` 显式启用备选链路 |
| `--h3_dit_tp` | `1` | 新路线接受 1（边界对照）、2 或 4；不控制文本 TP。TP4 完成 CPU 分布式检查，CUDA 成片待验证 |
| `--sp` | 保持原默认 | 新路线的 Ulysses CP 大小；八卡 TP2 时设 4 |
| `--h3_tp_comm` | `bf16` | 行并行 partial 的 BF16 归约或分块 FP8 交换、FP32 求和 |
| `--h3_cp_comm` | `auto` | PHAA 去程 QKV 和回程输出：`bf16` / `fp8`；auto 继承 `XGEN_PHAA_FP8` |
| `--h3_tp_comm_chunk_elements` | `262144` | 多卡 FP8 TP 每包元素数；128–4194304 且为 128 的倍数。较大包减少调用数、增加临时显存，需实测选择 |

未选择 `tp_cp` 时，非默认的新参数会报错，避免被静默忽略。显式 CP dtype 只覆盖新模式，旧 `XGEN_PHAA_FP8=1` 双向传输语义不变。

## 八卡离线配置

可直接使用独立启动脚本，原脚本不变。例如一张参考图的 5 秒 R2VA：

```bash
export MODEL_PATH=./models/h3-ref2va
export PROMPT_FILE=./inputs/prompt.txt
TASK_TYPE=ref2va bash example/diffusers/scripts/infer_minimax_h3_cuda_tp_cp.sh --reference_paths ./inputs/ref1.png
```

T2VA 省略 `TASK_TYPE` 和参考图参数；I2VA 使用 `TASK_TYPE=fl2va FIRST_FRAME=./inputs/first-frame.png`。通过 `SECONDS_VIDEO=10` 或 `15` 选择时长，通过 `TP_COMM` / `CP_COMM` 选择传输精度。默认 20 步、TP BF16、CP FP8、FP8 Linear + Sage。当前是待完整模型验证的 experimental 配方。

使用原有 [CUDA 启动步骤](H3-CUDA-SAGE.md)，在原 `torchrun` 命令中追加以下参数，并将原 `--sp 8` 替换为 `--sp 4`：

```bash
--h3_parallel_mode tp_cp \
--h3_dit_tp 2 \
--sp 4 \
--h3_tp_comm bf16 \
--h3_cp_comm fp8 \
--h3_cuda_precision fp8-sage \
--text_encoder_tp 8 \
--phaa_num 2 \
--h3_cuda_vae_sp
```

主模型仍为 W8A8 FP8 Linear，Sage 为 Q/K INT8、P×V FP8，Linear 和 Attention 接口输出 BF16。TP/CP 的通信 dtype 不会改变权重精度。也可显式选择 `fp8-bf16` 做 BF16 Attention 对照；新模式中它的通信 dtype 由新参数决定。

当前完整模型配置要求 world=8、文本 TP8、DiT TP×CP=8、CFG=DP=Ring=1；VAE 仍 SP8，DiT 常驻策略不变。每个工作流保持各自 FL2VA / Ref2VA 原始模块权重。不支持 Refiner/2K、NVFP4、LoRA、外部 AdaLN 表、FSDP 或其他 offload/cache 配方。不会因 OOM 自动切换配置，也不会缩小参考输入。

八卡可选择 TP2×CP4 或 TP4×CP2，不能把 TP4×CP4 当成八卡布局。TP4×CP2 的 CPU/Gloo 八进程检查覆盖原生 BF16 Linear、BF16/FP8 TP 通信、三种 packed 形状及 mask，最大相对 RMSE 约 0.254%；四个 TP peer 的预测一致、文本 TP8 保持独立。该检查没有运行 CUDA FP8 GEMM、Sage 或真实视频，不能替代完整模型验证。

## HTTP 配置

普通 Base 服务可以使用相同四个启动参数或在服务 JSON 中加入对应键。它们属于服务启动配置，不是请求字段；模型常驻后修改布局或传输精度必须重启。现有 MaaS 输入限制保持原样，本特性不新增 MaaS R2VA 或 2K 能力。尚不能用单层测试宣称 HTTP E2E 已通过。

离线常驻 manager 也在重新加载前拒绝布局、传输、精度、模型路径或 FL2VA/Ref2VA 工作流变化。T2VA/I2VA 可共用 FL2VA 模型并切换时长与参考输入；R2VA 使用独立 worker。这样不会在旧模型仍驻卡时意外加载另一套模型。

## 显存与延迟

- TP2 分片 QKV、输出投影、FFN gate/up/down 权重；CPU 完成分片后才转入设备。不先在 GPU 构建完整权重。
- 视频和音频只投影本 CP rank 的 Token。文本 refiner 保留完整文本上下文，文本编码器视觉分支不因此自动分片。
- 主干末端先做本地 FP32 输出头，再聚合低维预测，不聚合完整 hidden。FP32 输出精度保留。
- BF16 TP 增加每块两次行归约。FP8 TP 还增加动态 scale、打包、交换和 FP32 求和，短序列可能更慢；不能根据字节减少推导 E2E 提升。
- FP8 TP 默认每次最多处理 262144 个元素，可通过 `--h3_tp_comm_chunk_elements` 调整，仍避免保存各 rank 的完整 FP32 副本；原 partial、结果和通信缓冲仍需同时占显存。非默认包大小只适用于多卡 FP8 TP，常驻进程修改需重启。
- FP8 CP 复用 PHAA 的逐 Token/head E4M3 与 FP32 scale。发送、接收和 BF16 缓冲同时存在，峰值不会自动减半。

## 最小验证

不下载模型即可运行真实 H3 单层。CPU 测试覆盖 BF16 切片和 FP8 wire 重构；CUDA 测试再覆盖真实 FP8 GEMM、PHAA 和 Sage：

```bash
python tools/test_h3_parallel_config.py
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=4 tools/validate_h3_tp_cp.py --device cpu --cp 2 --output outputs/tp-cp-cpu
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=2 tools/validate_h3_tp_cp.py --fp8-linear --sage --output outputs/tp-cuda
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=2 tools/validate_h3_tp_cp.py --cp 2 --fp8-linear --sage --output outputs/cp-cuda
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=4 tools/validate_h3_tp_cp.py --cp 2 --fp8-linear --sage --production-adapter --output outputs/tp-cp-cuda
```

仅选择自己已预留的可见设备。输出每 rank JSON，包括误差、形状、FP8 分块统计和显存峰值。合成单层通过不等于完整权重成片、满载容量或人工音画精度通过。完整模型需同输入、seed、预热、20 实际步数与 SP8 对照。

本轮已完成的单层条件、四种传输误差和测试范围见 [最小验证记录](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda-tp-cp.md)。

实际层宽的 SP4 / TP2×CP2 预热后延迟与通信剖析见 [单层性能对比](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda-tp-cp-latency.md)。本轮 SP4 更快；当前 FP8 TP 的分块 all-gather 开销较高，仍需优化。可在四张已预留的可见 GPU 上复现：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=4 tools/benchmark_h3_parallel_block.py --output outputs/h3-parallel-block-20261106
```

## 观测与回退

### 原 SP8 的可选输出内存优化

CUDA FP8 profile 的原 SP8 路线可显式追加 `--h3_cuda_local_output`。它复用本页 TP+CP 的 local output helper：各 SP 分片完成原输出归一化和 FP32 视频/音频输出头，再收集低维预测，避免先收集完整 hidden 并同时转换为 FP32。DiT block、PHAA、参考输入和输出头精度保持不变。

默认关闭，NPU 和未选择 CUDA FP8 profile 的路径不启用；模型常驻后改变此开关需要重启。TP+CP 已自动使用本地输出，无需重复指定。关闭此参数并重启可恢复原输出路线。该选项仅解决输出阶段的张量峰值，不保证视觉条件编码等其他阶段不会 OOM。

八卡 RTX 5090 输出边界数值测试覆盖整除、尾部 padding、Token 少于卡数、tuple/dataclass 返回六种组合，每 rank 六项；FP32 输出最大绝对误差 `7.16e-7` 以下。测试使用合成 hidden 和小型输出头，不代表完整视频精度通过。可运行：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=8 tools/validate_h3_sp_local_output.py --device cuda --output outputs/sp-local-output
```

完整 R2VA 5/10/15 秒、1/5 图/Max 的输出修复验证仍在进行，尚无此修复的完整 20 步性能或画面精度结论。

### Ref2VA 条件编码的可选阶段卸载

显式 CUDA FP8 Ref2VA profile 可追加 `--h3_cuda_conditioning_offload`，默认关闭。它在原 Qwen3-VL 模型调用边界暂时将同一个 DiT 移到 CPU，条件编码成功后以原 dtype 搬回设备；不重读 checkpoint、不重新量化、不缩小参考图或删减 Token。DiT 推理和 VAE 解码期间仍保持原常驻策略。

该开关适用于多图/多模态条件编码与常驻 DiT 同时占显存的场景；代价是额外 CPU 暂存和每请求 D2H/H2D 搬运。`pipe._h3_cuda_conditioning_staging` 给出调用次数、累计搬运秒数及最近阶段状态，启动日志记录实际显存和搬运时间。编码失败保留原异常，DiT 留在 CPU，由原 worker 生命周期处理；不会自动重试。服务常驻后修改开关须重启。

此选项不改变 NPU、T2VA/I2VA 或默认 CUDA 路径。显式 CUDA FP8 Ref2VA 同时复用现有参考音频的整模块 staging，以满足音频 weight-norm 的设备契约。完整多模态成片及搬运成本尚待本轮实测，不据此宣称所有 Max 形状已通过。

Ref2VA 可显式选择 `--h3_cuda_precision fp8-bf16` 做 BF16 cuDNN Attention / BF16 通信对照。它仍使用 FP8 Linear，不是全 BF16 权重基线；`auto` 下 Ref2VA 仍保留原默认路径。该对照的完整视频验证尚未完成，不继承 T2VA/I2VA 的精度结论。

### 输入 Token Refiner FFN 的可选分块

多模态参考在进入主干前仍经过完整上下文的输入 Token Refiner。显式 CUDA FP8 Ref2VA 可追加 `--h3_cuda_refiner_ff_chunk_size 8192`，复用 Diffusers 的逐 token FeedForward 分块范式，单独处理末尾不足一块的 token。不切 attention、不减少参考或 Token、不改变权重精度，也不释放常驻 DiT。

默认值 `0` 保留原 FFN 路径。非零值必须为 128–32768 之间的 128 倍数；常驻服务修改此值需重启。较小块降低门控/激活临时张量峰值，但增加 GEMM 调用，可能影响时延和末位浮点舍入。`transformer._h3_refiner_ff_chunking` 记录调用、完成块数及最大输入长度，不保存条件张量。该实验选项不影响 NPU、Wan 或默认推理。关闭参数并重启即可回退。

该选项针对输入辅助 Token Refiner，和 2K 超分 Refiner 是不同模块。数值与完整 Max 成片验证分别记录，不能用 CPU 单元测试代替视频精度结论。

### 参考视频 VAE 编码阶段的可选卸载

在上述条件卸载之外追加 `--h3_cuda_reference_offload`，可让 DiT 在 Qwen 条件编码后继续留在 CPU，直到参考图片、视频、音频编码和原生预处理结束，再于第一个 DiT forward 的原有 module pre-hook 边界恢复 GPU。仍是每请求一次 D2H/H2D，主干所有层在 DiT 及输出 VAE 解码期间继续常驻；不改变参考尺寸、帧数、编码算子或精度，也不缓存条件。

此开关默认关闭，要求显式 CUDA FP8 Ref2VA 与 `--h3_cuda_conditioning_offload`，常驻服务变更须重启。它使用原 VAE 编码及 offload 路径，解决常驻主干和长参考视频编码激活同时占显存的问题，不承诺加速参考编码。阶段指标的 `restore_boundary=before_first_dit` 表明 H2D 已移出 Qwen span，应结合独立 H2D 计时查看。失败保留原异常和 CPU 上的 DiT，由原 worker 生命周期处理。

启动日志显示 TP/CP 组、通信 dtype 和 VAE 组。每块行投影的 `reduce_fn.metrics` 记录 TP 调用、payload 和 host enqueue 区间；FP8 另含分块与最大单包。Attention 的 `_h3_cp_metrics` 记录 CP 调用及包含 self exchange 的逻辑 payload。这些字节数不是实际网卡流量，host 区间也不是同步 GPU 耗时。Profiler 范围 `h3.tp.bf16.all_reduce`、`h3.tp.fp8.pack` / `exchange` / `fp32_sum` 可结合 CUDA/NCCL trace 定位等待。

回退时删除新参数、恢复原 `--sp 8` 和原精度配置，重启 worker。默认 SP 的投影、循环、输出聚合、缓存与显存调度保持原路径。
# PHAA FP8 混合 stride 打包（experimental）

`XGEN_PHAA_FP8_MIXED_STRIDES=1` 是默认关闭的 CUDA 内存优化。PHAA 的
Q/K/V 布局不同时，直接从各自的五维视图量化打包，避免三个 BF16
`contiguous()` 临时副本；FP8 E4M3 数值、逐 head FP32 scale、通信字节布局和
接收还原过程保持不变。仅适用于同步 FP8 PHAA 的三路、末维连续视图，
其他布局及单路回程保留原逻辑。不改变 NPU、BF16 通信或异步 PHAA。

日志记录首次命中的 shape/strides；`mixed_stride_metrics()` 返回调用数和累计
避免的逻辑副本字节数（不是峰值显存或实测网络流量）。关闭环境变量即可回退。
CUDA 测试要求新旧传输字节与接收结果逐位一致，包含不同 stride、尾部 head、
多 batch 和不支持布局的回退。

## DiT 通信启动预热（experimental，默认关闭）

在 `--h3_parallel_mode tp_cp` 下可增加 `--h3_parallel_warmup_mib 512`。
默认 `0` 保留原行为；范围 0–1024 MiB，修改后须重启常驻 worker。
它在 CPU 权重转换及上 GPU 之前，用已有 DiT TP/CP group 执行对应 dtype 的
collective：FP8 TP 的 FP32 最大值归约及 uint8 All-Gather，或 BF16 TP 归约，
以及 CP All-to-All。TP1×CP8 也可使用，不改变默认 SP 路径或 NPU。

用途是把首次 NCCL 运行的显存需求提前到初始化。NCCL 内部分配不经过 PyTorch
缓存分配器；显存紧张时，即使 PyTorch 留有未使用缓存，NCCL 仍可能 OOM。
此开关不改变权重、attention 数值、请求输入或 DiT 循环，不添加条件缓存。
启动时每卡 CP 临时张量约 `2 × 配置 MiB`，之后释放给 PyTorch 缓存复用；
NCCL 自身的运行资源仍驻留。它不能保证任意输入的激活峰值都能装入显存。

启动日志及 `model._h3_parallel_warmup` 分别记录耗时、通信操作、TP/CP 大小、
CP 单个缓冲区字节数和前后 driver free。free 差还包含 PyTorch 缓存的临时张量，
不能当作纯 NCCL 占用。预热只执行一次，初始化耗时应单列，不计入常驻请求 E2E。
关闭参数并重启可回退；失败保留原异常，不自动重试。


# H3 同组八卡 TP8 / CP8（experimental）

该模式让同一组八卡按层切换分片：主干 QKV、O 和 FFN 权重走 TP8，层间残差按 CP8 保存序列片段。文本编码器仍用原 FP8 TP8；AdaLN 保留 rank-owner 和精确投影缓存，SAP 的最终预测头仍为 FP32。默认仍是原 SP 路径。

```text
CP8 残差：每卡 S/8 个 token
    │ gather 序列
    ▼
TP8 QKV：每卡完整序列的 1/8 attention heads
    │ 与 Ulysses 的内部布局相同，融合相邻逆向重排
    ▼
CP8 / Ulysses Attention：复用 Sage，7 heads 按 4+3 分段
    │ TP8 行投影 + reduce-scatter
    ▼
CP8 残差：每卡 S/8 个 token
    │ gather → TP8 FFN → reduce-scatter
    ▼
下一层 CP8 残差
```

CP 与 TP 进程组成员都为同一组八个 rank；分别使用现有 `ParallelManager` 创建，不修改二维 mesh 的乘积校验。它与原有 `tp_cp` 模式是不同的可选模式，不能把 TP2×CP4 标作 TP8/CP8。

## 参数

在原 H3 `fp8-sage` 启动命令上使用以下参数片段：

```text
--h3_parallel_mode tp_cp_shared
--h3_dit_tp 8
--sp 8
--text_encoder_tp 8
--h3_tp_comm fp8
--h3_cp_comm fp8
--h3_tp_comm_chunk_elements 4194304
--phaa_num 2
```

`bf16` 和 `fp8` 两种桥接传输均可选；本模式要求 TP／CP wire dtype 相同，`h3_cp_comm=auto` 会跟随 TP。FP8 复用现有每128元素的 E4M3 packet，接收端按 rank 顺序 FP32 求和；不会在量化字节上直接归约。packet size 控制行投影归约 scratch，不能改变已运行 worker 的分组或传输配置。

本模式自动使用局部输入和 SAP 输出，不传 `--h3_cuda_local_output`。Ref2VA 可保留以下全并行开关：

```text
--h3_cuda_conditioning_offload
--h3_cuda_input_refiner cudnn-sp-fp8
--h3_cuda_precompute_conditioning
--h3_cuda_vision_sp
--h3_cuda_vae_encoder_parallel
--h3_cuda_audio_encoder_parallel
--h3_cuda_audio_decoder_parallel long
--h3_cuda_vae_sp
```

以上是参数片段，模型路径、prompt、参考媒体、时长、输出路径沿用现有入口。关闭新模式并恢复原参数即可回到 SP。它不启用 NVFP4、稀疏 Attention、跨请求条件缓存或自动失败恢复。

## 边界与可观测性

- 仅 CUDA H3，固定同组8卡，文本TP8；原 NPU、Wan 和未开启此模式的路径不变。
- 线性权重分片降低静态显存，序列 gather 和行归约增加通信。速度与形状、PCIe 和传输精度有关，不能只凭权重变小宣称更快。
- CP Attention 的 QKV all-to-all 与 TP 头分片融合，实际通信变为序列 gather 和行投影 reduce-scatter。后者的 FP8 packet 通过 all-to-all 交换；它不是额外叠加一套二维 TP×CP。
- 延续既有 RMSNorm、partial RoPE、indexed affine、EXP、VAE、编码保存逻辑；没有另建模型调度或解码系统。
- `model._h3_shared_sequence_exchange.metrics` 提供实际成员、通信类型、gather／reduce-scatter／attention 调用数、packet 和逻辑字节数。profile range 为 `h3.shared_tp_cp.*`。
- FP8 gather 为一次完整上下文分配通信和反量化缓冲；长序列与多参考输入仍需监控动态峰值，不保证所有输入都能避免 OOM。
- CPU/Gloo 的原生 H3 block 缩小尺寸数学验证和相关默认路径回归已通过。下表六项真实 CUDA 正式请求也已通过技术检查；它们与完整人工画面／听音验收分别记录，不能互相代替。

CPU 数学 gate（不加载权重、不使用GPU）：

```bash
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc-per-node=8 tools/validate_h3_shared_parallel.py
```

参考了 vLLM-Omni 的 Column/Row TP 范式；其通用 TP/SP rank 网格仍按乘积创建，本模式的同组序列桥接为 xGen 显式新增实现：[上游 parallel_state](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/distributed/parallel_state.py)。

## 768P 验证记录（2026-11-08）

运行源码 `55495dcb6931d274b8b9c10ad0a4f0869391bca3`。8×RTX 5090，1344×768、24FPS、seed42，Ref2VA，20次实际DiT；每项先完整1步预热，再正式测一次，一组八卡进程连续复用模型。配置为本文的FP8桥接＋Sage全并行组合，50层DiT在DiT／VAE阶段保留设备权重。原始PNG均2560×1440，输入顺序、prompt和预处理与历史SP对照一致。

运行环境为既有H3适配的native runtime：Torch 2.12.0、Diffusers 0.40.0、Transformers 5.15.0、SageAttention 2.2.0、Triton 3.7.0。验证了8个关键依赖源码文件的哈希；版本号本身不能代替H3原生适配文件。

| 参考图 | 时长档位 | 原Sage-FP8 SP8 E2E s | 全并行SP8 E2E s | 同组TP8／CP8 E2E s | TP8／CP8峰值allocated GiB/卡 |
|---|---:|---:|---:|---:|---:|
| 1张 | 5秒 | 195.65 | 106.06 | 228.60 | 12.22 |
| 1张 | 10秒 | 249.85 | 133.82 | 284.94 | 12.27 |
| 1张 | 15秒 | OOM | 197.93 | 441.35 | 12.32 |
| 5张 | 5秒 | OOM | 199.91 | 376.32 | 12.54 |
| 5张 | 10秒 | OOM | OOM | 685.62 | 13.39 |
| 5张 | 15秒 | OOM | OOM | 923.58 | 14.82 |

E2E取八rank最慢生成到保存，不含初始化、媒体技术检查和传输；本轮初始化75.31秒单列。前两列为不同日期／部分不同物理节点的历史记录，不能直接当成严格同机配对的架构加速比。时长沿用既有H3帧对齐，三档实际输出124／243／361帧；没有为了对齐时长修改成片。

本轮六项均未OOM，八rank逐项核对实际20步、主干200个TP8 FP8投影、CP8成员、Sage真实内核、通信调用数及固定条件每请求计算一次／复用19次。视频轨、音频轨、完整技术解码、文件SHA及12帧抽查分别留证。两条15秒样例仍有画面连续性或动作遵循待复核点，完整画面、口型和人工听音均未自动标记通过。纯图片参考不会触发Audio Encoder，不能据此宣称参考音频端到端已验证。

本轮是容量收益：历史SP8曾OOM的五图10／15秒均已完成；已成功的历史SP8项耗时则更低，默认仍保留SP，不把共享TP8／CP8推荐为最快配置。新增序列gather、行归约、FP8打包与窄TP GEMM的具体占比需要单独profiling，不能仅凭计数给出算子耗时归因。

一组worker完成后已退出，实时八卡显存／利用率及计算进程均为零。既有被动采样的驱动显存峰值18.89GiB/卡，worker树RSS峰值369.72GiB、PSS采样峰值230.98GiB；RSS不能等同独占物理RAM，采样可能遗漏短峰值。本轮没有增加固定形状内存专项，也不能据一次形状扫描宣称长期无泄露。当前CUDA成片验证仅覆盖上述Ref2VA六项；其他任务／桥接精度组合不沿用这些性能或人工精度结论。

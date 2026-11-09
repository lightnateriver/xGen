# H3 CUDA TP + CP 迁移约定

状态：已实现 experimental 备选路径，并完成 CPU/Gloo 与四卡 CUDA 合成单层验证。完整八卡原权重的视频性能、满载容量和人工音画审核尚未完成。参数和验证命令见 [使用说明](H3-CUDA-TP-CP.md)。

基线：v0.3.2-maas / 6081426。参考 vLLM-Omni e2b877f5 的 H3 TP、pure-Ulysses 本地准备及本地输出路径。NVFP4 实验不作为本分支的基线。

## 默认与备选路线

默认始终使用现有 SP 路径。TP + CP 是显式启用的 CUDA H3 experimental 备选，不根据输入大小或 OOM 自动切换。

原 SP 的输入准备、输出 gather、精度配置和显存调度都保留。本地输入/输出优化也先位于新路线内；验证时可显式采用 TP1×CP8 检查该边界优化，不能未经确认替换默认 SP 实现。

这里 CP 指 Token/Context Parallel，落在现有 Ulysses SP 轴上；不使用 CFG/batch 维度的 cp_size 代替。H3 首版 CFG=1、Ring=1。

## 任务范围

T2VA、I2VA（代码中的 fl2va）、R2VA（ref2va）共享同一套 DiT 并行和通信实现；FL2VA / Ref2VA 仍加载各自的原始权重。任务不会决定通信量化精度。

首先验证 R2VA 的 1 图 / 5 图 / 满载和大 Token 峰值，再对 T2VA / I2VA 做兼容性回归。支持的代码路径、技术验证、人工画面/口型/音频认可分别记录，不能相互代替。

首版仅针对 CUDA。NPU、其他模型和未启用新功能的旧脚本保持现状。文本 TP8、DiT TP2×Ulysses4、VAE SP8 使用各自通信组；DiT 新配置不改变文本 TP 的通信精度。

## 传输配置

TP 和 CP 的传输精度独立配置，提供四个组合：

| DiT TP 传输 | CP 传输 | 用途 |
|---|---|---|
| BF16 | BF16 | 新并行路径的精度对照 |
| BF16 | FP8 E4M3 | 保留高精度 TP，压缩 CP |
| FP8 E4M3 | BF16 | 单独检验 TP 传输量化 |
| FP8 E4M3 | FP8 E4M3 | 双向压缩的性能候选 |

新开关的拟定契约：

- DiT TP size 默认 1，表示不启用新 TP 路径。
- 并行模式默认 sp；tp_cp 必须显式选择。sp 模式下若传入仅属于新路线的非默认参数，应明确报错，避免参数被静默忽略。
- TP comm dtype 默认 BF16；只作用于新 DiT TP。
- CP comm dtype 默认 auto，继承原 profile / XGEN_PHAA_FP8 行为；显式 BF16 / FP8 只改变启用的新并行路径。
- 原 XGEN_PHAA_FP8 仍表示去程、回程同时开启；不改变旧开关的解释。
- 只有传输配置改变，权重、Linear W8A8、Sage、AdaLN、采样步数不跟随切换。
- FP32 视频/音频输出头及最终预测聚合保持 FP32；RoPE、模态和位置索引等保持既有数据类型。

配置必须在启动时解析并打日志；常驻 worker 不在请求间静默切换精度或通信组。与原 profile 的冲突处理必须显式定义，不能依靠环境变量覆盖顺序。

## FP8 TP 归约契约

FP8 只用于线上传输，不使用当前 PHAA 的打包字节直接做求和。

1. 对 BF16 partial sum 按小块计算动态 scale，编码为 FP8 数据及 scale。
2. 通过 TP 组交换量化分片；所有 rank 解码同一组数据，包含本 rank 已量化重构的分片。
3. 按一致顺序在 FP32 中求和，最后转回原 BF16 输出。bias 在归约后只加一次。
4. 分块限制临时缓冲，不能为大 Token 请求保存各 rank 的完整 FP32 副本。
5. 首先验证 TP2；其他 degree 必须通过独立数值、缓冲峰值和通信测试后才列入已支持配置。非法组合提前报错。

高精度求和无法恢复 FP8 编码误差。除传输误差外，还必须控制 TP 分片前后的权重/激活 scale、GEMM 输出精度和 fused QKV / gate-up 排列变化。

## 共用实现与 R2VA 特殊边界

共用部分：本地 embedding / RoPE、TP Linear、QKV head 分片、FFN、CP All-to-All、本地输出头和低维预测聚合。

R2VA 的差别保留在参考媒体编码和 packed metadata：参考/目标 Token、modality tag、timestep index、update mask、padding 和最终输出位置必须一起正确切分。不因开启 CP 而丢弃参考文件、静默缩小图片或减少视频帧。

pure-Ulysses 是首版路径。Ring / 任意 mask / 多请求 packed 推理另行验证，不从上游能力声明直接推导 xGen 已支持。

## 验证顺序

1. CPU / Gloo 小形状验证：分片重组、TP 组隔离、QKV 和 gate/up 映射、bias、FP32 最终输出、无重复/遗漏的参考 Token。
2. CUDA 通信验证：BF16 对照、FP8 有界误差、多 rank 一致性、scale 打包、不同形状、峰值临时缓冲；显式检查无低精度 silent fallback。
3. 固定 5 秒 R2VA 样例，对四种传输组合完成同条件预热后 20 步对照。1 步仅用于链路与 OOM 检查。
4. 选择有依据的配方扩展到 1 / 5 / 满载参考 × 5 / 10 / 15 秒，并补 T2VA / I2VA 回归。初始化、各阶段、通信与等待、显存/内存及人工质量结论分别记录。

在通过测试前，所有 FP8 TP 和新 R2VA TP 组合均为 experimental；不能继承旧 SP8 的成片精度结论。

## 分阶段开发计划

| 阶段 | 工作与拟修改位置 | 退出条件 |
|---|---|---|
| 1. 冻结默认行为 | 记录 6081426 默认参数、profile 解析和调用路线；新增独立 h3_parallel 配置解析；日志给出 mode、TP/CP 大小、通信 dtype | 不加新参数的 SP 路径仍走原分支，NPU/Wan 等不进入新逻辑；非法 rank 数和组合在模型加载前报错 |
| 2. 通信组隔离 | 复用 ParallelManager / ProcessGroupMesh，组建 DiT TP2×SP4；为新层绑定显式 TP group；保留旧全局 TP 的默认行为 | 文本 TP8 和 DiT TP2 互不覆盖，重复请求、异常清理和 VAE 阶段不改变组归属 |
| 3. 本地输入/输出边界 | 新增 h3_local_io helper；只在新路线选择本地 embedding / RoPE / metadata 和 FP32 输出投影，复用同一 DiT block loop | TP1×CP8 对照覆盖三模态、参考/目标、padding、最终 mask；聚合的是低维预测，默认 SP 不受影响 |
| 4. BF16 TP 链路 | H3 专用 TP adapter 复用 Column/Row Parallel 与 FP8 GEMM；CPU 加载时分片 QKV、gate/up、out/down 及 FP32 heads；统一 scale 和 bias 契约 | 不先在 GPU 构建完整权重；TP2×CP4 BF16 通信完成数值与单步链路验证 |
| 5. 独立通信精度 | 复用现有 CP FP8 编解码，提供 CP BF16/FP8；新增有界分块 FP8 TP 交换和 FP32 归约；旧开关语义不变 | 四种组合可独立验证；各 rank 结果一致；无 silent fallback；日志区分打包、传输、解码、归约耗时及缓冲峰值 |
| 6. GPU 验证与评审 | 先固定 5s R2VA 样例验证四种组合，再以 1 步检查 15s 满载容量，使用 20 步复核成片；扩展选定配置到 9 项 R2VA，补 T2VA/I2VA 回归 | 每个结论关联输入、源码、参数、warmup/measure 和视频哈希；人工音画审核与技术通过分开 |
| 7. 文档和交付 | 补独立部署教程、参数说明、有效限制和性能/精度表；README 将 TP+CP 列为备选；整理可审查提交 | 原 SP 启动命令保留；关闭新模式并重启 worker 即恢复原链路；真实验证前不宣传已支持全部形状或已有性能收益 |

### 最小侵入边界

- 新增 H3 配置/并行 adapter、本地 IO helper、FP8 reduction helper 和对应测试；共享层只增加默认保持旧行为的可选 group 绑定。
- 复用原 InferenceManager、模型加载入口、PHAA、Attention backend、VAE 和编码器；不复制一套完整推理循环，也不引入 vLLM 作为运行依赖。
- 用不可变的组/模型布局配置与每次 forward 的局部 metadata，避免请求间遗留 Token 索引和 mask。
- 不改 KV/条件缓存策略，不减少采样步数，不裁剪参考输入，不引入量化权重重做、失败重试或 latent 恢复。
- 本版保持现有 W8A8 Linear + Sage 可选配方；W4A8、NVFP4、Ring、DLO 作为后续独立工作，避免混淆迁移收益。

### 对照口径

同一权重、输入、seed、20 实际 DiT、Sage 内核、传输 dtype、常驻层数和 VAE 配置下对照 SP8 与 TP2×CP4。每种形状在同一模型进程中预热后正式计时，初始化单列。

记录参考媒体编码、各步 DiT、TP/CP 字节数与调用数、通信/计算重叠、输出聚合、VAE、编码封装、allocated/reserved、驱动显存、RSS/PSS。异步阶段区间不直接相加。失败或外部任务干扰保留记录，不混入正常均值。

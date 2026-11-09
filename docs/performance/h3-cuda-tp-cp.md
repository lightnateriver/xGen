# H3 CUDA TP/CP 最小验证（2026-11-06）

本轮验证 experimental 备选代码，默认仍是原 SP。没有读取完整 H3 权重，也没有生成视频。

## 已验证

- 71 项配置、切片、兼容性和 HTTP 边界测试通过，9 项按环境跳过；其中本地 12 项、远端 core 48 项、HTTP 9 项、基础量化 2 项。跳过项不能计入通过。
- CPU/Gloo 八 rank：TP2×CP4，文本全局 TP8 保持独立，实际文本行归约探针通过；BF16 和 FP8 TP 各 3 个 packed 布局通过。
- RTX 5090 四卡：TP2×CP2，FP8 Linear + Sage，正式 prepare_transformer、rank-owner AdaLN、indexed affine、Q/K RMS、partial RoPE 与输出 AdaLN 融合；四种传输组合各 3 个布局通过。
- CUDA 两卡：FP8 Linear + BF16 cuDNN Attention、TP2，各 3 个布局通过。另有独立 TP2 / CP2 的 Sage 探针。
- 逐 rank 断言 FP8 TP 输出逐元素相同；FP8 TP 按块重构误差 oracle 完全匹配，归约后 bias 只加一次；CPU 权重切片与 fused QKV / SwiGLU 映射单测通过。

## 单层条件

四卡主验收使用随机初始化的一层 H3，hidden=128、FFN=256、heads=56、head_dim=128、RoPE=96 channels。权重 seed=42，三种输入 seed=117/122/119。保留真实 H3 的 head 几何，缩小 hidden / FFN / timestep / 编码器维度以节省资源。它不是完整尺寸 H3 的单层容量测试。CPU 八 rank 和两卡 cuDNN 辅助验证使用 8 heads、head_dim=32 的更小几何。

packed 长度为 17 / 22 / 19，视频/音频/文本索引随机交错，包含两种 timestep（目标与参考）、CP 尾部 padding 和负 tag mask。每种配置复用同一模型连续处理三种布局。

误差是相对于同输入、同随机权重的未分片 W8A8 FP8 Linear + Sage 完整文档输出，取视频/音频两个 FP32 预测及全部 rank 的最差 relative RMSE；包括并行、激活量化尺度和融合差异，不是仅通信量化的消融。

| TP 传输 | CP 传输 | 最大 relative RMSE | 结果 |
|---|---|---:|---|
| BF16 | BF16 | 0.4397% | 通过 |
| BF16 | FP8 | 0.4627% | 通过 |
| FP8 | BF16 | 0.4559% | 通过 |
| FP8 | FP8 | 0.4778% | 通过 |

四卡测试的 PyTorch 峰值 allocated / reserved 为 62.07 / 84.00 MiB/卡（取最差 rank）。这是缩小模型的功能测试数据，不代表完整 H3 显存，不包含所有驱动/NCCL 开销，不能推导满载 OOM 边界或 E2E 提升。测试进程退出后，所用四张卡 driver memory 回到 0 MiB。

## 环境与证据

源码基于 6081426；原始 rank JSON、日志、source SHA-256 和退出检查保存在本地实验目录。环境为 RTX 5090、Torch 2.12.0+cu130、CUDA 13.0、Triton 3.7.0、SageAttention 2.2.0。Diffusers 为 H3 overlay，不应仅凭元数据版本重建环境。

## 后续验收

完整八卡原权重的 T2VA / I2VA / R2VA（1 图、5 图、满载）、预热后 20 步性能、15 秒容量、VAE/编码成片及人工画面/口型/听音仍需单独实测。单层通过不继承旧 SP 的人工精度结论。HTTP 此轮为启动参数和输入边界测试，没有声称服务成片通过。

启动与回退见 [使用说明](../H3-CUDA-TP-CP.md)。

# 性能与验证记录

[项目首页](../../README.md) · [文档目录](../README.md) · [20 步视频启动](../H3-VIDEO-QUICKSTART.md)

**当前推荐视频方案看 20 步报告；历史 50 步、不同解码器及组件微基准分别保留。** 本次文档整理没有重新运行性能测试，不改变已有数据和人工验收状态。

## 当前音视频方案

| 设备／方案 | 报告与数据 | 范围与验收 |
|---|---|---|
| CUDA 5090，Sage-FP8 全并行 Streaming＋AdaLN＋Align16 | [12 项在线性能与显存](h3-cuda-phaa-align16.md) · [CSV](h3-cuda-phaa-align16.csv) | T2VA/I2VA、R2VA 单图／五图，5／10／15 秒，20 步；12 项技术通过，完整人工动态／口型／听音待记录；跨节点差异不能全部归于 Align16 |
| CUDA 5090，同系列 Streaming 离线 | [输入规格、性能与显存](h3-cuda-streaming-matrix.md) · [CSV](h3-cuda-streaming-matrix.csv) | 20 步；owner／table 逐项标明，不能当作统一 table-only 对照 |
| NPU 910B1，Laser INT8 全并行 Streaming | [T2VA/I2VA 新复测及既有 R2VA](h3-npu-full-parallel-streaming.md) · [CSV](h3-npu-full-parallel-streaming.csv) | 20 步；T/I 与既有 R2VA 的代码版本分开记录；常驻推理保存，不是 HTTP SLA |
| NPU 910B3，VC＋Ascend 融合 | [同机 Laser 对照、显存及算子命中](h3-v06-npu-vc-fusions.md) | R2VA 单图 5／10 秒、20 步，样例画面与声音已验收；不覆盖全部多媒体／时长组合 |
| 跨设备 native-stream 存储与默认切换 | [共享 FP16 存储](h3-native-shared-fp16.md) · [后端发布验证](h3-vae-release.md) | 单模块、单步接入与完整请求分开，不将容量冒烟当作完整精度或加速结论 |
| CUDA v0.5 整理合入回归 | [三条 20 步冒烟与配置](h3-v05-cuda-release-validation.md) | T2VA/I2VA/R2VA；不能取代当前矩阵 |

## H3-Image

| 报告 | 范围 |
|---|---|
| [40 项性能与输入 shape](h3-image-npu-40cases.md) · [CSV](h3-image-npu-40cases.csv) | 四卡 NPU、6 步、历史文本 BF16；36 项生成成功、4 项预期拒绝，人工画质状态逐项记录 |
| [Image 首版发布验证](h3-image-v06-release-validation.json) | 合入前硬件证据与合入后 CPU 检查分开 |
| [文本 W8A8 验证摘要](h3-image-text-w8a8-default-validation.json) · [实测解释](../H3-IMAGE-TEXT-INT8.md) | 当前 NPU 文本 W8A8 主要减少显存，未证明稳定提速；旧 40 项矩阵不能重标为文本 INT8 |

## 历史完整视频性能

| 报告 | 适合查看什么 |
|---|---|
| [原首页历史性能汇总](h3-historical-overview.md) | B1/B3 的 20 步旧矩阵、50 步在线／离线表、早期 GPU/NVFP4 与 legacy decoder 对照，原始口径保留 |
| [NPU 35 项离线 benchmark](h3-npu-offline-20261105.md) · [正式 CSV](h3-npu-offline-20261105.csv) · [预热 CSV](h3-npu-offline-20261105-warmup.csv) | 20 步、768P／2K、多媒体参考；使用已移除的阶段清理策略，不能视为当前策略回归 |
| [NPU 50 步在线／离线](historical-pairs.csv) · [阶段拆分](historical-stages.csv) | 历史 B2/B3（B2 实际为 910B1）、冷启动／补测与缺失 E2E 标记 |
| [GPU 历史完整对照](h3-cuda.md) · [Sage 20 步 CSV](h3-cuda-sage-20step.csv) | 旧 FP8／BF16、NVFP4、Sparge、不同 VAE，不能直接与当前推荐方案比较 |
| [v0.4 R2VA 全并行](h3-cuda5090-full-parallel.md) | 旧解码器、条件暂存与早期整网优化的版本边界 |
| [v0.2 冒烟](v02-smoke.csv) · [旧验证摘要](validation-summary.json) | 早期超分链路与兼容性记录 |

## 组件与实验

| 报告 | 边界 |
|---|---|
| [条件准备重叠](h3-condition-overlap.md) | 默认关闭；CUDA 组件计时和 1 步容量／张量对照，非 20 步 E2E，NPU 真机待验收 |
| [VAE 三后端](h3-omni-vae.md) · [NPU 三后端](h3-vae-npu.md) | 最小组件测试，不能直接解释为整网加速或成片精度 |
| [TP/CP](h3-cuda-tp-cp.md) · [布局时延](h3-cuda-tp-cp-latency.md) | 实验布局，单层合成结果与完整视频分开 |
| [CPU 可见水印](h3-watermark-cpu.md) | 仅测编码前水印增量，不是模型推理收益 |

## 阅读口径

- 核对设备／卡数、代码版本、输入文本长度及媒体 shape、输出时长／分辨率、精度、并行布局和实际步数。
- 核对初始化、预热、重复次数、计时起止；推理、保存、HTTP、下载不是同一个范围。
- 显存区分驱动占用与 PyTorch allocated/reserved，主机内存区分 RSS/PSS；有限样本不能证明长期无泄露。
- 技术解码、抽帧诊断、人工动态／口型／听音分别记录，不沿用到未经验证的新配置。

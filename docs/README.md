# xGen 文档

[项目首页](../README.md) · [版本记录](H3-VERSIONS.md)

音视频模型从 **[20 步视频启动教程](H3-VIDEO-QUICKSTART.md)** 开始；图像生成从 **[H3-Image](H3-IMAGE.md)** 开始。两者使用各自的模型、VAE 和依赖配置。

## 部署与服务

| 需要做什么 | 主文档 |
|---|---|
| 下载模型、准备对应资产 | [模型与下载地址](H3-MODELS.md) |
| CUDA/NPU 离线生成，5／10／15 秒、768P／2K | [20 步视频启动教程](H3-VIDEO-QUICKSTART.md) |
| 选择 CUDA 5090 全并行方案及独立开关 | [CUDA 全并行配置](H3-CUDA-COMPOSABLE.md) |
| 选择 NPU Laser INT8 默认方案及回退 | [NPU 默认配置](H3-NPU-DEFAULT.md) |
| 启用已验证的 NPU VC＋Ascend 融合 | [安装、开关与适用范围](H3-NPU-FUSIONS.md) |
| 准备 CUDA 20 步 AdaLN 表及剔除权重的模型 | [CUDA AdaLN 资产](H3-CUDA-ADALN.md) |
| 部署 H3-Image 文生图／参考图编辑 | [图像入口](H3-IMAGE.md) · [NPU 四卡推荐部署](H3-IMAGE-NPU-OPTIMIZATIONS.md) · [文本 W8A8 与回退](H3-IMAGE-TEXT-INT8.md) |
| 启动 HTTP，提交、查询和下载 | [在线服务](H3-ONLINE.md) |
| 对接 CUDA MaaS | [API](../server/video/MAAS-API.md) · [V2 对齐范围](H3-MAAS-V2.md) · [OpenAPI](openapi/h3-maas-v2.json) |
| 自动运行 NPU 离线 benchmark | [Benchmark](H3-NPU-BENCHMARK.md) |
| 检查环境或让 Agent 启动 | [环境采集](../tools/README.md#cuda-和-npu-环境采集) · [Agent 指南](../.agents/skills/xgen-h3/SKILL.md) |

## 配置与验证

| 主题 | 文档 |
|---|---|
| CUDA/NPU 共用的组件调度 | [通用调度与硬件边界](H3-PORTABLE.md) |
| 视频解码与 native-stream | [VAE 后端](H3-OMNI-VAE.md) · [在线分块编码](H3-ONLINE-STREAMING.md) |
| FP8 PHAA Align16 | [作用范围与关闭方法](H3-PHAA-FP8-ALIGNMENT.md) |
| 参考图片尺寸 | [短边参数与默认 2048](H3-REFERENCE-IMAGE-SIZE.md) |
| Input Token Refiner | [配置与历史单模块测量](H3-INPUT-TOKEN-REFINER.md) |
| 功能性测试与验收边界 | [功能测试记录](H3-FUNCTIONAL-TESTS.md) |
| 性能、显存和精度状态 | [性能目录](performance/README.md) |

视频教程按 **20 次实际 Base DiT** 编写；2K Refiner 另计 3 步，H3-Image 固定 6 步。底层入口、历史配置和显式参数仍可使用其他步数，必须匹配相应权重和表。

## 回退、历史和实验

这些文档保留兼容与研究价值，不作为新用户的默认启动入口。

| 用途 | 文档 |
|---|---|
| 保留的 FP8 Linear＋BF16 attention 精度参考 | [fp8-bf16](H3-CUDA-PRECISION.md) |
| 保留的 Sage-FP8 基础方案 | [Sage 基础配置与历史性能](H3-CUDA-SAGE.md) |
| 历史 NVFP4／FastVideo／CUDA 2K 配方 | [CUDA 扩展部署](H3-CUDA.md) |
| Spark-H3／Veda／Turbo 可选配方 | [Attention 配方](H3-ATTENTION.md) · [验证记录](H3-ATTENTION-VALIDATION.md) |
| 默认关闭的 R2VA 条件准备重叠 | [配置、显存代价与待验证范围](H3-CONDITION-OVERLAP.md) |
| 默认关闭的 TP/CP 实验 | [启动与布局](H3-CUDA-TP-CP.md) · [设计](H3-CUDA-TP-CP-DESIGN.md) · [共享布局](H3-SHARED-TP-CP.md) |
| 旧部署工具与 2K 资产准备 | [基础部署](H3-DEPLOYMENT.md) |
| 版本升级与早期步数语义 | [版本表](H3-VERSIONS.md) · [v0.6](H3-V06-RELEASE.md) · [CUDA 默认升级](H3-V05-CUDA-UPGRADE.md) · [早期升级](H3-UPGRADE.md) |

每个主题的完整参数以表中的主文档为准；首页只保留推荐用法。实验参数不因文档整理而自动开启。

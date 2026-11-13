# 版本与升级

[文档目录](README.md) · [当前推荐部署](H3-VIDEO-QUICKSTART.md)

以下按版本倒序记录功能演进。“原默认”“旧路径”等措辞仅描述相应发布时的行为；当前 CUDA/NPU 推荐视频示例均按 **20 次实际 Base DiT** 编写。历史 50 步能力及已有显式配置继续保留。

| 分支 | 核心变化 | 文档 |
|---|---|---|
| [v0.6](https://github.com/lightnateriver/xGen/tree/v0.6) | 在最新v0.5基础上增加MiniMax-H3-Image六步文生图/参考图编辑与四卡NPU W8A8常驻方案，以及默认关闭的音视频VC＋Ascend融合、R2VA条件准备重叠；保留视频默认与GPU Align16。 | [图像入口](H3-IMAGE.md) · [优化](H3-IMAGE-NPU-OPTIMIZATIONS.md) · [40项性能](performance/h3-image-npu-40cases.md) · [VC融合实测](performance/h3-v06-npu-vc-fusions.md) · [条件重叠](H3-CONDITION-OVERLAP.md) |
| [v0.5](https://github.com/lightnateriver/xGen/tree/v0.5) | CUDA Base 默认 Sage-FP8 全并行 Streaming + AdaLN 表，覆盖 T2VA/I2VA/R2VA；保留旧精度回退。NPU Base 默认 Laser INT8全并行Streaming，覆盖T2VA/I2VA/R2VA、20实际步、SP8，共用组件调度与Stream FP16存储。 | [NPU默认与升级](H3-NPU-DEFAULT.md) · [复测](performance/h3-native-shared-fp16.md) · [通用适配](H3-PORTABLE.md) |
| [v0.4](https://github.com/lightnateriver/xGen/tree/v0.4) | 增加 `cuda-5090 r2va` 优选 Sage-FP8 全并行预设：SAP、本地输入投影、请求内固定条件复用及组件并行，DiT 全程常驻；各项可独立关闭。新增统一 `--maas`、多模态输入校验、首尾帧、768P 自适应比例和公网素材下载。旧 CUDA/NPU 默认保留，TP/CP 为默认关闭的实验选项。 | [启动与开关](H3-CUDA-COMPOSABLE.md) · [实测性能](performance/h3-cuda5090-full-parallel.md) · [MaaS API](../server/video/MAAS-API.md) · [在线启动](H3-ONLINE.md) |
| [v0.3.2-maas](https://github.com/lightnateriver/xGen/tree/v0.3.2-maas) | CUDA 提供默认精度参考 fp8-bf16 和更快的 Sage-FP8 两套标准方案；增加输出 AdaLN 融合与长视频实测。保留 MaaS 排队、超时、健康探针、原子发布、环境采集，以及 NPU 离线 benchmark、R2VA 多媒体参考、实际步数语义和逐视频显存回收。 | [CUDA 方案一](H3-CUDA-PRECISION.md) · [CUDA 方案二](H3-CUDA-SAGE.md) · [API](../server/video/MAAS-API.md) · [在线启动](H3-ONLINE.md) · [benchmark](H3-NPU-BENCHMARK.md) · [升级说明](H3-UPGRADE.md) |
| [v0.3.1](https://github.com/lightnateriver/xGen/tree/v0.3.1) | CUDA 配置集中到 adaptor。增加可选异步 FP8 PHAA、回程直写、DiT 算子融合和 VAE 空间切片。完成 CUDA、NPU 双设备冒烟。 | [实测](performance/h3-cuda.md#v031-双设备冒烟) |
| [v0.3-cuda](https://github.com/lightnateriver/xGen/tree/v0.3-cuda) | CUDA H3 FL2VA、NVFP4 DiT 常驻、Sage/Sparge、FP8 PHAA 双向通信、文本 TP8、VAE SP、FastVideo C4 预设及可选 FL2VA 2K。 | [CUDA 部署](H3-CUDA.md) |
| [v0.3](https://github.com/lightnateriver/xGen/tree/v0.3) | NPU Base DiT 增加可选 Spark-H3 和 Veda。支持 Veda 稀疏度配置和实验性 LightX2V 8 步配方。 | [启动和回退](H3-ATTENTION.md) · [实测](H3-ATTENTION-VALIDATION.md) |
| [v0.2](https://github.com/lightnateriver/xGen/tree/v0.2) | 增加默认关闭的 H3 → Open-H3-Regenerate-2K 链路，复用原 Manager 和 HTTP 服务。 | [NPU 部署](H3-DEPLOYMENT.md) |
| [v0.1](https://github.com/lightnateriver/xGen/tree/v0.1) | 原始 xGen，包含 Diffusers 适配、SP、FSDP、PHAA、量化和视频服务。 | [基础示例](../example/diffusers) |

## v0.6 当前补充

- 视频默认继承 Sage-FP8 全并行 Streaming＋AdaLN 表（CUDA）和 Laser INT8 全并行 Streaming（NPU）。
- NPU 可显式开启 VC＋Ascend 融合；当前整网性能和人工视听证据限定八卡 910B3、单图 R2VA 5／10 秒、20 步。
- H3-Image 使用独立模型与依赖，NPU Qwen 文本解码器默认 W8A8，CUDA 文本默认 BF16。
- R2VA 条件准备重叠使用 CUDA/NPU 共用开关，默认关闭；CUDA 一步接入和容量验证不代表 NPU 或完整步数精度验收。
- 文档按用户入口重组，当前推荐的 CUDA Base／MaaS／NPU Refiner 示例 JSON 统一为 20 步；Refiner 示例同步选择匹配的 21 点 Base 表。运行实现与低层默认参数保持原样。

详见 [v0.6 发布说明](H3-V06-RELEASE.md)、[CUDA 升级与兼容](H3-V05-CUDA-UPGRADE.md) 和[早期步数语义升级](H3-UPGRADE.md)。原始性能证据见[性能目录](performance/README.md)，不从历史版本名称推导当前默认值。

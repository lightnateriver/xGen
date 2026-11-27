# xGen 文档

[项目首页](../README.md)

安装和资产准备集中在模型文档，运行命令集中在视频、图像和在线指南；设备差异与开关分开说明。源码目录按模型组织，文档按使用任务和职责组织。

| 需要做什么 | 文档 |
|---|---|
| 理解分层职责、组件归属和开发检查 | [架构](ARCHITECTURE.md) |
| 配置环境、下载模型、准备 AdaLN 和 Refiner | [模型与资产](H3-MODELS.md) |
| 文字、首尾帧、多媒体参考生成视频，或开启 2K | [视频生成](H3-VIDEO-QUICKSTART.md) |
| 文生图和参考图编辑 | [H3-Image](H3-IMAGE.md) |
| CUDA 默认方案、依赖和精度回退 | [CUDA](H3-CUDA.md) |
| NPU 默认方案、VC 与融合算子安装 | [NPU](H3-NPU-DEFAULT.md) |
| 启动 HTTP、提交任务和对接 MaaS | [在线服务](H3-ONLINE.md) |
| 查配置优先级、组件开关和组合约束 | [统一配置](H3-CONFIGURATION.md) |
| 查版本能力和迁移注意事项 | [版本与迁移](H3-VERSIONS.md) |

视频示例显式使用 20 次实际 Base DiT；2K Refiner 另计 3 步，H3-Image 固定 6 步。步数必须匹配模型与 AdaLN 资产。示例中的卡数、精度和任务是具体配置，不代表所有输入规模都能运行。

协议详情由 [MaaS API](../server/video/MAAS-API.md)、[FL2VA API](../server/video/MAAS-FL2VA-API.md) 和 [OpenAPI](openapi/h3-maas-v2.json) 维护。各包 README 说明各自公共接口；第三方来源、许可和版本信息保留在对应实现目录。

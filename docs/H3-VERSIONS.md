# 版本能力与迁移

[文档目录](README.md)

本页记录版本能力和配置迁移。

| 版本 | 主要能力 |
|---|---|
| v0.7 | 收拢框架分层、模型装配和请求生命周期；5090 Sage-FP8 默认开启 DiT 扩展融合及请求内 RoPE 表 |
| v0.6 | 独立 H3-Image 生成/编辑；NPU 文本 W8A8、VC 与 Ascend 融合；Ref2VA 图片缩放选择；CUDA native-stream 可选 VAE 编译 |
| v0.5 | CUDA/NPU 共用组件调度、full-parallel 与 native-stream；继续保留各自精度和后端约束 |
| v0.4 | CUDA H3 组件并行与全并行预设；MaaS V2、首尾帧、多媒体校验与输出发布 |
| v0.3 系列 | CUDA 精度/通信适配与可选 Base attention，旧配置保留显式选择 |
| v0.2 | 可选 Base → latent upscaler → Refiner 2K 链路 |

<a id="migration"></a>
## 升级时核对

1. **实际步数**：H3 的 `num_inference_steps` 填实际 Base DiT 次数。普通 NPU 21/51 点表分别匹配 20/50 次；Turbo/LoRA 调度按各自规则校验。不能只改文件名或请求步数。
2. **模型与资产**：T2VA/I2VA 使用 FL2VA，Ref2VA 使用独立模型与表；CUDA/NPU 准备工具及布局不能混用。原始权重用于 owner，stripped Transformer 与对应表一起使用。
3. **启动层级**：推荐 shell/JSON 会显式选择配置，低层 CLI 的未指定默认不一定相同。部署前查看 [CUDA](H3-CUDA.md)、[NPU](H3-NPU-DEFAULT.md) 或 [Image](H3-IMAGE.md)。
4. **原生 VAE**：native-stream 需要独立可信 `video_vae` 目录和授权。更改 decoder、表模式、精度、并行或固定模型后重启，不能在同一常驻实例内热切换。
5. **服务协议**：普通 HTTP、旧任务接口和 MaaS V2 的状态、错误与交付字段分别遵守 [API](../server/video/MAAS-API.md)。模拟 IR/2K 不等于真实后端。
6. **扩展能力**：TP/CP、条件重叠、编译、稀疏 attention 与 Turbo 均按 [统一配置](H3-CONFIGURATION.md) 的适用范围显式启用。

<a id="documentation"></a>
## 文档和开发工具

主文档按使用任务与架构职责收拢，实验过程、专项验证/基准记录和独立基准脚本已从当前工作区清理。第三方许可证、来源声明、协议定义和模块 README 仍由各自目录维护。

模型实现继续采用既有包入口与补丁注册方式，组件加载由 adaptor 统一编排。质量检查及运行验证的边界见 [架构](ARCHITECTURE.md#validation)。

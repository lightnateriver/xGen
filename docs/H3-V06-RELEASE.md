# xGen v0.6：MiniMax-H3-Image NPU

v0.6以最新v0.5提交 `ca6b6f0adcca48e523d8b81f32c9b784c577f095` 为基线，加入独立的MiniMax-H3-Image六步文生图与参考图编辑入口。合入的两个开发提交为 `cc602ebe83766e095ad464bed6015cd303b0a83f`、`268f46eb8dbf2a1d3f804526e85b923ecb7fb41a`；发布提交压缩合入代码及去敏文档。

## 部署与兼容

使用[图像专用可选Diffusers依赖](H3-IMAGE.md#环境与权重)，固定提交 `0905b667fe3f37f43b6ef9c6e05ad20636206938`。推荐[四卡W8A8常驻命令](H3-IMAGE-NPU-OPTIMIZATIONS.md)：SP4/text TP4、dense SDPA、六步AdaLN表、DiT常驻、条件缓存、refiner/vision/VAE tile并行、FFN chunk2048、RGB8输出与本地输入投影。

这些是图像入口显式启用的优化。省略优化开关保留BF16卸载路径。视频的NPU Laser INT8 Streaming默认、CUDA Sage-FP8/AdaLN默认及GPU PHAA Align16继承v0.5；Image不使用视频AdaLN表、音频VAE或MP4 Streaming。v0.6表示Git发布分支，独立Python子包沿用原有包版本。

## 已完成的硬件验证

[40项性能/shape报告](performance/h3-image-npu-40cases.md)及[CSV](performance/h3-image-npu-40cases.csv)记录固定六步、batch1、seed42、warmup0/repeat1：36项生成成功，4项真实输入校验按预期拒绝，0 OOM；28项目视通过、8项部分符合。两版共同成功35项累计推理4429.153→691.899秒，6.401×。初始化另计。

最大组合4096有效prompt token、16张8192×2048参考图、5120×4096输出完成六步，1958.130秒，单卡allocated50.009 GiB、reserved58.811 GiB。常驻小输入显存约37 GiB/卡，高于原offload约21 GiB/卡；相对BF16常驻约55 GiB/卡，W8A8减少显存。历史设备布局和首请求策略不同，不能把全方案加速归于单个算子。

该硬件矩阵针对合并前开发源码；发布时检查Image执行源码逐字节一致，并重新运行CPU分布式/配置回归，未重新加载模型跑NPU矩阵。[发布检查摘要](performance/h3-image-v06-release-validation.json)分别标明新检查与既有硬件证据。原始prompt、媒体、逐样本业务输出、机器信息与私有路径不随公开仓库分发。

合并后新检查：84项FP32/BF16 SP与本地输入张量对照通过，另有完整小Transformer TP、FFN分块和VAE拼接独立验证通过；190项CPU回归及13个在线子测试通过，1项CUDA checkpoint测试跳过。GPU transport的实际CUDA执行未在本次NPU环境重测；检查了其CPU/meta拒绝合同，并保留最新v0.5实现。

## 已知范围

纯Laser产生非有限值及黑图，不在推荐方案中。混合Laser尾层回退、SP2/DiT TP2及QKV/SwiGLU/RoPE融合仅保留为显式实验选项；Triton SwiGLU在算子探针中更慢。图像CUDA与HTTP服务没有做本次硬件验证。多图编号、人物姿势与最大组合参考保真度的局限见性能报告。

## 合并结果

唯一文本冲突在README版本介绍：同时保留v0.5 GPU Align16说明和v0.6 Image入口。运行代码无冲突，未覆盖v0.5的视频执行模块、配置和测试。

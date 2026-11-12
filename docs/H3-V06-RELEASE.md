# xGen v0.6：MiniMax-H3-Image NPU

v0.6以最新v0.5提交 `ca6b6f0adcca48e523d8b81f32c9b784c577f095` 为基线，加入独立的MiniMax-H3-Image六步文生图与参考图编辑入口。合入的两个开发提交为 `cc602ebe83766e095ad464bed6015cd303b0a83f`、`268f46eb8dbf2a1d3f804526e85b923ecb7fb41a`；发布提交压缩合入代码及去敏文档。

## 部署与兼容

使用[图像专用可选Diffusers依赖](H3-IMAGE.md#环境与权重)，固定提交 `0905b667fe3f37f43b6ef9c6e05ad20636206938`。推荐[四卡W8A8常驻命令](H3-IMAGE-NPU-OPTIMIZATIONS.md)：SP4/text TP4、dense SDPA、六步AdaLN表、DiT常驻、条件缓存、refiner/vision/VAE tile并行、FFN chunk2048、RGB8输出与本地输入投影。

除NPU文本W8A8默认开启外，这些是图像入口显式启用的优化。使用 `--no-image_text_int8` 并省略其他优化开关可保留原BF16卸载路径。视频的NPU Laser INT8 Streaming默认、CUDA Sage-FP8/AdaLN默认及GPU PHAA Align16继承v0.5；Image不使用视频AdaLN表、音频VAE或MP4 Streaming。v0.6表示Git发布分支，独立Python子包沿用原有包版本。

## 已完成的硬件验证

[40项性能/shape报告](performance/h3-image-npu-40cases.md)及[CSV](performance/h3-image-npu-40cases.csv)记录固定六步、batch1、seed42、warmup0/repeat1：36项生成成功，4项真实输入校验按预期拒绝，0 OOM；28项目视通过、8项部分符合。两版共同成功35项累计推理4429.153→691.899秒，6.401×。初始化另计。

最大组合4096有效prompt token、16张8192×2048参考图、5120×4096输出完成六步，1958.130秒，单卡allocated50.009 GiB、reserved58.811 GiB。常驻小输入显存约37 GiB/卡，高于原offload约21 GiB/卡；相对BF16常驻约55 GiB/卡，W8A8减少显存。历史设备布局和首请求策略不同，不能把全方案加速归于单个算子。

该硬件矩阵针对合并前开发源码；发布时检查Image执行源码逐字节一致，并重新运行CPU分布式/配置回归，未重新加载模型跑NPU矩阵。[发布检查摘要](performance/h3-image-v06-release-validation.json)分别标明新检查与既有硬件证据。原始prompt、媒体、逐样本业务输出、机器信息与私有路径不随公开仓库分发。

合并后新检查：84项FP32/BF16 SP与本地输入张量对照通过，另有完整小Transformer TP、FFN分块和VAE拼接独立验证通过；190项CPU回归及13个在线子测试通过，1项CUDA checkpoint测试跳过。GPU transport的实际CUDA执行未在本次NPU环境重测；检查了其CPU/meta拒绝合同，并保留最新v0.5实现。

## 已知范围

纯Laser产生非有限值及黑图，不在推荐方案中。混合Laser尾层回退、SP2/DiT TP2及QKV/SwiGLU/RoPE融合仅保留为显式实验选项；Triton SwiGLU在算子探针中更慢。图像CUDA与HTTP服务没有做本次硬件验证。多图编号、人物姿势与最大组合参考保真度的局限见性能报告。

## 合并结果

唯一文本冲突在README版本介绍：同时保留v0.5 GPU Align16说明和v0.6 Image入口。运行代码无冲突，未覆盖v0.5的视频执行模块、配置和测试。

## 2026-11-12：可选音视频 VC＋Ascend 融合

将 v0.5_op 的 VC ExpCast 与融合算子迁入现有视频适配层，沿用 Manager、组件调度、PHAA 和 Native-stream，不增加独立推理引擎。新能力通过 `--atten vc --h3-opts all` 显式开启，默认关闭；H3-Image 执行文件与依赖配置保留。

同机八卡 910B3、单图 R2VA 5s/10s、20 次实际 DiT，在原 Laser INT8 全并行 Native-stream 基线上 E2E 降低 10.01% / 12.22%。设备算子及媒体技术检查通过，两个时长的画面和声音已由评审者确认可接受。[性能、显存、实际命中与验证范围](performance/h3-v06-npu-vc-fusions.md) · [构建、启用与回退](H3-NPU-FUSIONS.md)。

## 2026-11-12：Image NPU Qwen3-VL 默认 W8A8

基于最新v0.6（包含上述VC融合更新）合入Image文本W8A8及TP分片兼容。Image命令行和Python加载入口在NPU上默认量化保留的50层Qwen解码器350个attention/MLP投影；embedding、norm、vision和lm_head保留原精度。CUDA默认BF16，显式 `--no-image_text_int8` 或Python `image_text_int8=False` 可回退。TP4文本原本常驻，TP1是否常驻继续由独立选项决定；不修改视频默认。

四卡910B1、89-token文本、4096×4096、六步PDD，各组加载一次并运行1次首请求＋3次复用：文本BF16/W8A8复用均值14.752/14.617秒，峰值allocated37.254/31.759 GiB每卡。主要收益是省显存，未证明稳定提速。该硬件验证发生在默认值调整前，量化及TP执行文件沿用实测实现；默认配置、显式回退、CUDA隔离及最新v0.6兼容性另做CPU回归。40项旧矩阵的文本精度仍为BF16，不将它重标为文本INT8验证。[精度范围与实测](H3-IMAGE-TEXT-INT8.md) · [部署与回退](H3-IMAGE-NPU-OPTIMIZATIONS.md)。

[本次合并验证摘要](performance/h3-image-text-w8a8-default-validation.json)：257项测试及13个服务子测试通过，9项CUDA硬件／checkpoint检查跳过；另有24项BF16 SP4本地输入张量对照、每rank12项文本TP4通信对照和DiT TP／FFN分块／VAE拼接检查通过。CPU验证使用原始RMSNorm，旧版兼容测试提供原发布Git对象；原始失败记录与修复后日志分别保留。

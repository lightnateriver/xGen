# xGen v0.6：MiniMax-H3-Image NPU

v0.6以最新v0.5提交 `ca6b6f0adcca48e523d8b81f32c9b784c577f095` 为基线，加入独立的MiniMax-H3-Image六步文生图与参考图编辑入口。合入的两个开发提交为 `cc602ebe83766e095ad464bed6015cd303b0a83f`、`268f46eb8dbf2a1d3f804526e85b923ecb7fb41a`；发布提交压缩合入代码及去敏文档。

## 部署与兼容

使用[图像专用可选Diffusers依赖](H3-IMAGE.md#环境与权重)，固定提交 `0905b667fe3f37f43b6ef9c6e05ad20636206938`。推荐[四卡W8A8常驻命令](H3-IMAGE-NPU-OPTIMIZATIONS.md)：SP4/text TP4、dense SDPA、六步AdaLN表、DiT常驻、条件缓存、refiner/vision/VAE tile并行、FFN chunk2048、RGB8输出与本地输入投影。

除NPU文本W8A8默认开启外，这些是图像入口显式启用的优化。使用 `--no-image_text_int8` 并省略其他优化开关可保留原BF16卸载路径。视频的NPU Laser INT8 Streaming默认、CUDA Sage-FP8/AdaLN默认及GPU PHAA Align16继承v0.5；Image不使用视频AdaLN表、音频VAE或MP4 Streaming。v0.6表示Git发布分支，独立Python子包沿用原有包版本。

## 已完成的硬件验证

[40项性能/shape报告](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-image-npu-40cases.md)及[CSV](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-image-npu-40cases.csv)记录固定六步、batch1、seed42、warmup0/repeat1：36项生成成功，4项真实输入校验按预期拒绝，0 OOM；28项目视通过、8项部分符合。两版共同成功35项累计推理4429.153→691.899秒，6.401×。初始化另计。

最大组合4096有效prompt token、16张8192×2048参考图、5120×4096输出完成六步，1958.130秒，单卡allocated50.009 GiB、reserved58.811 GiB。常驻小输入显存约37 GiB/卡，高于原offload约21 GiB/卡；相对BF16常驻约55 GiB/卡，W8A8减少显存。历史设备布局和首请求策略不同，不能把全方案加速归于单个算子。

该硬件矩阵针对合并前开发源码；发布时检查Image执行源码逐字节一致，并重新运行CPU分布式/配置回归，未重新加载模型跑NPU矩阵。[发布检查摘要](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-image-v06-release-validation.json)分别标明新检查与既有硬件证据。原始prompt、媒体、逐样本业务输出、机器信息与私有路径不随公开仓库分发。

合并后新检查：84项FP32/BF16 SP与本地输入张量对照通过，另有完整小Transformer TP、FFN分块和VAE拼接独立验证通过；190项CPU回归及13个在线子测试通过，1项CUDA checkpoint测试跳过。GPU transport的实际CUDA执行未在本次NPU环境重测；检查了其CPU/meta拒绝合同，并保留最新v0.5实现。

## 已知范围

纯Laser产生非有限值及黑图，不在推荐方案中。混合Laser尾层回退、SP2/DiT TP2及QKV/SwiGLU/RoPE融合仅保留为显式实验选项；Triton SwiGLU在算子探针中更慢。图像CUDA与HTTP服务没有做本次硬件验证。多图编号、人物姿势与最大组合参考保真度的局限见性能报告。

## 合并结果

唯一文本冲突在README版本介绍：同时保留v0.5 GPU Align16说明和v0.6 Image入口。运行代码无冲突，未覆盖v0.5的视频执行模块、配置和测试。

## 2026-11-12：可选音视频 VC＋Ascend 融合

将 v0.5_op 的 VC ExpCast 与融合算子迁入现有视频适配层，沿用 Manager、组件调度、PHAA 和 Native-stream，不增加独立推理引擎。新能力通过 `--atten vc --h3-opts all` 显式开启，默认关闭；H3-Image 执行文件与依赖配置保留。

同机八卡 910B3、单图 R2VA 5s/10s、20 次实际 DiT，在原 Laser INT8 全并行 Native-stream 基线上 E2E 降低 10.01% / 12.22%。设备算子及媒体技术检查通过，两个时长的画面和声音已由评审者确认可接受。[性能、显存、实际命中与验证范围](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-v06-npu-vc-fusions.md) · [构建、启用与回退](H3-NPU-FUSIONS.md)。

## 2026-11-12：Image NPU Qwen3-VL 默认 W8A8

基于最新v0.6（包含上述VC融合更新）合入Image文本W8A8及TP分片兼容。Image命令行和Python加载入口在NPU上默认量化保留的50层Qwen解码器350个attention/MLP投影；embedding、norm、vision和lm_head保留原精度。CUDA默认BF16，显式 `--no-image_text_int8` 或Python `image_text_int8=False` 可回退。TP4文本原本常驻，TP1是否常驻继续由独立选项决定；不修改视频默认。

四卡910B1、89-token文本、4096×4096、六步PDD，各组加载一次并运行1次首请求＋3次复用：文本BF16/W8A8复用均值14.752/14.617秒，峰值allocated37.254/31.759 GiB每卡。主要收益是省显存，未证明稳定提速。该硬件验证发生在默认值调整前，量化及TP执行文件沿用实测实现；默认配置、显式回退、CUDA隔离及最新v0.6兼容性另做CPU回归。40项旧矩阵的文本精度仍为BF16，不将它重标为文本INT8验证。[精度范围与实测](H3-IMAGE-TEXT-INT8.md) · [部署与回退](H3-IMAGE-NPU-OPTIMIZATIONS.md)。

[本次合并验证摘要](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-image-text-w8a8-default-validation.json)：257项测试及13个服务子测试通过，9项CUDA硬件／checkpoint检查跳过；另有24项BF16 SP4本地输入张量对照、每rank12项文本TP4通信对照和DiT TP／FFN分块／VAE拼接检查通过。CPU验证使用原始RMSNorm，旧版兼容测试提供原发布Git对象；原始失败记录与修复后日志分别保留。

## R2VA 条件准备重叠（实验性）

新增 CUDA/NPU 共用的 `--h3_condition_overlap`，默认关闭，不随全并行预设自动开启。复用原 Ref2VA 文本／参考 VAE block，条件准备完成后进入原 DiT，不改变精度、步数或 offload 策略。CPU 契约回归 251 通过、7 跳过；八卡 CUDA 五图 5／10 秒各 1 步接入、条件张量逐位对照及视频技术检查通过，不能视为完整步数质量或长期稳定性验收。NPU 真机验证仍待完成。[配置与回退](H3-CONDITION-OVERLAP.md) · [验证范围](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-condition-overlap.md)

## 2026-11-13：文档与 20 步推荐示例

首页精简为推荐方案、安装、离线／在线启动与性能摘要；完整入口见[文档目录](README.md)，旧性能表迁到[历史汇总](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-historical-overview.md)，原始数值和计时／精度状态保留。修正旧文档中 CUDA/NPU 默认、Token Refiner 参数默认与预设值、在线验证范围的过时描述。

当前推荐视频命令显式选择 20 次实际 Base DiT。CUDA Base、MaaS 和 NPU Refiner 示例 JSON 改为 20 步，Refiner 示例的 Base 表路径同步改为 21 点版本；NPU Base／Ref2VA 与 CUDA Ref2VA 示例原本已为 20 步。2K Refiner 仍为独立 3 步，H3-Image 仍为 6 步。**没有修改运行实现、低层默认参数或历史 50 步能力，也没有重新跑性能矩阵。** 已部署配置不会被自动重写，更新示例时须准备匹配资产并重启服务。

本轮完成 287 个仓库内链接及标题锚点检查、251 项 CPU 回归（7 项跳过）、22 条模拟启动命令和 6 份推荐 JSON 的 20 步配对检查。模拟命令拦截 torchrun，不加载权重或执行 GPU/NPU 推理；不能代替硬件回归。原生 VAE 下载目录另已核对官方固定版本，其余既有外部链接未做全量重新访问。历史首页性能表按原数值和原计时／验收说明迁移，未覆盖底层报告或 CSV。

## 2026-11-13：NPU 768P Base 默认 VC 全并行

按当前推荐方案，将 `run_h3.sh npu ... 768p`、`h3_npu_base.example.json` 和 `h3_npu_ref2va.example.json` 默认选择改为 VC＋Ascend 融合：`atten=vc`、`h3_opts=all`。保留 20 步、SP8／文本 TP8、INT8 Linear、PHAA2、AdaLN 表、full-parallel 与 Native-stream；不改变算子数值实现。CUDA、H3-Image、2K 及低层 CLI 的未指定默认保持原样。上方较早发布记录中的“默认关闭／保持旧默认”仅描述对应历史提交。

首页以当前 VC 方案为 NPU 推荐入口；旧 attention 的性能继续作为历史对照，不能重标。当前硬件性能和人工视听认可仍仅覆盖八卡 910B3、单图 R2VA 5／10 秒、20 步；本次默认切换不等于补齐其他场景硬件验证。启动前需安装 VC／融合库并核查命中日志，独立关闭和显式回退见 [NPU 默认方案](H3-NPU-DEFAULT.md)。Benchmark 的配置记录同步区分 768P VC／融合与 2K 旧配方。

默认切换的 CPU 回归为 297 项通过、7 项跳过，另做 22 条模拟启动和 6 份推荐 JSON 检查。覆盖默认选择、显式历史回退、融合列表／单项覆盖、真实 HTTP 参数构造及 CUDA／2K 隔离。Benchmark 的 3 项独立 CPU 测试检查忙卡预检不会启动进程、配置记录与旧报告字段缺失兼容；原始性能 CSV 与算子数值实现未改变。本轮不加载模型或重新执行 GPU/NPU 硬件性能测试。

## 2026-11-13：性能文档收敛为一张总表

`docs/performance/` 仅保留[当前视频性能总表](performance/README.md)，汇总 8 卡、20 步、768P 的 5090 Sage-FP8 全并行 Streaming 和 NPU VC INT8 全并行 Streaming。按 T2VA/I2VA/R2VA 单图／五图、5/10/15 秒列出已有结果与待测项，不能用旧配方填补缺项。过程 Markdown、CSV 和验证快照从当前目录移除，历史证据保留在提交 `6822583e0021e3ca64701d7c8aa49789a0d55f1c`；关联引用已转为当前总表或固定提交链接。此项仅整理文档，不改运行代码或重跑推理。

## 2026-11-13：按指定 910B2 来源修正总表

NPU 列改用提供的性能对照表“性能对照”工作表中 `B2:B13`（910B2 截图端到端）及 `C2:C13`（去噪），补齐 12 项，并移除混入的 910B3 显存与缺项判断。设备名称按原表记录；去噪保持原指标名称。5090 按本次确认继续采用最新在线 Align16 结果，不混入另附的 owner／table 离线矩阵。原表没有的显存、预热、配置日志与人工验收不补造；本次没有重新推理或修改运行代码。

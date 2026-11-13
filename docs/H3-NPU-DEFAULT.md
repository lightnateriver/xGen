# NPU Base 默认：VC＋Ascend 融合全并行 Streaming

v0.6 推荐的 NPU **768P Base** 入口使用 **VC attention＋Ascend 融合＋INT8 DiT＋全并行 Streaming＋AdaLN 表**。离线 `run_h3.sh npu` 和在线 Base／Ref2VA 示例 JSON 对齐，默认 20 次实际 Base DiT。CUDA、H3-Image、2K Refiner 及未使用推荐入口的低层 CLI 默认不由本页改变。

## 默认配置

| 配置 | 值 |
|---|---|
| 进程、DiT SP、文本 TP | 8 / 8 / 8 |
| Base 实际计算步数 | 20，匹配 21 点 AdaLN 表 |
| attention / DiT Linear | VC ExpCast / INT8 |
| Ascend 融合 | `h3_opts=all`，请求开启；实际命中由形状与已安装算子决定 |
| PHAA / RoPE | 2（4+3 头）/ fused |
| 调度 | `h3_features=full-parallel`，50 层 DiT 常驻 |
| 输入 Token Refiner | SP8、已选 NPU attention、BF16 FFN；R2VA chunk2048，T2VA/I2VA chunk0 |
| 条件处理 | 本地输入／输出投影、请求内条件复用、Qwen 视觉 SP、VAE tile 编码、音频并行 |
| 视频 decoder | `native-stream`，`vae_matmul_quant=none`；Linear FP16 存储、FP16 autocast，其余原生参数 FP32 |

固定条件在同一请求的多个 DiT 步骤间复用，结束后释放，不是跨请求内容缓存。有图／视频时才运行视觉与 VAE 编码，有参考音轨时才运行 Audio Encoder，长音频 Decoder 并行仅在支持的形状生效。T2VA/I2VA 共用 FL2VA 模型与对应条件表，R2VA 使用独立模型及表。

## 安装与模型准备

先完成 [VC 与 Ascend 融合算子构建](H3-NPU-FUSIONS.md)、[20 步模型资产准备](H3-VIDEO-QUICKSTART.md#准备模型和输入)及[原生 video_vae 下载](H3-MODELS.md#下载原生视频-vae)。`video_vae` 与转换后的 Diffusers `vae` 不同。

`all` 不代表九项全部执行。SP8／PHAA2 不满足部分专用布局；部分条件预计算也会复用现有缓存。检查 `[VC_ATTENTION] enabled=expcast`、`[H3_OPT] enabled=`、请求 `summary=` 和 `fallback=`，确认真实路径。VC 缺失或初始化失败会警告并回退 SDPA；融合库缺失或形状不支持时保留原算子。回退结果不能当作 VC／融合性能，执行期算子错误不会被自动重试掩盖。

## 离线

下面示例生成单图 R2VA。先设置工作流对应资产：

```bash
export BASE_MODEL=weights/base
export BASE_TRANSFORMER=model-cache/base-ref2va
export BASE_ADALN=model-cache/ref2va-21step.safetensors
export H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export PROMPT_FILE=inputs/prompt.txt
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png
```

仅对可信模型代码授权。T2VA/I2VA 换用 FL2VA 的 Transformer 与表，完整命令见[视频教程](H3-VIDEO-QUICKSTART.md)。脚本默认传 `--atten vc --h3-opts all`；额外参数放在 `--` 后，可覆盖 attention、融合列表和单项开关。

`TASK_QUEUE_ENABLE=2`、`HCCL_BUFFSIZE=512`、`OMP_NUM_THREADS=8`、`COMBINED_ENABLE=1`、`MEMORY_FRAGMENTATION=1`、`INF_NAN_MODE_FORCE_DISABLE=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`ENCRYPTION_WEIGHT=false` 沿用既有配方，显式环境值优先。`H3_BASE_STEPS=50` 仍可使用，但必须准备匹配的 51 点表，不沿用本页 20 步性能。

## 在线

使用 `h3_npu_base.example.json` 部署 T2VA/I2VA，`h3_npu_ref2va.example.json` 部署 R2VA。两份示例均已设置 `"atten":"vc"`、`"h3_opts":"all"`、20 步及 native-stream。先填写本地权重、原生 VAE 和配对的 AdaLN 路径，设置上述环境变量，再按[在线指南](H3-ONLINE.md#npu-base)启动。

服务外层 CLI 没有离线 `--atten`／`--h3-opts` 参数；使用启动 JSON 设置。改变配方后重启服务。现有性能来自常驻推理与保存，不是 HTTP SLA。

## 独立关闭与回退

保留 VC，关闭新增融合：

```bash
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- --h3-opts none
```

显式回退历史 attention 和原算子：

```bash
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- --atten laser --h3-opts none
```

在线回退在 JSON 中设置 `"atten":"laser"`、`"h3_opts":"none"` 后重启。只关闭某一项可使用 CLI `--no-h3-opt-rope` 或 JSON `"h3_opt_rope":false`；列表与单项的优先级见[融合开关](H3-NPU-FUSIONS.md)。推荐入口显式传入 `all`，所以单独设置 `H3_OPT_FEATURES` 不能覆盖它。

VAE 仍可独立选择 legacy／lightning／native-stream，不与 attention 绑定；回退旧 VAE 时移除原生路径及授权，按[VAE 指南](H3-OMNI-VAE.md)配对量化选项。2K 使用独立协调器与原有配方，不继承本页 VC／融合默认；Base 20 步与 Refiner 3 步分别统计。H3-Image 使用自己的入口和默认设置。

## 性能与验收范围

| 八卡 910B3，单图 R2VA | VC＋融合常驻 E2E | DiT | peak allocated GiB/卡 |
|---|---:|---:|---:|
| 5s / 768P / 20 步 | 77.202 s | 62.207 s | 31.121 |
| 10s / 768P / 20 步 | 170.123 s | 151.042 s | 31.183 |

每项一次匹配预热及一次正式请求，不含初始化；输入为 2560×1440 单图、固定文本及 seed42。样例画面与声音已验收，当前数据和验收范围见[性能总表](performance/README.md)。T2VA/I2VA、多图、15 秒及更多媒体组合的当前 VC 整网性能与人工精度须分别验证；不能将旧 attention 的结果重新标为 VC。

旧全并行性能和 VAE 存储验证保留在 Git 历史，当前目录只展示最新视频方案总表。本次只调整推荐启动入口与示例配置，没有改变 VC 或融合算子的数值实现，也没有重新运行硬件性能矩阵。CPU 配置、回退和模拟启动检查不替代 NPU 成片验收。

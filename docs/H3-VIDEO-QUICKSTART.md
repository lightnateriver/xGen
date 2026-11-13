# H3 音视频：当前配方与启动

从仓库根目录执行，以下推荐示例统一使用 **20 次实际 Base DiT**。768P 支持 T2VA、I2VA 和 R2VA，离线脚本时长可填 `5`、`10`、`15`。2K 是独立的 Base → Refiner 链路，另做 3 次 Refiner。

| 设备与用途 | 当前配方 | 进一步说明 |
|---|---|---|
| 八卡 RTX 5090，768P Base | Sage-FP8 全并行、BF16 AdaLN 表、Native-stream；SP8、文本 FP8 TP8、PHAA2、DiT 全常驻 | [CUDA 配方与单项开关](H3-CUDA-COMPOSABLE.md) |
| 八卡昇腾，768P Base | VC＋Ascend 融合、DiT INT8、AdaLN 表、全并行、Native-stream；SP8、文本 TP8、PHAA2 | [NPU 默认方案](H3-NPU-DEFAULT.md) |
| 八卡 910B3，已评审范围 | 单图 R2VA 5s/10s、20 步 | [性能与验收范围](performance/README.md) |
| 八卡昇腾，2K | 20 次 Base DiT＋3 次 Refiner，独立 legacy VAE 链路 | [2K 资产与部署](H3-DEPLOYMENT.md) |

CUDA 当前组合已有 12 项真实在线 20 步技术验证，完整动态、口型和听音人工复核待完成。NPU VC 的画面与声音认可仅覆盖已评审的单图 R2VA 5s/10s 样例。详细结果见[性能索引](performance/README.md)，不要把不同设备或历史配方的数值直接比较。

## 准备模型和输入

准备 Linux、八张可用设备、匹配的 PyTorch/CUDA 或 CANN/torch-npu，以及支持 H.264/AAC 的 FFmpeg 和 FFprobe。CUDA Sage 配方需要兼容的 SageAttention、Triton、Diffusers 和 Transformers；NPU 默认方案需先安装 VC 和融合算子，见[NPU 构建说明](H3-NPU-FUSIONS.md)。

```bash
python -m pip install --no-deps -e ./x_base -e ./x_diffusers
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
python tools/collect_env.py
```

模型下载入口见[模型与资产](H3-MODELS.md)。这里统一使用下面的示例路径：

```text
weights/base/                  完整 H3 模型根目录，含 modular_model_index.json
  transformer/                 T2VA / I2VA 的原始 FL2VA Transformer
  transformer_ref/             R2VA 的原始 Ref2VA Transformer
  …                            文本编码器、处理器、VAE、音频 VAE、调度器等组件
weights/native/FL2VA/video_vae/ 独立的原生视频 VAE，含可信模型代码及权重
inputs/prompt.txt              UTF-8 提示词
inputs/first-frame.png         I2VA 首帧
inputs/ref01.png               R2VA 参考图，可按顺序追加更多参考文件
```

`video_vae` 不是 Diffusers 的 `vae` 子目录。只有确认原生模型代码可信后才开启下面的授权。完整模型根目录和工作流对应的剥离 Transformer、AdaLN 表都要保留；权重和输入素材不随源码发布。

```bash
export BASE_MODEL=weights/base
export PROMPT_FILE=inputs/prompt.txt
export H3_VAE_MODEL_PATH=weights/native/FL2VA/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
export H3_BASE_STEPS=20
```

`H3_BASE_STEPS=20` 是本页的显式选择。底层 CUDA/2K 启动脚本仍保留原来的 50 步默认；NPU 768P 脚本默认 20 步。切换步数时必须同时换用匹配的 AdaLN 表，不能仅改文件名。

### CUDA：一次性准备 AdaLN 资产

按 [CUDA AdaLN 教程](H3-CUDA-ADALN.md#一次性准备)，分别从 `weights/base/transformer` 和 `weights/base/transformer_ref` 生成 **20 步**资产。将教程中的输出位置对应到：

```text
weights/base/xgen-adaln/fl2va/transformer/
weights/base/xgen-adaln/fl2va/tables/adaln_fl2va_bf16_20step.safetensors
weights/base/xgen-adaln/ref2va/transformer/
weights/base/xgen-adaln/ref2va/tables/adaln_ref2va_bf16_20step.safetensors
```

T2VA/I2VA 共用 FL2VA 资产，R2VA 使用 Ref2VA 资产。脚本按任务与实际步数选择表，缺失或不匹配时明确报错。自定义目录、owner 回退及精度范围见同一教程。

### NPU：一次性准备 20 步资产

在已加载 CANN 环境的目标 NPU 机器上执行。工具使用 NPU 0；输出 Transformer 目录须为新目录。20 次实际计算对应此工具的 **21 点表**，两种工作流分别准备：

```bash
export XGEN_DEVICE=npu
python -m x_diffusers.framework.pipeline.h3_regenerate.prepare_base_cache \
  --source weights/base/transformer --model-variant T2VA --steps 20 \
  --output model-cache/t2va-21step.safetensors \
  --stripped-output model-cache/base-t2va

python -m x_diffusers.framework.pipeline.h3_regenerate.prepare_base_cache \
  --source weights/base/transformer_ref --model-variant Ref2VA --steps 20 \
  --output model-cache/ref2va-21step.safetensors \
  --stripped-output model-cache/base-ref2va
```

第一组同时用于 T2VA/I2VA；CUDA 与 NPU 的表工具及资产布局分别管理，勿混用。原 [2K 部署教程](H3-DEPLOYMENT.md#43-一次性预处理) 中的 50 步示例是可选历史配置，本页使用上面的 20 步资产。

## 离线生成 768P

每条命令单独启动一个八卡任务，输出写入 `outputs/`。`PROMPT_FILE` 和 `OUTPUT_PATH` 可分别覆盖提示词与输出文件；每次运行前确认使用上面的公共环境变量。

### CUDA / RTX 5090

```bash
bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
bash example/diffusers/scripts/run_h3.sh cuda i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh cuda r2va 15 768p inputs/ref01.png
```

这三类任务都可选择 5/10/15 秒；`cuda-5090` 与 `cuda` 在 Base 768P 的默认配方相同。R2VA 多参考输入按顺序追加文件即可。该配方默认包含 PHAA FP8 Align16，不自动选择 NVFP4、稀疏 attention 或实验 TP/CP。

### NPU

推荐启动环境如下；离线脚本会为未显式设置的变量填入这些值，在线启动需自行设置：

```bash
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false OMP_NUM_THREADS=8 HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True INF_NAN_MODE_FORCE_DISABLE=1
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p
bash example/diffusers/scripts/run_h3.sh npu i2va 10 768p inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh npu r2va 15 768p inputs/ref01.png
```

NPU 默认 VC＋Ascend 融合全并行，DiT Linear 为 INT8，VAE `matmul_quant=none`，不是 CUDA FP8 配方。R2VA 输入 Token Refiner 使用 chunk2048，T2VA/I2VA 使用 chunk0；无需手工拼接这些默认参数。

默认已经选择 VC＋Ascend 融合，需先按[构建说明](H3-NPU-FUSIONS.md)安装。需要保留 VC、独立关闭新增融合时：

```bash
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- \
  --h3-opts none
```

已评审的性能与质量范围为八卡 910B3、单图 R2VA、5s/10s、20 步。`all` 是请求开启各项，实际命中由形状、后端及已安装库决定；看各 rank 的命中与回退日志。需要回退历史 attention 和原算子时用 `--atten laser --h3-opts none`。详见[同机性能与验收范围](performance/README.md)。

## 2K：独立 Base → Refiner 链路

NPU 使用 **20 次 Base DiT＋3 次 Refiner**。先按[2K 资产准备](H3-DEPLOYMENT.md#4-模型与目录准备)取得完整 Refiner 源包、Sharp 表并完成预处理，再运行：

```bash
export REFINER_WEIGHTS=weights/refiner
export REFINER_PREPARED=model-cache/refiner
export H3_BASE_STEPS=20
bash example/diffusers/scripts/run_h3.sh npu t2va 5 2k
bash example/diffusers/scripts/run_h3.sh npu i2va 10 2k inputs/first-frame.png
bash example/diffusers/scripts/run_h3.sh npu r2va 15 2k inputs/ref01.png
```

脚本将 20 步及对应工作流的 21 点表传给现有 Refiner 入口，Refiner 固定另做 3 步。这里沿用 legacy VAE 和独立协调器；768P Native-stream/全并行的性能结论不套用于 2K。可选时长为 5/10/15 秒，功能范围与各项实测范围分别见部署教程。

CUDA 2K 为[实验性 I2VA 路线](H3-CUDA.md#实验性-cuda-2k-refiner)，需要匹配的 NVFP4 Base 文件、FL2VA 组件及 CUDA 预处理的 Refiner；不能直接套用上面的 CUDA 768P 资产或推断其性能。T2VA/R2VA 2K 使用 NPU 路线。

## 在线 Base 与 CUDA MaaS

先复制相应示例 JSON，填写本地模型及可信原生 VAE 路径。当前推荐示例均使用 20 次实际 Base DiT；CUDA JSON 选择对应表目录，NPU JSON 使用上述 21 点表。公共 shell 变量不会自动替换 JSON 的路径。配置变更后重启服务，每次选择一个实例启动。

```bash
# CUDA：T2VA / I2VA 共用实例
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001

# CUDA：R2VA 独立实例
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode ref2va \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json --host 127.0.0.1 --port 5001

# NPU：先设置上面的 NPU 启动环境
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

NPU R2VA 将最后一条的服务模式改为 `ref2va`，配置改为 `h3_npu_ref2va.example.json`。两份 NPU 示例 JSON 已默认设置 `"atten": "vc"`、`"h3_opts": "all"`；改变配方后需重启。不要将离线的 `--atten`／`--h3-opts` 直接追加到服务启动命令。其真实 HTTP 性能需单独验证。

普通 Base 使用 `/generate`、`/status/<task_id>`、`/download/<task_id>`。T2VA/I2VA 实例的 `t2va` 模式同时接受两类请求，R2VA 单独部署。请求示例、4～15 秒整数时长及媒体限制见[在线指南](H3-ONLINE.md)；20 步部署的请求也填 20，或省略步数使用匹配的配置/表。

CUDA MaaS 使用同一模型配方和 20 步启动配置，先安装 `server/requirements-maas.txt`：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode t2va \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

R2VA MaaS 使用独立的 `ref2va` 服务模式与 `h3_cuda5090_ref2va.example.json`。提交一条真实请求预热，等 `/ready` 返回 HTTP 200 后再接流量；V2 接口与兼容入口见 [MaaS API](../server/video/MAAS-API.md)。NPU 目前使用普通 HTTP；MaaS 的 IR/2K 为显式开启的模拟联调后端，不代表真实 Refiner。

2K 在线使用 `h3_refiner.example.json`：Base 20 步、匹配 21 点表、Refiner 3 步；每个实例服务一种任务。I2VA 切换为 `fl2va`，R2VA 切换为 `ref2va` 并配对 Ref2VA Transformer 和表，详见[2K 在线配置](H3-ONLINE.md#2k-refiner)。

### 首尾帧与常驻复用

首尾帧使用 FL2VA 服务与同一套 20 步资产。在服务端可读取素材的前提下，用 `/generate` 提交：

```json
{"task_type":"fl2va","prompt":"The subject walks towards the camera.","duration":5,"resolution":"768p","num_inference_steps":20,"first_frame":"inputs/first-frame.png","last_frame":"inputs/last-frame.png"}
```

提交、查询和下载命令见[在线请求](H3-ONLINE.md#提交原服务请求)；MaaS 的首／尾帧角色与比例规则见 [API](../server/video/MAAS-API.md)。接口支持与具体样例的人工画面验收分开记录。

前面的 shell 命令每次冷启动。连续服务请求则复用同一工作流的常驻模型；低层可复用 `InferenceManager`，示例见[同一 Manager 只初始化一次](H3-DEPLOYMENT.md#7-连续离线请求同一-manager-只初始化一次)。改变权重、精度或启动布局时需新建实例；性能测试分别统计初始化、匹配预热和正式请求。

## 验证范围与可选功能

离线冷启动命令包含进程启动和模型加载；常驻模型预热后的推理耗时、HTTP 延迟和下载耗时是不同口径。各配方的硬件、步骤、显存与质量范围以[性能记录](performance/README.md)中的具体表为准。

[条件准备重叠](H3-CONDITION-OVERLAP.md)仍默认关闭，只用于 Base Ref2VA 原生 SP；现有 CUDA 单步验证证明集成与容量，不代表完整 20 步质量或稳定提速，NPU 设备验证待完成。其他精度回退、单项组件优化和实验布局分别见 [CUDA 配方](H3-CUDA-COMPOSABLE.md)、[NPU 默认](H3-NPU-DEFAULT.md)及[版本历史](H3-VERSIONS.md)。

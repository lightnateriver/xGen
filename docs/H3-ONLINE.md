# MiniMax-H3 在线服务

> 八卡 CUDA H3 Base 推荐 Sage-FP8 全并行 Streaming＋BF16 AdaLN 表，NPU 推荐 VC＋Ascend 融合 INT8 全并行 Streaming。当前推荐示例 JSON 统一为 **20 次实际 Base DiT**；需先[准备匹配资产](H3-VIDEO-QUICKSTART.md#准备模型和输入)。目录模式按步数选表，固定表必须匹配；配置更改后重启服务。

从仓库根目录执行。先按 [权重下载](../README.md#h3-权重下载) 准备模型和素材，并用 `python tools/collect_env.py` 确认当前环境能看到八张卡。以下示例绑定本机，默认端口为 5001。每次选择一个服务启动。

## 选择服务

R2VA Base 可在服务启动时显式加 `--h3_condition_overlap`，或在启动 JSON 中设置 `"h3_condition_overlap": true`；CUDA/NPU 使用相同配置，默认均关闭。它是启动选项，修改后需重启，不能按生成请求切换。配套 `h3_condition_overlap_timeout` 默认600秒，仅约束条件分支等待，不是生成超时。适用边界、回退和 NPU 待实测状态见[说明](H3-CONDITION-OVERLAP.md)。

| 服务 | 任务 | 分辨率 | 请求时长 |
|---|---|---|---|
| NPU Base | T2VA、I2VA、R2VA | 768P | 4 至 15 秒，接受整数 |
| NPU Refiner | T2VA、I2VA、R2VA | 2K | 5、10 或 15 秒 |
| CUDA Base Sage-FP8 全并行 Streaming | T2VA、I2VA；独立 Ref2VA 实例服务 R2VA | 768P | 接口接受 4 至 15 秒；已有 5/10/15 秒、单图/五图的 12 项在线技术验证 |
| CUDA 保留的 fp8-bf16 / Sage 基础配方 | T2VA、I2VA；资产和支持范围见各教程 | 768P | 接口接受 4 至 15 秒；旧配方证据不等于当前全并行性能 |
| CUDA Refiner | I2VA | 2K | 接口接受 5、10 或 15 秒，已有 5 和 10 秒结果 |
| CUDA MaaS V2 | T2VA、首尾帧 I2VA、独立 R2VA | 真实 768P；2K 仅模拟 | 4 至 15 秒整数 |

Base 服务生成 15 秒视频时，在请求里设 `"duration":15`。模型按 VAE 要求生成 362 帧，服务保存 360 帧和对应的音频。历史性能表里的 15 秒在线 768P 来自独立批测程序。

所有 H3 在线入口的 `prompt` 最多 7,000 个 UTF-16 码元，包括 Base、Refiner 和 MaaS 模式。计数与 [MiniMax 官方 CLI](https://github.com/MiniMax-AI/cli/blob/main/src/video/v2.ts) 的 JavaScript `text.length` 一致：常见中文、英文字母、空格或换行各计 1，`🙂` 计 2。因此 7,000 个常见汉字或 3,500 个 `🙂` 达到上限，再加一个英文字母就超限。服务在 JSON 解码后、分词和入队前检查原文，超长拒绝，不截断；`/params` 用 `length_unit=utf16_code_units` 标明口径。这不是 7,000 token，也不包含图片、视频、音频产生的 token。

MaaS V2 使用官网错误信封，旧 MaaS 接口返回 HTTP 400、`error=invalid_input`、`field=prompt`；普通 `/generate` 返回 HTTP 400，原 `/v1/tasks` 保留 HTTP 200、任务 `failed` 的错误契约。Wan 的 2,000 字符限制及计数方式保持不变。

原 Base 服务的5/10秒保存保留原生padding，例如本轮5秒请求保存124帧；v0.4保持这一行为。功能测试按各接口实际帧数契约检查，不把所有视频一律断言为时长×24帧。

每条 H3 视频完成保存后，各 rank 会释放请求输出引用，同步设备并通过 PyTorch 回收空闲显存；常驻模型保留。日志 `H3_REQUEST_MEMORY_RELEASE` 给出耗时和前后 allocated/reserved。该清理也用于离线入口，不能代替单条请求内部的峰值显存验证。

## NPU Base

v0.6 推荐入口默认 **VC＋Ascend 融合 INT8 全并行 Streaming**，T2VA/I2VA/R2VA均包含SP8/text TP8/PHAA2、AdaLN查表、实际20步、通用full-parallel和Native-stream。T2VA/I2VA FFN chunk0，R2VA另使用chunk2048；无图时视觉编码不执行，长音频并行覆盖10/15秒形状。先在示例JSON配置可信原生`video_vae`路径。需先[安装 VC 和融合库](H3-NPU-FUSIONS.md)。当前[实测报告](performance/README.md)覆盖单图 R2VA 5/10 秒，其他形状与 HTTP 性能须单独验证；历史结果不重标为 VC。

启动前使用 `TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 MEMORY_FRAGMENTATION=1 ENCRYPTION_WEIGHT=false OMP_NUM_THREADS=8 HCCL_BUFFSIZE=512 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；参数含义与回退见[默认方案](H3-NPU-DEFAULT.md)。

两份 NPU Base／Ref2VA JSON 均已配置 `"atten":"vc"` 和 `"h3_opts":"all"`；需要回退时修改 JSON 并重启，不把离线的 attention／融合参数直接加到服务外层 CLI。

先按 [模型准备](H3-DEPLOYMENT.md#4-模型与目录准备) 生成 Base Transformer 和 AdaLN 表。下面的服务接收 T2VA 和 I2VA。

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

R2VA 单独启动。使用 `example/diffusers/configs/h3_npu_ref2va.example.json`，修改模型和VAE路径；该示例已配置Ref2VA权重、21点AdaLN表和输入Refiner FFN chunk2048。启动时将 `--config` 指向此文件，将 `--service-mode` 改为 `ref2va`。

## CUDA Base

GPU 的 Sage-FP8 全并行 Streaming 预设默认开启每 peer 16字节包对齐。关闭时在启动前设置 `export XGEN_PHAA_FP8_ALIGN16=0`，所有 rank 必须一致，修改后重启服务。配置与回退见[Align16指南](H3-PHAA-FP8-ALIGNMENT.md)，八卡20步T2VA/I2VA/R2VA的12项真实在线结果见[性能对比](performance/README.md)。

默认配置为 [Sage-FP8 全并行 Streaming + AdaLN 表](H3-CUDA-ADALN.md)。准备完整 Base 模块、相应 FL2VA／Ref2VA 表及剔除权重后的 Transformer。推荐 `--service-mode t2va`；`t2va` 与 `fl2va` 两种服务模式均使用 FL2VA 模型，同时接受 T2VA / I2VA。R2VA 独立使用 ref2va 模式和对应资产。保留的 [fp8-bf16 JSON](../example/diffusers/configs/h3_cuda_bf16.example.json) 可直接回退原精度方案；改变配方需重启服务。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001
```

### CUDA 标准方案二：Sage-FP8

以下为保留的 Sage 基础方案，显式关闭全并行预设和表。复用相同服务接口，采用 Sage attention、FP8 通信、输出调制融合和 expandable segments。修改 [Sage 配置](../example/diffusers/configs/h3_cuda_sage.example.json) 中的模型目录后，从新终端启动：

```bash
XGEN_DEVICE=cuda XGEN_PHAA_FP8=1 XGEN_H3_FUSED_OUTPUT_ADALN=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 NCCL_MAX_CTAS=2 \
torchrun --standalone --nproc_per_node=8 server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_sage.example.json --host 127.0.0.1 --port 5001
```

这是尚未完成在线端到端实测的部署示例；离线六项成片与性能见 [方案二教程](H3-CUDA-SAGE.md)。precision 只能在启动时选择，切换后须重启。MaaS 可复用此 JSON 和环境变量，将服务模式参数换为 `--maas`；输入和素材限制见 MaaS API，不能将历史离线 20 步性能当作新 MaaS 组合的性能。

### RTX 5090 R2VA 全并行

此方案复用原 Base HTTP 服务，面向八卡 RTX 5090；R2VA 的 MaaS 输入使用参考媒体协议。已有[全 12 项在线技术验证](performance/README.md)，完整人工动态／口型／听音复核及更多媒体组合、长期压测仍按各自状态记录。先修改 JSON 中的模型路径，再启动：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 server/video/video_server.py --init --service-mode ref2va --config example/diffusers/configs/h3_cuda5090_ref2va.example.json --host 127.0.0.1 --port 5001
```

使用下文 R2VA 的 `/generate` 请求。`duration` 可填4至15的整数，`ref_path` 使用服务能够访问的参考文件，`num_inference_steps`填实际DiT步数。单项优化、并行布局和精度由服务启动时确定，生成请求不能更改；更改启动配置后重启服务。常驻请求可以更换prompt、时长和参考素材，请求内条件结果不会跨请求复用。

初始化、生成请求和HTTP下载是不同计时边界；离线表的生成保存耗时不能作为HTTP端到端延迟。[配置与回退](H3-CUDA-COMPOSABLE.md) · [功能测试和性能](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-cuda5090-full-parallel.md)

### 历史 NVFP4 的 15 秒 I2VA

八卡实测使用打包 NVFP4 权重和分层加载。复制 [历史 NVFP4 模型配置](../example/diffusers/configs/h3_cuda_nvfp4_legacy.example.json) 为 `config-cuda-base-15s.json`，把 `h3_nvfp4_resident` 改成 `false`。在新终端执行下列命令，使用原 Base API。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XGEN_H3_CUDA_VAE_TILES=1 XGEN_PHAA_FP8=1 XGEN_PHAA_GATHER_PRIORITY=1
export XGEN_PHAA_DEVICE_JOIN=1 XGEN_PHAA_ASYNC_EXCHANGE=1
export XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1 NCCL_MAX_CTAS=2
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config config-cuda-base-15s.json --host 127.0.0.1 --port 5001
```

用 I2VA 请求示例，将 `duration` 设为 15。设 `num_inference_steps` 为 1 可以跑 1 次 DiT，适合检查生成和下载链路。本配置已通过 1 次 DiT 的 15 秒音视频验证。全驻留配置在同一用例的 VAE 解码阶段显存不足。

## H3 视频解码后端

CUDA/NPU 共用 `--h3_vae_backend legacy|lightning|native-stream`。CUDA 推荐的 T2VA/I2VA/R2VA 配置默认 native-stream，示例 JSON 已包含原生 `video_vae` 路径及授权；使用前必须改为本地可信模型目录。v0.5 NPU Base在线默认同样为Native-stream。后端只在服务启动时选择，生成请求不能切换。

也可在原启动命令后追加：

```bash
--h3_vae_backend native-stream \
--h3_vae_model_path weights/h3-native/video_vae \
--h3_vae_trust_remote_code --vae_matmul_quant none
```

显式 CLI 覆盖 JSON，`none` 可关闭 NPU 配置中的 VAE INT8。回退旧后端时，将配置改为 legacy 或 lightning，并移除原生 VAE 路径和授权后重启；保留 DiT 的原有量化与并行参数。模型暂存、初始化成本及验证范围见 [VAE 后端](H3-OMNI-VAE.md)。

## 2K Refiner

NPU T2VA 可直接使用 [Refiner 配置](../example/diffusers/configs/h3_refiner.example.json)。

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29558 \
  server/video/video_server.py --init --config example/diffusers/configs/h3_refiner.example.json \
  --host 127.0.0.1 --port 5001
```

每个 2K 实例只服务一种任务。I2VA 把配置的 `task_type` 改为 `fl2va`。R2VA 改为 `ref2va` 并同时替换 Transformer 和 AdaLN 表。CUDA 2K 使用 FL2VA 打包 NVFP4 Base 和 CUDA 预处理后的 Refiner。 [CUDA 2K 配置与命令](H3-CUDA.md#实验性-cuda-2k-refiner)

## 提交原服务请求

使用 `/generate` 提交。先确认服务能响应：

```bash
curl -fsS http://127.0.0.1:5001/params
```

选择下面对应任务的 JSON，保存为 `request.json`。本地路径相对于服务的启动目录，远程客户端需要用服务可读取的素材。2K 请求把 `resolution` 改为 `2k`。`num_inference_steps` 直接填 Base DiT 次数。有 AdaLN 表时，请求值须与表内记录的执行次数一致；省略则从表内读取。没有 AdaLN 表时，省略则使用启动配置的步数，低层默认 50；本页推荐示例 JSON 均显式配置为 20。

**T2VA**

```json
{"task_type":"t2va","prompt":"A small robot walks through a garden.","duration":5,"resolution":"768p","num_inference_steps":20}
```

**I2VA**

```json
{"task_type":"fl2va","prompt":"The camera follows the moving subject.","duration":5,"resolution":"768p","num_inference_steps":20,"first_frame":"inputs/first-frame.png"}
```

**R2VA**

```json
{"task_type":"ref2va","prompt":"The characters walk together.","duration":5,"resolution":"768p","num_inference_steps":20,"ref_path":["inputs/ref01.png","inputs/ref02.png","inputs/ref03.png","inputs/ref04.png","inputs/ref05.png"]}
```

```bash
curl -fsS -H 'Content-Type: application/json' --data-binary @request.json http://127.0.0.1:5001/generate
```

把返回的 `task_id` 复制到 `TASK_ID`。查询状态到 `completed` 后下载。

```bash
TASK_ID=replace-with-task-id
curl -fsS "http://127.0.0.1:5001/status/$TASK_ID"
curl -fsS -o result.mp4 "http://127.0.0.1:5001/download/$TASK_ID"
```

## CUDA MaaS

MaaS 是平台层，V2 是 HTTP 协议，`--maas` 是 xGen 的服务能力开关，CUDA/NPU 是计算后端。当前只有 CUDA 启动路径开放了 `--maas`；V2 协议本身不绑定 CUDA。NPU 仍使用上文旧 HTTP/离线入口，尚未接入本节 V2 服务。详见 [分层与启用范围](H3-MAAS-V2.md#接口平台与计算后端)。

准备完整 FL2VA 模块、可信原生 VAE 及 [20 步 Sage-FP8 全并行表资产](H3-CUDA-ADALN.md)。示例 JSON 已设为 20 步，修改其中的本地路径后启动：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

MaaS 推荐使用官网格式的 `POST /v2/video_generation`、`content[]/role/ratio` 和 `GET /v2/query/video_generation/<task_id>`。旧 `/v1/tasks` 及 `/generate` 保持兼容。每副本配置独立 `--maas-state-dir`，先提交一条真实预热请求，等 `/ready` 返回 HTTP 200 再接流量。IR/2K 默认关闭，仅可通过 `--maas-mock-backends` 启用模拟联调。 [预热和高性能开关](H3-CUDA.md#maas-在线服务) · [MaaS API](../server/video/MAAS-API.md)

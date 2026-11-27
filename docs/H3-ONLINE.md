# H3 在线服务

[文档目录](README.md) · [模型资产](H3-MODELS.md) · [配置规则](H3-CONFIGURATION.md#priority)

服务复用原 Manager、Pipeline 和模型装配。先复制相应示例 JSON，填写本地模型、可信原生 VAE 和配对 AdaLN 路径；shell 变量不会自动替换 JSON。以下命令每次选择一个实例，从仓库根目录执行。

<a id="routes"></a>
## 服务与协议

| 服务 | 任务和限制 |
|---|---|
| Base 普通 HTTP | T2VA/I2VA 使用 FL2VA 实例；Ref2VA 使用独立实例。768P，4–15 秒整数 |
| NPU Refiner | 固定 T2VA/I2VA/Ref2VA 中的一种工作流，2K，5/10/15 秒 |
| CUDA Refiner | 独立 packed NVFP4 FL2VA/I2VA 路线，不能套用默认 Base 表 |
| CUDA MaaS | 真实 Base 768P；IR/2K 模拟仅用于显式开启的接口联调 |

Base 和 Refiner 都有自己的资产、帧数和后端契约。精度、并行、固定模型、LoRA、decoder 和条件重叠属于启动设置；请求不能任意覆盖。

<a id="cuda"></a>
## CUDA Base

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_cuda_base.example.json \
  --host 127.0.0.1 --port 5001
```

`t2va` 与 `fl2va` 服务模式的 FL2VA 模型均可接收文字/首帧任务。Ref2VA 改为 `--service-mode ref2va`，使用 `h3_cuda5090_ref2va.example.json`。默认配置与回退见 [CUDA](H3-CUDA.md)。

<a id="npu"></a>
## NPU Base

先加载 CANN，安装 [VC 与融合库](H3-NPU-DEFAULT.md#operators)，并设置部署环境：

```bash
export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false OMP_NUM_THREADS=8 HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True INF_NAN_MODE_FORCE_DISABLE=1
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json \
  --host 127.0.0.1 --port 5001
```

Ref2VA 改为对应模式及 `h3_npu_ref2va.example.json`。两份示例均使用 `atten=vc`、`h3_opts=all`、20 次实际 Base DiT 和 native-stream。attention/融合在启动 JSON 中修改，不把离线的 `--atten` / `--h3-opts` 直接传给服务外层。

<a id="refiner"></a>
## 2K 服务

NPU 可从 `example/diffusers/configs/h3_refiner.example.json` 准备，修改资产路径和固定任务，再用原服务入口启动。I2VA 的 `task_type` 为 `fl2va`；Ref2VA 同时换用对应 Transformer 与表。Base 20 步、Refiner 3 步分别配置。CUDA 2K 的独立资产要求见 [视频](H3-VIDEO-QUICKSTART.md#refiner)。

<a id="requests"></a>
## 普通 HTTP 请求

先用 `GET /params` 检查服务，再保存对应请求为 `request.json`。下面为首尾帧示例：

```json
{"task_type":"fl2va","prompt":"The subject walks towards the camera.","duration":5,"resolution":"768p","num_inference_steps":20,"first_frame":"inputs/first-frame.png","last_frame":"inputs/last-frame.png"}
```

T2VA 使用 `task_type=t2va` 并省略帧字段。Ref2VA 使用独立实例、`task_type=ref2va` 和有序 `ref_path` 数组。路径须由服务端读取，远端客户端的本机路径不能直接作为服务输入。

```bash
curl -fsS http://127.0.0.1:5001/params
curl -fsS -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:5001/generate
TASK_ID=replace-with-task-id
curl -fsS "http://127.0.0.1:5001/status/$TASK_ID"
curl -fsS -o result.mp4 "http://127.0.0.1:5001/download/$TASK_ID"
```

查询到 `completed` 后下载。Base 步数填实际 DiT 次数并匹配表；省略时由所用配置/资产确定。Ref2VA 媒体数量、音轨和参考图缩放规则见 [视频](H3-VIDEO-QUICKSTART.md#references) 与 [配置](H3-CONFIGURATION.md#reference-images)。

H3 prompt 上限为 7000 个 UTF-16 码元，在入队前检查原文，不截断；这不是 token 数量。常见中英文计 1，补充平面 emoji 通常计 2。`/params` 标明计数口径。不同协议的错误信封和状态码以各自 API 文档为准。

<a id="maas"></a>
## CUDA MaaS

MaaS 平台处理身份、租户、配额、计费和路由；xGen 的 `--maas` 注册协议、队列、探针和结果发布。V2 协议与计算设备是两层，当前这个服务入口只开放 CUDA；NPU 继续使用普通 HTTP。

```bash
python -m pip install -r server/requirements-maas.txt
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode t2va \
  --config example/diffusers/configs/h3_maas.example.json \
  --host 127.0.0.1 --port 5001
```

Ref2VA 使用独立模式和 `h3_cuda5090_ref2va.example.json`。启动要求 `--init` 且分布式进程数与 SP 匹配；更改配置后重启。先完成真实请求预热，再检查 `/ready` 后接流量。

协议字段、路由、取消、错误、队列上限和超时由 [MaaS API](../server/video/MAAS-API.md)、[首尾帧 API](../server/video/MAAS-FL2VA-API.md) 与 [OpenAPI](openapi/h3-maas-v2.json) 统一维护，不在部署页复制一套。

`/v2/video_generation` 与 V2 查询使用自己的响应状态；旧 `/v1/tasks` 保留其兼容契约。`--maas-mock-backends` 的 IR/2K 是模拟结果，不能当作真实插帧或 Refiner 服务。可见水印属于输出发布阶段，不决定模型精度。

<a id="service-settings"></a>
## MaaS 运行配置

| 配置 | 作用范围 |
|---|---|
| `--maas-state-dir` / `XGEN_MAAS_STATE_DIR` | 默认 `outputs/maas-state`；每副本独立，重启复用 |
| `XGEN_MAAS_API_KEY` | 可选 V2 内部 Bearer 校验，不改变旧接口 |
| `XGEN_MAAS_FILE_BASE_URL` | 平台文件服务前缀，`mm_file://ID` 经平台前缀解析后使用受控下载 |
| `XGEN_MAAS_URL_TTL_S` | 签名下载 URL 有效期，默认 3600 秒，范围 1–604800 秒；不是推理超时 |
| `--maas-queue-capacity`、`--maas-queue-timeout-s` | 默认 10 条等待、600 秒；不限制已派发任务的执行时长 |
| `XGEN_MAAS_MOCK_TIMEOUT_S` | 模拟 FFmpeg 分支的 180 秒进程时限，不限制真实模型生成 |

回调 challenge/通知超时为 3 秒；媒体下载、探针和 DNS 约束见 [API](../server/video/MAAS-API.md)。客户身份、租户/文件归属、配额、跨副本路由和计费由上层平台维护，查询需路由到持有任务的实例。

模拟后端集中在 [maas_v2_backends.py](../server/video/maas_v2_backends.py)，接收已校验请求与有序媒体。接入真实外部后端需要单独实现提交/查询/取消和持久化映射；当前模拟分支不发送真实第三方生成请求，也不提供真实 IR/2K 质量或用量。

<a id="delivery"></a>
## 输出和生命周期

Native-stream 已编码结果复用原文件发布器，不把占位输出再次编码。请求使用独立暂存文件，校验后原子发布，失败清理暂存文件。MaaS 输出目录遵循 Python 临时目录设置，`TMPDIR` 可由部署环境指定。

15 秒在线交付 360 帧，视频和音频按输出契约裁剪；5/10 秒及离线入口可能保留模型原生 padding，不能一律断言为时长×24。在线完成、文件成功下载、完整解码和人工质量评审分别检查。

常驻模型保留，每请求临时条件和输出使用后释放。`H3_REQUEST_MEMORY_RELEASE` 记录回收时间及前后显存，不代替请求内部峰值检查。停服时确认所属 worker 和编码任务退出，初始化、服务端生成、发布与客户端等待分别记录。

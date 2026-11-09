# H3 MaaS API（768P）

用 `--init --maas` 启动 CUDA H3 Base 服务。`--maas-fl2va-input` 是弃用的 CLI 别名，行为与 `--maas` 相同。不开这些开关时，普通在线入口的规则保持不变。

安装 `pip install -r server/requirements-maas.txt`，环境还需提供 `ffmpeg`、`ffprobe`。HEIC/HEIF 使用 pillow-heif 解码，缺少插件时启动直接报错，不宣称支持无法解码的格式。

## 部署

从仓库根目录执行，先填写示例配置中的模型路径。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode fl2va \
  --config example/diffusers/configs/h3_maas.example.json
```

R2VA 使用独立模型实例，复用现有八卡 5090 配置：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode ref2va \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json
```

步数、权重、量化、attention 和并行策略由启动配置决定。MaaS 不允许请求覆盖这些配置；步数显式传入时必须匹配启动配置。T2VA/I2VA 与 R2VA 使用不同实例，不在请求间临时切换模型。Refiner、2K、1080P、旧 reduced-step warmup 不在本次开放范围。

## 提交

`POST /v1/tasks` 使用 `{"id":"job-1","input":{...}}`，可选 `created_at` 为客户端 Unix 毫秒时间戳。`POST /generate` 直接接收 input 对象。请求须为 UTF-8 JSON。

T2VA：

```json
{"id":"text-1","input":{"task_type":"t2va","prompt":"两个人交谈后起身。","duration":5,"resolution":"768p","aspect_ratio":"16:9"}}
```

I2VA（仅尾帧）；首帧使用 `first_frame`，同时提供两个字段即首尾帧：

```json
{"id":"frame-1","input":{"task_type":"fl2va","prompt":"镜头缓慢接近人物。","last_frame":"https://assets.example.com/end.png","duration":10}}
```

R2VA（提交至 ref2va 实例）：

```json
{"id":"ref-1","input":{"task_type":"ref2va","prompt":"参考人物外观和视频动作生成对话。","ref_path":["https://assets.example.com/person.webp","https://assets.example.com/motion.mp4","https://assets.example.com/voice.wav"],"duration":15,"aspect_ratio":"adaptive"}}
```

图片、视频、音频均可使用公网 HTTP(S) URL 或 Base64 data URI，不接受服务器本地路径、`file://`、私有 IP 或平台专有文件 ID。URL 不需带扩展名；下载后按真实内容识别，保留素材顺序。请求结构沿用 xGen 字段，不兼容官方 content[] 协议。

| 输入 | 规则 |
|---|---|
| prompt | 必填非空，最多 7000 UTF-16 码元，按 JavaScript string.length 计数；禁止 NUL 和非法 Unicode，不截断 |
| model | MiniMax-H3，可省略 |
| resolution | 固定 768p，可省略，也接受 768P 写法；不开放 width/height |
| duration | 5..15 的 JSON 整数，默认 5；4 秒尚未开放 |
| seed | 0..2147483647 的 JSON 整数，默认 42 |
| enable_audio | JSON 布尔值，默认 true |
| first_frame/last_frame | I2VA 至少一个，每个最多一张；不可与 ref_path 混用 |
| ref_path | R2VA 必填非空数组，最多 9 图、3 视频、3 音频，总计最多 12；音频需搭配图或视频 |
| 图片 | JPEG/PNG/WebP/HEIC/HEIF；单文件最多 30 MiB；每边 256..5760；宽/高为 0.4..2.5；仅静态图 |
| 图片像素乘积 | 每张首尾帧最多 8388608；每张参考图最多 16777216，保留原来的两套上限，不按多图累计 |
| 视频 | MP4/MOV，H.264/H.265；视频中的音轨 AAC/MP3；单文件最多 50 MiB；每边 256..5760，比例 0.4..2.5，帧率 23.976..60 |
| 音频 | WAV/MP3，单文件最多 15 MiB |
| 视频/音频时长 | 每段 2..15 秒；各类分别累计不超过 15 秒，以实际媒体探测结果为准 |
| HTTP 请求体 | 最多 64 MiB，含 JSON/Base64。URL 指向文件另外按逐文件上限检查，不把下载总字节数误作请求体大小 |

文件大小按 1024×1024 字节计算，与官方 CLI 相同。格式和时长按 Pillow/ffprobe 实际识别，文件名及 MIME 声明不能替代内容校验。

## 比例

T2VA 支持 21:9、16:9、4:3、1:1、3:4、9:16，默认 16:9，不接受 adaptive。I2VA 总是使用首帧（无首帧时取尾帧）经 EXIF 方向修正后的比例；首尾帧比例不同时仍以首帧为准，管线按目标尺寸处理尾帧。R2VA 可显式指定上述比例，默认 adaptive：按输入顺序取第一份图片或视频的比例，音频不参与。

R2VA 的自动选比例是 xGen 明确规定的本地规则，官网未公开其内部选择算法。长宽按现有 1344×768 的像素预算计算并向下对齐到 32，16:9/9:16 保留 1344×768/768×1344。不是任意分辨率，也不把短边一律设为 768。结果包含实际 width/height；实际宽高比受网格对齐影响。

## 下载与临时素材

素材在受理入队前下载和校验，最多同时两个 admission 请求。忙时返回 503，不占 GPU；下载耗时发生在任务受理之前，不计入原来的排队时间。只连接经过公网 IP 检查的实际地址，HTTPS 验证原域名证书；每次重定向重新检查目标，最多 3 次。不信任 Content-Length，流式检查真实字节数；下载失败不自动重试。

`XGEN_MAAS_DOWNLOAD_TIMEOUT_S` 设置单文件下载期限，默认 60 秒，允许 (0,3600]；DNS 查询还受系统解析器超时控制。`XGEN_MAAS_MEDIA_ROOT` 指定临时素材目录，须提前创建，八卡进程应可见同一路径。素材由 HTTP 接入阶段准备一次；rank 不重复下载。请求拒绝、去重、取消、排队过期、GPU 失败或完成后清理对应素材；进程被 SIGKILL 或主机掉电时可能留目录，需要运维清理。

输入媒体的 ffprobe 探测超时为 30 秒；编码后的输出校验超时为 15 秒。素材在受理之前准备，这段耗时不包含在原来的 `timings_s.queue` 中。

## 沿用的可靠性接口

查询 `/v1/tasks/<id>`、取消 `DELETE /v1/tasks/<id>`、探针 `/health` 和 `/ready`、规格 `/params`、指标 `/metrics`、结果 `/download/<id>` 保持原路由。`/config` 只读。先发真实请求预热，通过 `/ready` 后再接平台流量。

保留同 ID 去重、默认 10 条等待队列、600 秒排队期限和单调时钟计时。配置参数仍为 `--maas-queue-capacity`、`--maas-queue-timeout-s` 和已有阶段就绪时限。校验错误返回 400，文件/请求体超限返回 413，非 JSON 返回 415；去重输入冲突返回 409。客户端输入校验失败不会改变 worker 就绪状态，不恢复 latent，不自动重试生成。

## 验证边界

CPU 回归通过 **46 项测试、242 个子用例**，覆盖新旧 CLI 参数、T2VA/I2VA/R2VA 路由、真实 JPEG/PNG/WebP/HEIC/HEIF 解码、音视频探测、受控 HTTP 下载、素材限额、输入去重和临时文件清理，以及原有 MaaS 可靠性测试。本地文档链接检查通过。测试入口为 [test_maas.py](test_maas.py)、[test_maas_fl2va.py](test_maas_fl2va.py) 和 [test_maas_artifacts.py](test_maas_artifacts.py)，在安装服务依赖的 CPU 环境运行：

```bash
CUDA_VISIBLE_DEVICES='' XGEN_DEVICE=cuda python -m pytest -q \
  server/video/test_maas.py server/video/test_maas_fl2va.py server/video/test_maas_artifacts.py
```

本次接口与 CPU 媒体探测测试不代表所有比例、首尾帧、9 图或最大混合输入都已在 GPU 完成生成。真实八卡 1 步容量穿刺、完整步数成片和人工画面/音频验收需分别记录；历史 16:9 单首帧或 R2VA 性能不能代替新组合验收。

## 可靠性字段详细说明

以下沿用原有队列、状态、计时和超时语义；输入规格以上文统一 MaaS 规则为准。

## 任务状态、排队与耗时

等待中的任务快照。

```json
{
  "id": "fl2va-0001",
  "status": "pending",
  "output": "",
  "error_code": null,
  "failure_stage": null,
  "timings_s": {},
  "queue_elapsed_s": 12.345,
  "queue_ahead": 2
}
```

| 状态 | 含义 |
|---|---|
| `pending` | 已受理，等待派发，可取消 |
| `running` | 派发线程已领取，可能正在派发、加载、推理、编码或交付 |
| `succeeded` | 成片已通过校验并发布 |
| `failed` | 任务失败或过期，查看 `error_code` 和 `failure_stage` |
| `cancelled` | 派发前已取消，重复取消返回当前终态 |

默认每实例最多 **10 条等待任务**，执行中的任务另计。队列满返回 `503 queue_full`。待派发任务自受理起等待满 **600 秒**后过期。服务在受理、查询、领取及定期清理时检查过期。取消释放等待槽位，下一次受理前会清理已过期的项。部署时可通过 `--maas-queue-capacity` 和 `--maas-queue-timeout-s` 配置。

`queue_elapsed_s` 是当前已等待的秒数。`queue_ahead` 是排在前面的有效等待任务数，执行中的任务另计。两个字段仅在 `pending` 时返回。`timings_s` 中的字段随阶段结束填充。

| 字段 | 计时范围 |
|---|---|
| `queue` | 服务受理到派发线程领取。取消或过期时，记录到服务检测该状态的时刻 |
| `inference` | 两次设备同步之间的 `infer_single()`，包含模型推理和管线内解码。首次模型加载发生在此之前 |
| `encode` | 本地编码、媒体校验和原子发布 |
| `delivery` | 本地发布到交付确认。启用 OBS 时包含复制和可读性确认 |
| `total` | 服务受理到终态，包含派发、首次模型加载和阶段间开销。客户端下载另计 |

耗时按服务端单调时钟计秒，保留三位小数。失败时，尚未完成的阶段字段可能缺失。任务记录保存在实例内存，终态约保留 2 小时，每 5 分钟清理。实例重启后无法查询旧记录。

## 成片输出

成功响应中的 `output` 是 **JSON 字符串**，调用方需要再解析一次。

```json
{
  "id": "fl2va-0001",
  "status": "succeeded",
  "output": "{\"video_path\":\"<server-local-path>/video_example.mp4\",\"infer_time\":50.23}",
  "error_code": null,
  "failure_stage": null,
  "timings_s": {
    "queue": 0.001,
    "inference": 50.232,
    "encode": 0.836,
    "delivery": 0.0,
    "total": 51.655
  }
}
```

- `video_path` 是服务端路径。客户端通过 `GET /download/<id>` 取 MP4。未成功的任务下载返回 `400`，任务或文件不存在时返回 `404`。
- 配置 `EXTERNAL_BASE_URL` 或 `--external-base-url` 时，解析后的结果包含 `download_url`。启用 OBS 时，该字段为签名链接，有效期由 `OBS_URL_EXPIRE_SECONDS` 配置。
- 服务校验临时成片的分辨率、帧率、帧数、视频包数和所需音轨，通过后才原子发布。启用 OBS 时，还须确认对象和签名链接可读，任务才进入 `succeeded`。

## 错误响应

提交失败时检查 HTTP 状态和 `error`。查询已存在的任务时，HTTP `200` 表示查询成功，任务成败由 `status` 表示。

| HTTP | 错误 | 含义 |
|---|---|---|
| `400` | `invalid_input` | JSON、字段类型、规格或图片内容无效 |
| `413` | `invalid_input` | 请求体或图片大小、像素数、边长超限 |
| `415` | `invalid_input` | 请求内容类型不是 JSON |
| `404` | 任务不存在 | 找不到记录。取消接口返回 `task_not_found` |
| `409` | `task_conflict` | 同 ID 对应不同输入 |
| `409` | `task_not_cancellable` | 已派发或已进入其他终态 |
| `429` | `too_many_tasks` | 当前实例的任务记录已达 1,000 条，包含终态记录 |
| `503` | `queue_full` | 无空余等待槽位 |
| `503` | `worker_unavailable` | 实例降级。未初始化或关闭中也可返回 `503`，响应文案不同 |

输入校验失败示例。

```json
{"error":"invalid_input","field":"duration","message":"Expected 5"}
```

| 任务 `error_code` | `failure_stage` | 含义 |
|---|---|---|
| `queue_expired` | `queue` | 待派发时达到等待上限 |
| `cancelled_by_user` | `queue` | 待派发时被调用方取消 |
| `encode_failed` | `encode` | 本地编码、校验或发布失败 |
| `storage_failed` | `storage` | OBS 复制或可读性验证失败 |
| `worker_lost` | 当前阶段，等待任务为 `queue` | 服务捕获到派发或推理链路故障 |
| `service_shutdown` | `shutdown` | 退出清理标记未结束的任务。进程退出后记录不可查询 |

排队过期后的任务查询示例，HTTP 状态为 `200`。

```json
{
  "id": "fl2va-0002",
  "status": "failed",
  "output": "",
  "error_code": "queue_expired",
  "failure_stage": "queue",
  "timings_s": {"queue": 600.041, "total": 600.041}
}
```

## 探针、超时与实例指标

`/health` 以 HTTP `200` 表示服务进程可达，同时附带就绪快照以及 `model`、`profile`、`rank`、`world_size` 等字段。`/ready` 返回就绪快照，并按 `ready` 返回 `200` 或 `503`。正常推理期间可以保持就绪。

```json
{
  "state": "ready",
  "ready": true,
  "reason": null,
  "phase": "generating",
  "model_loaded": true,
  "workers": {"expected": 8, "group_state": "confirmed"}
}
```

| `state` | 可能的 `reason` | 含义 |
|---|---|---|
| `starting` | `workers_unverified`、`model_not_loaded`、`model_loading`、`warmup_pending` | 等待 worker 确认、模型加载或首单预热 |
| `ready` | `null` | 已完成预热，worker 组已确认 |
| `degraded` | `worker_failed`、`progress_timeout` | 捕获 worker 故障或当前阶段超时 |
| `draining` | `shutdown_requested` | 收到关闭信号 |

`workers.group_state` 为 `confirmed`、`unverified` 或 `failed`。`loading`、`generating`、`saving` 的就绪阈值分别为 **600、1,200、180 秒**，到点会影响就绪和新任务受理，当前任务可以继续执行。`idle`、`dispatching`、`confirming_workers` 仅记录阶段。排队的 600 秒限制只影响尚未派发的任务。

编码后校验的 `ffprobe` 子进程超时为 **15 秒**。启用 OBS 时，元数据轮询预算为 **10 秒**，签名链接探测的读取超时为 **5 秒**。完整的成功交付耗时由 `timings_s.delivery` 记录。

排队时限由启动参数 `--maas-queue-timeout-s` 配置。三个阶段就绪阈值、FFprobe 校验时限和 OBS 两个操作时限当前为代码固定值。OBS 签名链接有效期由 `OBS_URL_EXPIRE_SECONDS` 单独配置。

`GET /metrics` 返回 `tasks.total`、`tasks.completed`、`tasks.running`、`tasks.pending`、`tasks.failed`，以及 `performance.avg_infer_time`、`min_infer_time`、`max_infer_time`，耗时单位为秒。统计范围为实例内仍保留的任务。

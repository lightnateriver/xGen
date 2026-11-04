# xGen MaaS FL2VA 接口

本文适用于开启 `--maas-fl2va-input` 的在线服务。当前规格为 CUDA MiniMax-H3 Base FL2VA，输出 1344×768、16:9、5 秒、24 fps、124 帧。51 个调度点对应 50 次 DiT。该模式接收首帧和文本，暂不支持 10 秒、15 秒、2K、Refiner 或 REF2VA。

## 启动和接入

可直接复制的配置、八卡启动和预热命令见 [CUDA MaaS 启动](../../docs/H3-CUDA.md#maas-在线服务)。

沿用已有 H3 CUDA FL2VA 配置，加上 `--init --maas-fl2va-input`。启动配置使用 `model=MiniMax-H3`、`task_type=fl2va`、`num_inference_steps=51` 和 `h3_attention=legacy`，SP 与实际 rank 数一致。环境需提供 `ffmpeg` 和 `ffprobe`，可通过 `FFMPEG_PATH`、`FFPROBE_PATH` 指定。使用真实请求预热，省略启动参数 `--warmup`。权重和环境配置见 [H3 CUDA 部署教程](../../docs/H3-CUDA.md)。

1. 启动后直接向实例提交一条符合规格的预热请求。首条成片发布成功且各 rank 确认后，`/ready` 返回 `200`。
2. MaaS 使用自有任务 ID 调用 `POST /v1/tasks`，轮询 `GET /v1/tasks/<id>`。
3. 状态为 `succeeded` 时取成片，为 `failed` 或 `cancelled` 时读取错误信息并结束轮询。

| 方法和路径 | 用途 | 成功响应 |
|---|---|---|
| `POST /v1/tasks` | 提交任务，由调用方提供 ID | `200`，任务快照 |
| `GET /v1/tasks/<id>` | 查询任务状态、排队及耗时 | `200`，任务快照 |
| `GET /v1/tasks` | 列出当前实例保留的任务 | `200`，`{"tasks":[...]}` |
| `DELETE /v1/tasks/<id>` | 取消待派发任务 | `200`，`cancelled` 任务快照 |
| `GET /download/<id>` | 下载已成功的成片 | `200`，`video/mp4` 附件 |
| `GET /health` | HTTP 进程存活探针，附带就绪详情 | `200` |
| `GET /ready` | 负载均衡就绪探针 | 就绪 `200`，其余 `503` |
| `GET /params` | 查询固定规格、输入字段和请求体上限 | `200`，含 `profile`、`parameters`、`max_body_bytes` |
| `GET /metrics` | 查询当前实例的任务数和推理耗时 | `200`，JSON 汇总 |
| `GET /config` | 查看启动配置 | `200`，配置对象。写入返回 `405` |

兼容接口仍可用。`POST /generate` 直接接收下文的 `input` 对象，由服务生成 ID，返回 `{"status":"accepted","task_id":"..."}`。`GET /status/<id>` 返回任务详情以及 ISO 格式的 `created_at`、`started_at`、`completed_at`，成功状态写作 `completed`。`GET /result/<id>` 返回成功任务的详情和结果对象。`POST /cancel/<id>` 与 MaaS 取消接口语义相同。`POST /init` 在该模式下返回 `405`。

## 输入

使用 `Content-Type: application/json` 和 UTF-8。将示例中的图片占位符替换为实际 JPEG 的 Base64 数据，保存为 `request.json`。

```json
{
  "id": "fl2va-0001",
  "input": {
    "prompt": "固定镜头，两位人物轮流交谈，最后左侧人物起身。",
    "first_frame": "data:image/jpeg;base64,<BASE64_JPEG>",
    "seed": 42,
    "enable_audio": true
  }
}
```

```bash
curl -sS -H 'Content-Type: application/json' --data-binary @request.json "$BASE_URL/v1/tasks"
curl -sS "$BASE_URL/v1/tasks/fl2va-0001"
curl -fS -o result.mp4 "$BASE_URL/download/fl2va-0001"
```

`BASE_URL` 指向 xGen 实例或 MaaS 网关。任务成功后执行下载命令。

| 字段 | 类型 | 约束 |
|---|---|---|
| `id` | 字符串，必填 | 1 至 128 个 ASCII 字母、数字、下划线或连字符，首字符须为字母或数字 |
| `created_at` | 整数，可选 | 客户端 Unix 毫秒时间戳，须大于 0 且早于 2100 年。排队和总耗时从服务实际受理时计算 |
| `input.prompt` | 字符串，必填 | 最多 2,000 个 Unicode 码点，允许中文和换行，须含非空白内容，禁止 NUL |
| `input.first_frame` | 字符串，必填 | 单帧 JPEG 或 PNG。前缀为 `data:image/jpeg;base64,` 或 `data:image/png;base64,`。解码后的文件最多 4 MiB，图像最多 8,388,608 像素，每边 32 至 4096 像素，长短边比例最大 4:1 |
| `input.seed` | 整数，可选 | 0 至 2,147,483,647，默认 42 |
| `input.enable_audio` | 布尔值，可选 | 默认 `true`。开启时，成片必须有包含数据的音轨 |

固定字段可省略。显式传入时须匹配下列值和类型。

```json
{
  "model": "MiniMax-H3",
  "task_type": "fl2va",
  "resolution": "768p",
  "aspect_ratio": "16:9",
  "duration": 5,
  "num_inference_steps": 51
}
```

完整请求体上限为 **5,625,176 字节**，包含 JSON 和 Base64。远程图片 URL、本地文件路径、未知字段、重复 JSON 字段和非有限数值会被拒绝。

相同 ID 和相同的规范化输入返回已有任务。规范化会补全默认值。相同 ID 携带不同输入返回 `409 task_conflict`。去重仅在单个实例且记录仍保留时生效，轮询和同 ID 重提交应路由到持有任务的实例。

实例由平台网关代理，鉴权、租户配额和请求路由由平台负责。`/config`、实例任务列表和运行指标适合内部运维访问。客户端用任务 ID 或 `download_url` 获取结果。状态查询与下载是独立请求，网关的单次 HTTP 读取超时可与推理耗时独立设置。

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

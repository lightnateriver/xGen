# H3 MaaS API（768P）

`--init --maas` 开启 CUDA H3 Base 的输入校验、排队、探针和成片发布。旧 CLI 开关 `--maas-fl2va-input` 是弃用别名；不开 MaaS 时普通入口保持原有行为。

## 与 MiniMax 官网的对齐范围

**部分对齐，不兼容官方请求／响应协议。** 以下比较标准 MiniMax-H3（核对日期：2026-11-10），不包含 Highspeed、Lite、Max、Context-IR 或 Regenerate。

| 范围 | 结论 |
|---|---|
| 输入限额与模式约束 | 下文的 7000 UTF-16 文本计数、素材格式／大小／数量／尺寸／比例、参考音视频时长、64 MiB 请求体及输入模式互斥已按官方文档／CLI 实现 |
| 本地收窄或额外限制 | 只开放 768P、5～15 秒整数（官网标准 H3 为 768P/2K、4～15 秒）；保留图片像素乘积上限，只接受静态图片；额外拒绝 NUL、非法 Unicode，并限制下载目标及超时 |
| 不声明一致的部分 | xGen 路由、字段、错误码、任务响应、模型部署及推理配置；R2VA 自动选比例算法、实际输出尺寸、生成结果、速度和服务等级；不接受 `mm_file://` 或平台文件 ID |

依据：[官方输入要求](https://platform.minimax.io/docs/guides/video-generation)、[创建接口](https://platform.minimax.cn/docs/api-reference/video-generation-v2-create)、[CLI 文本校验](https://github.com/MiniMax-AI/cli/blob/main/src/video/v2.ts)、[CLI 字节限额](https://github.com/MiniMax-AI/cli/blob/main/src/utils/media.ts)。对齐指输入规则实现，不代表兼容认证或 GPU 精度验收。鉴权、租户配额、限流及内容审核由接入平台另行核验。

## 启动

从仓库根目录执行，先填写配置中的模型路径，并准备 `ffmpeg`、`ffprobe`：

```bash
pip install -r server/requirements-maas.txt

XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode fl2va \
  --config example/diffusers/configs/h3_maas.example.json
```

上例服务 T2VA/I2VA。R2VA 使用 Ref2VA 权重和独立实例：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode ref2va \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json
```

SP 须等于实际 rank 数。步数、权重、量化、attention 和并行策略在启动时固定；请求不得覆盖，显式传入步数须匹配配置。不开放 Refiner、2K、1080P 或旧 reduced-step warmup。HEIC/HEIF 需要依赖中的 pillow-heif，缺失时启动报错。先提交真实预热请求，等 `/ready` 返回 200 再接平台流量。

## 请求与输入限制

使用 UTF-8、`Content-Type: application/json`。`POST /v1/tasks` 接收 `{"id":"job-1","input":{...}}`，`POST /generate` 直接接收 input 并生成 ID。

`id` 为 1～128 个 ASCII 字母、数字、下划线或连字符，首位须为字母或数字。可选 `created_at` 是大于 0、早于 2100 年的 Unix 毫秒时间戳；服务计时仍从实际受理开始。未知字段、重复 JSON 字段及非有限数值会被拒绝。

| 字段 | 规则 |
|---|---|
| `prompt` | 必填非空，最多 **7000 UTF-16 码元**，与 JavaScript `string.length` 一致；汉字通常计 1，`🙂` 计 2；禁止 NUL、非法 Unicode，超长拒绝、不截断，不按 tokenizer token 数计数 |
| `model` / `resolution` | 默认 MiniMax-H3 / 768p；也接受 768P 写法，不接受任意 width/height |
| `task_type` | t2va / fl2va / ref2va；须符合实例工作流，不能用普通实例临时切换 Ref2VA 权重 |
| `duration` | 5～15 的 JSON 整数，默认 5；4 秒暂未开放 |
| `num_inference_steps` | 可省略；显式传入时须匹配启动配置 |
| `seed` / `enable_audio` | JSON 整数 0～2147483647，默认 42 / JSON 布尔值，默认 true |
| `first_frame` / `last_frame` | I2VA 至少一个，各最多 1 张，不可与参考素材混用；T2VA 不接受媒体 |
| `ref_path` | R2VA 必填数组：最多 **9 图、3 视频、3 音频，总计 12 个文件**；音频需搭配图片或视频 |
| `aspect_ratio` | 21:9、16:9、4:3、1:1、3:4、9:16 或 adaptive；具体规则见下文 |

素材可以使用公网 HTTP(S) URL 或 Base64 data URI，按真实内容识别，保留输入顺序；URL 不必带扩展名。不接受私有 IP、服务器本地路径、`file://` 或平台文件 ID。字段沿用 xGen，不提供官方 content[]/role/ratio 协议层。

| 素材 | 格式及硬限制 |
|---|---|
| 图片 | 静态 JPEG/PNG/WebP/HEIC/HEIF；每张 **≤30 MiB**；每边 **256～5760**；宽/高 **0.4～2.5** |
| 图片像素乘积 | 每张首尾帧 ≤8,388,608；每张参考图 ≤16,777,216，分别检查，不按多图累计 |
| 视频 | MP4/MOV，H.264/H.265，附带音轨 AAC/MP3；每段 **≤50 MiB**；每边 256～5760，宽/高 0.4～2.5，帧率 23.976～60 |
| 音频 | WAV/MP3；每段 **≤15 MiB** |
| 音视频时长 | 每段 **2～15 秒**；**视频合计 ≤15 秒，音频独立合计 ≤15 秒**，以实际探测值为准 |
| 请求体 | **≤64 MiB**，包含 JSON/Base64；URL 指向的素材另按单文件上限检查 |

1 MiB = 1024×1024 字节，与官方 CLI 的计算相同。所有素材在入队前校验，格式声明或文件名不能替代实际解码／探测。

### 比例

- T2VA：使用具体比例，默认 16:9，拒绝 adaptive。
- I2VA：按首帧（无首帧时按尾帧）经 EXIF 修正后的比例自适应；首尾帧比例不同仍以首帧为准，管线按目标尺寸处理尾帧。传入其他有效比例也按自适应处理。
- R2VA：可指定具体比例，默认 adaptive，按素材顺序取第一份图片或视频的显示比例，音频不参与。

R2VA 自动选比例是本地规则，不声明与官网算法一致。输出按 1344×768 像素预算计算并向下对齐到 32；16:9/9:16 保留 1344×768/768×1344，实际比例受对齐影响。这不等于任意分辨率，也不保证短边为 768。

### 三种请求示例

```json
{"id":"text-1","input":{"task_type":"t2va","prompt":"两个人交谈后起身。","duration":5,"aspect_ratio":"16:9"}}
```

```json
{"id":"frame-1","input":{"task_type":"fl2va","prompt":"镜头缓慢接近人物。","last_frame":"https://assets.example.com/end.png","duration":10}}
```

```json
{"id":"ref-1","input":{"task_type":"ref2va","prompt":"参考人物外观、动作和声音。","ref_path":["https://assets.example.com/person.webp","https://assets.example.com/motion.mp4","https://assets.example.com/voice.wav"],"duration":15}}
```

I2VA 使用 `first_frame` 即首帧，两个字段同时提供即首尾帧。R2VA 示例须提交至 ref2va 实例。

## 查询与成片

| 方法／路径 | 用途与响应 |
|---|---|
| `POST /v1/tasks` | 提交；成功返回 200 和任务快照。同 ID、同规范化输入去重，不同输入返回 409 |
| `GET /v1/tasks/<id>` / `GET /v1/tasks` | 查询一个任务／列出实例保留的任务 |
| `DELETE /v1/tasks/<id>` | 取消待派发任务；已派发或其他不可取消终态返回 409 |
| `GET /download/<id>` | 下载成功的 MP4；未成功返回 400，任务或文件不存在返回 404 |
| `GET /health` / `GET /ready` | 存活始终返回 200；就绪返回 200，其余 503，均附就绪详情 |
| `GET /params` / `GET /metrics` | 输入规格／当前实例任务数和推理耗时 |
| `GET /config` | 查看启动配置；写入及 `POST /init` 返回 405 |

兼容路由：`POST /generate` 返回 `accepted` 和 `task_id`；`GET /status/<id>`、`GET /result/<id>` 查询详情或结果；`POST /cancel/<id>` 取消。兼容路由将成功状态写作 `completed`，`/v1/tasks` 使用 `succeeded`。

任务状态：`pending` 等待派发，`running` 已领取（含加载／编码／交付），`succeeded` 已验收发布，`failed` 失败或过期，`cancelled` 派发前取消。查询返回 200 只代表查询成功，仍需检查任务状态。

成功任务的 `output` 是 **JSON 字符串，需再解析一次**：

```json
{"id":"ref-1","status":"succeeded","output":"{\"video_path\":\"<server-local-path>/video_example.mp4\",\"infer_time\":50.23,\"width\":1344,\"height\":768,\"resolution\":\"768p\"}","error_code":null,"failure_stage":null,"timings_s":{"queue":0.001,"inference":50.232,"encode":0.836,"delivery":0.0,"total":51.655}}
```

`video_path` 是服务端路径，客户端使用下载接口。设置 `EXTERNAL_BASE_URL` 或 `--external-base-url` 后，结果还包含 `download_url`；启用 OBS 时为签名链接，有效期由 `OBS_URL_EXPIRE_SECONDS` 设置。临时成片须通过本请求的尺寸、帧数、24 fps、包数及所需音轨校验，才原子发布；OBS 还需对象和签名链接可读。示例耗时仅说明字段，不是新 MaaS 性能承诺。

## 排队、计时与错误

默认最多 **10 条等待任务**，执行中的任务另计；容量由 `--maas-queue-capacity` 设置。待派发任务从实际受理起等待满 **600 秒**过期，由 `--maas-queue-timeout-s` 设置，不限制已派发任务的推理时长。受理、查询、领取和定期清理时检查过期；取消／过期释放槽位。

等待任务包含 `queue_elapsed_s`（已等待秒数）和 `queue_ahead`（前面有效等待任务数，不含执行中任务）；派发或结束后保留实际排队耗时。所有时间用服务端单调时钟计秒、保留三位小数，失败时未完成阶段可缺失。

| `timings_s` 字段 | 范围 |
|---|---|
| `queue` | 受理到领取；取消／过期时到被服务检测的时刻 |
| `inference` | 两次设备同步之间的 infer_single，含管线内解码，不含前置模型加载 |
| `encode` | 本地编码、媒体校验及原子发布 |
| `delivery` | 本地发布到交付确认，含启用 OBS 后的复制和可读性确认 |
| `total` | 受理到终态，含派发、加载及阶段间开销；素材受理前处理和客户端下载另计 |

任务记录在实例内存，终态约保留 2 小时，每 5 分钟清理，重启后丢失；去重及查询应路由到持有记录的同一实例。

| HTTP／错误 | 含义 |
|---|---|
| 400 `invalid_input` | JSON、字段、组合、规格或媒体内容无效，响应含 `field`、`message` |
| 413 `invalid_input` / 415 `invalid_input` | 请求体／素材字节或像素超限 / 非 JSON 内容类型 |
| 404 | 记录不存在；取消接口为 `task_not_found` |
| 409 `task_conflict` / `task_not_cancellable` | 同 ID 不同输入 / 任务不可取消 |
| 429 `too_many_tasks` | 实例记录达到 1000 条，包含终态 |
| 503 | `queue_full`：队列满；`worker_unavailable`：实例降级；未初始化、关闭中或素材处理入口忙也返回 503，文案不同 |

任务失败用 `error_code` / `failure_stage` 表示：`queue_expired`、`cancelled_by_user` 对应 queue；`encode_failed` 对应 encode；`storage_failed` 对应 storage；`worker_lost` 对应故障阶段（等待任务为 queue）；`service_shutdown` 对应 shutdown。输入校验失败不会改变 worker 就绪状态，不恢复 latent、不自动重试生成。

## 下载、探针与超时

受理前最多同时处理两份素材请求，忙时返回 503，不占 GPU。只连接校验过的公网 IP，HTTPS 验证原域名证书，每次重定向重新检查目标，最多 3 次；按实际读取字节限额，不信任 Content-Length。素材只准备一次，八卡不重复下载。

| 操作 | 默认值与配置 |
|---|---|
| 单文件下载 | 60 秒，`XGEN_MAAS_DOWNLOAD_TIMEOUT_S` 可设 (0,3600]；DNS 仍受系统解析器时限控制 |
| 输入 ffprobe / 成片 ffprobe | 30 秒 / 15 秒，代码固定 |
| loading / generating / saving 就绪阈值 | 600 / 1200 / 180 秒，代码固定；超限影响就绪和新受理，当前任务可以继续执行 |
| OBS 元数据轮询 / 签名链接读取 | 10 秒 / 5 秒，代码固定 |

`idle`、`dispatching`、`confirming_workers` 仅记录状态。正常生成期间可保持就绪；`/ready` 首先要求模型加载、真实预热发布成功且全部 rank 确认。就绪快照包含 `state`、`ready`、`reason`、`phase`、`model_loaded`、`workers.expected/group_state`，`/health` 还含 model/profile/rank/world_size。

- `state`：starting（worker／模型／预热未就绪）、ready、degraded（worker_failed／progress_timeout）、draining（shutdown_requested）。starting 的 reason 为 workers_unverified、model_not_loaded、model_loading 或 warmup_pending。
- `workers.group_state`：confirmed、unverified 或 failed。
- `/metrics`：实例保留记录的 tasks.total/completed/running/pending/failed，以及 performance.avg_infer_time/min_infer_time/max_infer_time（秒）。

`XGEN_MAAS_MEDIA_ROOT` 可指定预先创建、各 rank 可见的临时目录。拒绝、去重、取消、过期、失败和完成时清理素材；SIGKILL／掉电可能残留，需运维清理。OBS 使用 `OBS_LOCAL_PATH`；鉴权、租户路由及访问内部配置／指标的权限由平台负责。

## 验证范围

CPU 回归 **46 项、242 个子用例通过**，覆盖 CLI、路由、真实图片解码（含 HEIC/HEIF）、音视频探测、受控下载、输入校验、去重和清理及原可靠性能力。测试文件：[新 MaaS](test_maas.py)、[原 MaaS 回归](test_maas_fl2va.py)、[成片发布](test_maas_artifacts.py)。

```bash
CUDA_VISIBLE_DEVICES='' XGEN_DEVICE=cuda python -m pytest -q \
  server/video/test_maas.py server/video/test_maas_fl2va.py server/video/test_maas_artifacts.py
```

新比例、首尾帧、9 图及最大混合输入的 **GPU 容量、完整步数成片和人工画面／音频验收仍待完成**；历史 16:9 离线性能不能代替新 MaaS 组合的验收。

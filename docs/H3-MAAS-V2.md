# H3 MaaS V2：部署、兼容边界与后端替换

本页对应 v0.4 的 `--maas`。主接口说明见 [MAAS-API.md](../server/video/MAAS-API.md)，机器可读协议见 [OpenAPI](openapi/h3-maas-v2.json)。使用中国站标准 MiniMax-H3 V2 的字段和任务语义；协议一致不代表采用 MiniMax 的托管模型、计费或服务等级。

## 接口、平台与计算后端

| 层 | 职责 |
|---|---|
| 上层 MaaS 平台 | 对客户提供服务，处理身份、租户、配额、计费、任务路由及集群扩容 |
| V2 HTTP 协议 | 约定请求字段、创建/查询/取消等路径和响应格式，供 MaaS 调用 xGen |
| xGen 的 `--maas` | 注册 V2 接口，启用输入校验、队列、探针、任务记录和成片发布 |
| CUDA/NPU 后端 | 按模型、精度和并行配置实际执行计算；独立于 HTTP 字段定义 |

```text
客户端 → MaaS 平台 → V2 HTTP → xGen 在线服务（--maas）
                                       |
                                       +→ 当前：CUDA worker
                                       +→ NPU：此入口尚未开放/验收
```

当前实现保留 `maas.validate_startup` 的 CUDA-only 校验，另外要求 `--init` 和 SP 等于分布式进程数。这属于已发布服务入口的限制，不能解释为 V2 协议要求 GPU。NPU 仍可使用原离线及旧 HTTP 入口；本轮文档与发布不移除硬件校验，也不声明 NPU MaaS V2 已可用。后续应在共用 V2 协议下分别接入和验证后端。

| 启动方式 | 接口/功能 |
|---|---|
| 离线推理命令 | 生成文件，不提供 HTTP 接口 |
| 原 HTTP 服务，不带 `--maas` | 原 `/generate`、`/v1/tasks` 等路径；不注册新增 `/v2/*` |
| 当前 CUDA HTTP 服务，`--init --maas` | V2 生成/查询/列表/取消/删除、可选可见水印、原可靠性能力 |
| 再加 `--maas-mock-backends` | 在同一服务中启用 IR/2K 模拟；不是独立 CPU-only HTTP 服务 |

`--maas` 本身不决定模型精度、attention 或并行布局。水印当前接在 MaaS 在线请求中；CPU [水印基准脚本](../tools/bench_h3_watermark.py)可独立运行，不需要启动 HTTP 服务或加载 GPU 模型。

## 能力范围

| 能力 | 实现 |
|---|---|
| 768P，4～15 秒整数 | 复用真实 CUDA Base 推理、原有队列和原子发布；T2VA/I2VA 与 Ref2VA 分实例 |
| 首帧/尾帧/首尾帧、R2VA | content role 自动选择工作流，复用原媒体检查；保留输入参考顺序 |
| AIGC 水印 | 默认 false；true 在 FFmpeg 编码前叠加可见标识，不增加第二次转码 |
| IR | **模拟**：原样返回提示词，不扩写，不运行 IR 模型 |
| 2K | **模拟**：输出写有 MOCK / NOT MODEL OUTPUT 的 24 fps、静音测试卡，不调用 Refiner 或供应商 |
| 再生成 | 不提供 `/v2/video_regeneration` |
| 状态、查询、列表、取消/删除 | 官网 V2 格式；queued/running/succeeded/failed/cancelled，Unix 秒时间戳 |
| 任务历史 | 单副本七天元信息，SQLite；不存权重或输入媒体，不恢复 latent、不自动重跑 |
| 回调 | challenge 验证及 FIFO 状态通知；独立于 GPU worker |

IR 和 2K 默认不接受请求，返回 HTTP 422；只有启动时显式加入 `--maas-mock-backends` 才提供模拟结果。请求不能切换后端。模拟结果不返回真实 usage，也不参与真实性能、精度或计费统计。

水印的 CPU 开销、编码净增和计时边界见 [CPU 基准报告](https://github.com/lightnateriver/xGen/blob/6822583e0021e3ca64701d7c8aa49789a0d55f1c/docs/performance/h3-watermark-cpu.md)。这不是 GPU 推理性能报告。

## 启动

准备 CUDA 环境、模型及可信 VAE 路径后，在仓库根目录执行：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode fl2va \
  --config example/diffusers/configs/h3_maas.example.json \
  --maas-state-dir outputs/maas-fl2va \
  --host 127.0.0.1 --port 5001
```

R2VA 用独立 Ref2VA 实例，将工作流、配置及历史目录一起换成对应值：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas --service-mode ref2va \
  --config example/diffusers/configs/h3_cuda5090_ref2va.example.json \
  --maas-state-dir outputs/maas-ref2va \
  --host 127.0.0.1 --port 5002
```

需要 IR/2K 协议联调时，在选定命令追加 `--maas-mock-backends`。模拟分支不进入 GPU 生成队列，快速填写异步任务记录；不能用于验证真实供应商的排队、推理耗时或扩容能力。模拟请求仍须通过真实输入及媒体校验。

模型精度、步数、attention、并行和 VAE 后端仍由启动配置确定。真实生成不因 V2 协议改为其他推理框架或重复构建模型。每副本独占状态目录，程序用文件锁拒绝两个活动实例共享同一目录；同一副本重启继续使用原目录。

## 请求示例

T2VA，4 秒、768P、可见水印：

```json
{
  "model": "MiniMax-H3",
  "content": [{"type": "text", "text": "两个人交谈后起身，镜头缓慢后移。"}],
  "resolution": "768P",
  "duration": 4,
  "ratio": "16:9",
  "aigc_watermark": true
}
```

I2VA 将图片放入 content，role 为 first_frame/last_frame；R2VA 使用 reference_image/reference_video/reference_audio。媒体字段为嵌套对象，例如 `{"type":"image_url","role":"reference_image","image_url":{"url":"https://assets.example.com/ref.png"}}`。

IR 模拟，提交至 `POST /v2/h3_context_ir`；不传 resolution：

```json
{
  "model": "MiniMax-H3",
  "content": [{"type": "text", "text": "两个人交谈后起身。"}],
  "duration": 5,
  "ratio": "16:9"
}
```

2K 模拟沿用 generation 请求，仅将 resolution 改为 `2K`。两者都返回 `{"task_id":"..."}` 并使用相同查询/列表/删除接口。创建、单任务查询和下载的响应头带 `X-XGen-Mock: ir` 或 `2k`；开启模拟能力的实例在所有 V2 响应上带 `X-XGen-Mock-Enabled: ir,2k`，包括列表。列表中任务仍需逐项查询以确认后端类型。

## 状态、错误和生命周期

- 创建任务的 ID 由服务端生成。V2 不接收旧协议的 id/input/created_at，也不公开 seed、steps、GPU 精度等部署字段。
- 查询为 `{"task":{...}}`。成功视频使用对象 content.url，IR 使用 content.prompt；没有需要二次 JSON 解码的 output 字段，也不泄露服务器文件路径。
- 列表为 items/total。page_num 从 1 开始；本地默认 page_num=1、page_size=20，page_size 上限 1000。这是本地资源策略，**不是官网公布的默认值或最大值**。filter.task_ids 通过重复 query key 传递，其他参数不允许重复。
- DELETE：排队中取消并清理素材；成功/失败删除记录；运行中和已取消返回错误。删除记录不自动删除归档视频，也不触发退款或推理重试。取消与 worker 领取共用原队列锁。
- HTTP 参数错误为 400；能力未配置为 422；容量拒绝为 429；服务异常为 500。统一官方错误信封及 request_id；task.failed 的业务错误另放 task.error。内部诊断码不伪装成 MiniMax 的专有数字错误码。
- V2 元信息保留七天，旧接口的内存记录仍约保留两小时。重启时历史中的 queued/running 变为 failed，code=engine_restarted；不会重建模型去重跑旧任务。
- 下载地址使用本副本的 V2 路由，带 HMAC 签名，默认一小时过期；重新查询/列表会刷新 URL，下载无需内部 Bearer。签名密钥保存在状态目录并在重启后复用，不对外输出。EXTERNAL_BASE_URL/`--external-base-url` 应设为 MaaS 可达地址。其可用性仍依赖真实成片文件；平台应归档成片并管理存储生命周期。数据库过期或删除记录不自动清理视频文件。
- 真实视频 usage 返回输入视频计量秒数（本地向上取整）、输出请求秒数、总秒数和图片数；参考音频按各段之和四舍五入。不伪造厂商的折算 token 数；这些计量规则不能直接当作 MiniMax 账单。

## 回调及可配置项

callback_url 先接收 `{"challenge":"..."}`，本实现要求 HTTP 200、三秒内原样返回该 JSON 对象；随后通知体与查询相同。验证失败不入队，准备的媒体被清理。通知按顺序在独立线程发送，单次失败只记录日志，不重跑生成，也不改变任务终态。

官方说明未完整公开验证编码、签名及重试策略。本实现的 JSON challenge、无自动重发属于明确的本地约定，不宣称复刻所有托管服务细节。通知目标仅接受经检查的公网 HTTP(S) URL，不跟随重定向。DNS 解析仍受系统解析器时限控制。

| 配置 | 默认及含义 |
|---|---|
| `--maas-state-dir` / XGEN_MAAS_STATE_DIR | 默认 outputs/maas-state；每副本独立，重启复用 |
| `--maas-mock-backends` | 默认关闭；只启用明确标记的 IR/2K 测试替身 |
| XGEN_MAAS_API_KEY | 可选内部 Bearer token；仅约束 V2，不改变旧接口；不可写入仓库 |
| XGEN_MAAS_FILE_BASE_URL | 可选平台公网文件服务前缀；将 mm_file://ID 解析为前缀/ID 后复用受控下载；不是 MiniMax 托管文件的访问凭证 |
| XGEN_MAAS_URL_TTL_S | 本地签名下载 URL 默认 3600 秒，可设 1～604800 秒；不限制推理时长 |
| `--maas-queue-capacity` / `--maas-queue-timeout-s` | 10 条等待、600 秒；已派发任务不受排队超时限制 |
| callback challenge / 通知 | 3 秒；仅网络通知，DNS 例外见上文；不限制 GPU 生成时长 |
| XGEN_MAAS_MOCK_TIMEOUT_S | 180 秒；仅模拟测试卡 FFmpeg 的进程时限，与真实推理无关 |
| 媒体下载及探针 | 沿用 [原 MaaS 配置](../server/video/MAAS-API.md#下载探针与超时)，未新增真实推理截止时间 |

MaaS 对客户负责鉴权、租户及文件归属、配额、跨副本任务路由、聚合列表、计费和客户回调。引擎的七天记录只覆盖本副本；不应将轮询随机分发给未持有该任务的实例。平台的 task_id 可映射为引擎 task_id+pool+replica，不必强求两段 ID 完全相同。

## 与官网仍有区别的内容

1. 当前只支持一个 text 项；不会悄悄丢弃或拼接多段文本。每项上限以官方 CLI 的 7000 UTF-16 码元计数。
2. 保留本地静态图片、图片像素乘积、NUL/非法 Unicode、公网访问及下载时限等限制。R2VA 的 adaptive 选图规则和实际像素尺寸不保证与 MiniMax 的实现一致。
3. 纯参考音频暂沿用官方 CLI 的规则，需搭配图片或视频。尾帧独立输入按创建正文与指南支持；官方局部 role 描述仍有歧义。
4. 水印为本地可见 AIGC 标识；IR/2K 为明确模拟，未接真实供应商；再生成不实现。不能标注为“官网全部模型能力兼容”。
5. 引擎不复刻平台的账户计费、300 RPM/30 inflight 配额或内容审核；上层 MaaS 须实现相应策略。

## 后续替换 IR/2K

模拟实现集中在 [maas_v2_backends.py](../server/video/maas_v2_backends.py)。context_ir 接收原 V2 请求及已校验的 normalized 输入，返回 prompt；video_2k 接收同样信息及目标输出路径、尺寸、时长。原始 content 的顺序及已物化的媒体路径都可供适配器使用。

真实供应商协议确定后，将这两处替换为提交/查询/取消适配，并保持本地 task_id 和官网 V2 响应结构。若供应商异步返回，应持久化 provider_task_id 并独立轮询/接收状态；不要把长时间远端推理直接塞进当前快速完成的模拟分支。真实接入应移除模拟标识、接入真实用量并单独做失败/超时/取消测试。当前代码没有向任何第三方发送真实生成请求。

## 验证范围

新增 [test_maas_v2.py](../server/video/test_maas_v2.py) 覆盖真实 Flask 请求、原队列、持久化、回调、角色和类型检查、4 秒、五图/adaptive、Bearer、竞争取消、IR/2K 模拟及水印。实际使用 FFmpeg 编码/ffprobe 检查测试卡和水印视频；测试替身不代表真实模型生成。

本地 Windows 校验使用实际服务类源码 AST 和真实 Flask/Pillow/FFmpeg，将模型导入及设备操作隔离；没有加载模型或执行 GPU/NPU。常规环境应直接运行完整回归：

2026-11-10 本地结果为 **72 项测试、278 个子用例通过**。一项原有 CLI 子进程检查因缺少完整模型运行时未执行；本轮另直接调用真实 main() 验证新旧 CLI 开关及参数守卫。OpenAPI 引用、文档本地链接、编译、差异空白及改动文件脱敏另有静态校验。

```bash
CUDA_VISIBLE_DEVICES='' XGEN_DEVICE=cuda python -m pytest -q \
  server/video/test_maas_v2.py server/video/test_maas.py \
  server/video/test_maas_fl2va.py server/video/test_maas_artifacts.py
```

真实 GPU 4 秒、水印及完整步数成片仍需独立验收。协议与 CPU 测试通过不等于人工画面、音频或长时间服务稳定性已经通过。

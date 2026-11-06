# 视频生成服务

MiniMax-H3 的 NPU、CUDA、2K 和 MaaS 启动方式见 [H3 在线服务指南](../../docs/H3-ONLINE.md)。

基于 Wan2.2-T2V/I2V-14B 模型的视频生成服务，支持文生视频(t2v)和图生视频(i2v)。

## CUDA H3 的 MaaS 输入校验

平台对接所需的请求、响应、错误码、计时和探针语义见 [MaaS FL2VA 接口文档](MAAS-FL2VA-API.md)。

在已有的 H3 CUDA FL2VA 服务启动命令上添加 `--maas-fl2va-input`。模型配置使用 `MiniMax-H3`、`task_type=fl2va` 和 `num_inference_steps=50`，并按 [CUDA 部署教程](../../docs/H3-CUDA.md) 准备权重。这个模式固定使用 H3 Base、768P 横屏、5 秒、24 fps、50 次实际 DiT。省略开关时使用原有接口校验。启用后 `/params` 返回这套规格，`/config` 保持只读。

`POST /v1/tasks` 接受 JSON 对象 `{"id":"job-1","input":{...}}`，可选 `created_at` 使用 Unix 毫秒。`POST /generate` 直接接收 `input` 对象的内容。`prompt` 和 `first_frame` 必填，`seed` 默认为 42，`enable_audio` 默认为 `true`。固定字段可以省略，显式传入时必须匹配 `/params`。

- `prompt` 最多 2000 个 Unicode 码点，允许中文和换行。`seed` 为 0 至 2147483647 的 JSON 整数，`enable_audio` 为 JSON 布尔值。
- `first_frame` 使用单帧 JPEG 或 PNG 的 Base64 data URI。解码前文件最多 4 MiB，图像最多 8388608 像素，每边 32 至 4096 像素，长边与短边的比值最多 4。调用方可先从平台素材库取图并调整到限额内，再传给 xGen。
- 非法 JSON、字段类型、组合或坏图返回 `400`，超过请求体或素材资源上限返回 `413`，非 JSON 内容类型返回 `415`。错误响应包含 `error`、`field` 和 `message`。

启用该模式时，用 `--init` 启动并保持 SP 与实际 rank 数一致。`/health` 保留原有存活字段，增加 `state`、`ready`、`reason`、`phase`、`model_loaded` 和 `workers`。`/ready` 返回同一份就绪状态，只有 `ready=true` 时返回 `200`，其余返回 `503`。省略 `--maas-fl2va-input` 时，沿用原 `/health` 响应。

首次启动后，从内网直接提交一条符合规格的预热请求。首条视频保存成功，且各 rank 通过现有心跳确认后，`/ready` 才通过。正常生成中保持就绪。摘流、已捕获的 worker 错误、或单阶段达到时限会使就绪失败。加载、生成与保存的就绪时限分别为 600、1200、180 秒。空闲、派发与确认阶段仅记录状态。探针只读状态，超时不会由探针直接终止进程。

启用该模式时，`POST /v1/tasks` 对当前进程中仍保留的任务 ID 去重。同 ID、同规格返回现有任务，同 ID、不同规格返回 `409 task_conflict`。等待队列默认最多 10 条，已派发的任务另计。队列满返回 `503 queue_full`。任务自受理起等待满 600 秒后过期，过期只影响待派发任务。可用 `--maas-queue-capacity` 和 `--maas-queue-timeout-s` 调整这两个默认值。`DELETE /v1/tasks/<id>` 和 `POST /cancel/<id>` 可取消待派发任务，已派发时返回 `409`。取消时释放槽位，受理、查询、派发和定期清理时先移除已过期的队列项。

待派发任务的查询结果包含 `queue_elapsed_s`（自受理以来已等待的秒数）和 `queue_ahead`（排在前面的有效等待任务数，执行中的任务另计）。派发、取消或过期后，查询结果保留在 `timings_s.queue` 中的实际排队耗时。

配置 `FFPROBE_PATH` 或在 `PATH` 提供 `ffprobe`。服务将编码结果写入同目录的 `.pending-*.mp4`，核对 1344×768、24 fps、124 帧元数据及实际视频包数，并检查所需音轨包含数据包后，原子替换到最终路径。配置 `OBS_LOCAL_PATH` 时，还要确认对象可访问并用签名链接读到数据后，任务才成功。消费输出目录时只读最终的 `video_*.mp4`。

任务查询增加 `error_code`、`failure_stage` 和 `timings_s`。耗时使用秒，分别记录 `queue`、`inference`、`encode`、`delivery` 和 `total`。尚未完成的阶段可能缺失。常见终态原因包括 `queue_expired`、`cancelled_by_user`、`encode_failed`、`storage_failed`、`worker_lost` 和 `service_shutdown`。终态任务记录在实例内存中约保留 2 小时。

在已安装服务依赖的 CUDA Python 环境运行 CPU 路由测试：

```bash
CUDA_VISIBLE_DEVICES='' XGEN_DEVICE=cuda python server/video/test_maas_fl2va.py
python server/video/test_maas_artifacts.py
```

## 功能特性

- 异步 HTTP API 接口
- 支持文生视频 (T2V) 和图生视频 (I2V)
- 支持单卡和多卡并行推理
- 支持 5s 和 10s 视频生成
- 支持超分辨率 (SeedVR2)
- 支持帧插值 (平滑视频)
- 支持视频下载接口
- 支持任务状态查询
- 支持配置管理
- 支持推理指标统计
- 支持定时清理历史视频

## 目录结构

```
server/
├── video_server.py           # 服务主程序
├── start_video_service.sh    # 启动脚本
├── video_service_ctl.sh      # 服务管理脚本
├── config_wan22_t2v.json     # T2V 模型配置
├── config_wan22_i2v.json     # I2V 模型配置
├── Video_Readme.md           # 本文档
├── VIDEO_API.md              # API 详细文档
└── client_example.py         # 客户端示例
```

## 快速开始

### 1. 启动服务

在服务器容器内执行：

```bash
# 进入项目目录
cd /Ascend_cloud_aigc_poc/aigc_inference/torch_npu/x_gen/server

# 单卡 T2V
bash start_video_service.sh wan2.2-t2v-14b 1 5001

# 4卡 T2V (推荐)
bash start_video_service.sh wan2.2-t2v-14b 4 5006 --warmup

# 8卡 T2V (最高性能)
bash start_video_service.sh wan2.2-t2v-14b 8 5006 --warmup

# 8卡 I2V
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --warmup

# 8卡 + 10s 模式 + 预热
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --ten-second --warmup
```

### 2. 启动参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| model | 模型名称 | wan2.2-t2v-14b, wan2.2-i2v-14b |
| num_gpus | GPU 数量 | 1, 2, 4, 8 |
| port | 服务端口 | 如 5001, 5006 |

**选项参数：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| --devices \<ids\> | 指定 NPU 设备 | 自动选择 |
| --steps \<n\> | 默认推理步数 | 单卡 40, 多卡 20 |
| --ten-second | 启用 10s 视频模式 | 关闭 |
| --no-init | 跳过预初始化 | 预初始化 |
| --warmup | 启动时预热 | 关闭 |

### 3. 启动示例

```bash
# 单卡 T2V，端口 5001
bash start_video_service.sh wan2.2-t2v-14b 1 5001

# 4卡 T2V，带预热 (默认 NPU 4,5,6,7)
bash start_video_service.sh wan2.2-t2v-14b 4 5006 --warmup

# 8卡 T2V，带预热
bash start_video_service.sh wan2.2-t2v-14b 8 5006 --warmup

# 8卡 I2V，启用 10s 模式
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --ten-second --warmup
```

### 4. 检查服务状态

```bash
curl http://127.0.0.1:5006/health
```

### 5. 停止服务

```bash
pkill -f "video_server.py.*port 5006"
```

## API 接口

### 1. 健康检查

GET /health

### 2. 提交视频生成任务

POST /generate
Content-Type: application/json

**T2V 请求示例**：
```json
{
  "prompt": "A cat walking on a sunny beach",
  "width": 832,
  "height": 480,
  "frames": 81,
  "num_inference_steps": 40,
  "seed": 42
}
```

**I2V 请求示例**：
```json
{
  "prompt": "The person starts walking forward",
  "i2v_image_path": "./input_image.jpg",
  "width": 832,
  "height": 480,
  "frames": 81,
  "num_inference_steps": 20
}
```

**请求参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | string | **必填** | 文本提示词 |
| width | int | 832 | 视频宽度 |
| height | int | 480 | 视频高度 |
| frames | int | 81 | 帧数 |
| num_inference_steps | int | 40 | 推理步数 |
| seed | int | 随机 | 随机种子 |
| duration | int | 5 | 视频时长 (5 或 10) |
| save_fps | int | 16 | 输出帧率 |
| i2v_image_path | string | - | I2V 输入图片路径 |
| ten_second | bool | false | 启用 10s 视频生成 |
| adopt_sr | bool | false | 启用超分辨率 |
| resolution | int | 480 | SR 目标分辨率 |

### 3. 查询任务状态

GET /status/\<task_id\>

### 4. 下载视频

GET /download/\<task_id\>

### 5. 获取推理指标

GET /metrics

## 客户端示例

### Python 示例

```python
import requests
import time

BASE_URL = "http://127.0.0.1:5006"

# 提交任务
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "width": 832,
        "height": 480,
        "frames": 81,
        "num_inference_steps": 20,
        "seed": 42
    }
)
task_id = response.json()["task_id"]

# 轮询状态
while True:
    status = requests.get(f"{BASE_URL}/status/{task_id}").json()
    if status["status"] == "completed":
        print(f"Video: {status['result']['video_path']}")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    time.sleep(10)

# 下载视频
video_data = requests.get(f"{BASE_URL}/download/{task_id}").content
with open("output.mp4", "wb") as f:
    f.write(video_data)
```


# Video Generation Service API

## 服务端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | 推理统计指标 |
| `/generate` | POST | 提交视频生成任务 |
| `/status/<task_id>` | GET | 查询任务状态 |
| `/result/<task_id>` | GET | 获取任务结果 |
| `/tasks` | GET | 列出所有任务 |
| `/cancel/<task_id>` | POST | 取消任务 |
| `/params` | GET | 查看参数规范 |
| `/config` | GET/POST | 查看/更新配置 |

## API 使用示例

### 健康检查
```bash
curl -s http://127.0.0.1:5001/health | python -m json.tool
```

响应:
```json
{
  "status": "healthy",
  "model": "Wan2.2-T2V-A14B",
  "task_type": "t2v",
  "rank": 0,
  "world_size": 1,
  "is_initialized": true,
  "tasks": {
    "total": 10,
    "running": 1,
    "pending": 0,
    "completed": 8,
    "failed": 1
  }
}
```

### 提交 T2V 任务
```bash
curl -X POST http://127.0.0.1:5001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat walking on a sunny beach",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 40,
    "seed": 42
  }'
```

响应:
```json
{
  "message": "Task submitted successfully",
  "status": "accepted",
  "task_id": "92cd8423-f828-4abe-92cd-17dcd46de1f2"
}
```

### 提交 I2V 任务
```bash
curl -X POST http://127.0.0.1:5002/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The person starts walking forward",
    "i2v_image_path": "./input_image.jpg",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 20,
    "seed": 42
  }'
```

### 提交 10s 视频任务 (I2V + 10s + SR)
```bash
curl -X POST http://127.0.0.1:5001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat walking gracefully on a sunny beach",
    "i2v_image_path": "./input_image.jpg",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 6,
    "seed": 42,
    "ten_second": true,
    "adopt_sr": true,
    "resolution": 1080
  }'
```

> **10s 视频生成说明**:
> - `ten_second=true`: 启用两阶段生成 (5s base + 5s extension)
> - `adopt_sr=true`: 启用超分辨率 (SeedVR2)
> - `resolution`: SR 目标分辨率 (480/720/1080)
> - 推荐: 8 GPUs + 6 steps + 10s + SR 1080p
> - 自动启用: x=True, joint=True, inf_vram_blocks_num=1

### 查询任务状态
```bash
curl -s http://127.0.0.1:5001/status/<task_id> | python -m json.tool
```

响应:
```json
{
  "task_id": "92cd8423-...",
  "task_type": "video_generation",
  "status": "completed",
  "progress": 100.0,
  "result": {
    "video_path": "/tmp/generated_videos/video_xxx.mp4",
    "infer_time": 261.82
  },
  "created_at": "2026-04-09T16:11:12.960996",
  "started_at": "2026-04-09T16:11:12.961216",
  "completed_at": "2026-04-09T16:16:41.616901",
  "infer_time": 261.82,
  "error": null
}
```

### 获取推理指标
```bash
curl -s http://127.0.0.1:5001/metrics | python -m json.tool
```

响应:
```json
{
  "completed_tasks": 8,
  "avg_infer_time": 250.5,
  "min_infer_time": 180.2,
  "max_infer_time": 320.8,
  "model": "Wan2.2-T2V-A14B",
  "world_size": 1
}
```

## 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | string | **必填** | 文本提示词 (≤2000字符) |
| negative_prompt | string | "" | 负向提示词 (≤2000字符) |
| width | int | 832 | 视频宽度 (64-2048, **必须为偶数**) |
| height | int | 480 | 视频高度 (64-2048, **必须为偶数**) |
| frames | int | 81 | 帧数 (1-200) |
| num_inference_steps | int | 40 | 推理步数 (1-200) |
| seed | int | 42 | 随机种子 (≥0) |
| guidance_scale | float | 模型默认 | 引导强度 1 (0-100) |
| guidance_scale_2 | float | 模型默认 | 引导强度 2 (0-100) |
| i2v_image_path | string | None | I2V 输入图片路径 (I2V模型必填) |
| ten_second | bool | false | 启用 10s 视频生成 (两阶段: 5s+5s) |
| adopt_sr | bool | false | 启用超分辨率 (SeedVR2) |
| resolution | int | 480 | SR 目标分辨率 (480/720/1080) |
| frame_interpolation | bool | false | 启用帧插值 (平滑视频) |

### 参数校验

服务会对所有参数进行校验：

**错误示例** (参数无效，请求被拒绝):
```json
{
  "status": "error",
  "message": "Parameter validation failed",
  "details": {
    "errors": [
      "prompt is required and cannot be empty",
      "width must be even, got 833"
    ],
    "warnings": []
  }
}
```

**警告示例** (参数有效，但有潜在问题):
```json
{
  "status": "accepted",
  "task_id": "xxx",
  "message": "Task submitted successfully",
  "warnings": [
    "High resolution (1920x1080) may cause OOM, recommended: 832x480 or 480p/720p",
    "Large frame count (150) will significantly increase generation time"
  ]
}
```

### 类型自动转换

字符串形式的数字会自动转换为对应类型：
```json
{
  "prompt": "test",
  "width": "832",    // 自动转为 int
  "seed": "42"       // 自动转为 int
}
```

## 任务状态

| 状态 | 说明 |
|------|------|
| pending | 等待处理 |
| running | 正在推理 |
| completed | 完成 |
| failed | 失败 |
| cancelled | 已取消 |

## Python 调用示例

```python
import requests
import time

# 提交任务
response = requests.post(
    "http://127.0.0.1:5001/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "width": 832,
        "height": 480,
        "frames": 81,
        "num_inference_steps": 40,
        "seed": 42
    }
)
task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# 轮询状态
while True:
    status = requests.get(f"http://127.0.0.1:5001/status/{task_id}").json()
    print(f"Status: {status['status']}")
    
    if status["status"] == "completed":
        print(f"Video saved to: {status['result']['video_path']}")
        print(f"Inference time: {status['infer_time']:.2f}s")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(10)
```

## 性能参考

| GPU 数量 | 推理时间 (20步) | 加速比 |
|----------|-----------------|--------|
| 1 x 910B3 | ~260s | 1.0x |
| 2 x 910B3 | ~130s | 2.0x |
| 8 x 910B3 | ~60s | 4.3x |

> 实际性能取决于分辨率、帧数、模型等参数
> 
> 默认开启: `atten_laser`, `matmul_a8w8`, `rope_fused`


## CUDA H3 默认配方

CUDA H3 Base 的 T2VA / I2VA 默认采用 `fp8-bf16`：FP8 Linear、BF16 cuDNN attention / 通信、文本 TP8、PHAA2、50 层常驻。使用完整原始模块权重；默认 Base / MaaS 无需 packed NVFP4 文件。NPU 的默认配置保持不变。

从仓库根目录运行八卡服务：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas-fl2va-input \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

先填写 JSON 中的模型路径。普通 Base HTTP 使用 `h3_cuda_base.example.json` 并去掉 MaaS 开关。通过配置或启动参数 `--h3_cuda_precision none` 可选择旧 CUDA 路径，服务启动后不能在请求之间切换 precision。MaaS 仍固定 I2VA、5 秒、768P 和 50 次实际 DiT；排队、超时、健康探针及原子发布接口不变。

详细的精度、锁页内存成本、验证范围及历史性能见 [CUDA 默认配方](../../docs/H3-CUDA-PRECISION.md)。当前只将 5 秒、20 步 T2VA / I2VA 的人工审核结论记录为该组样例通过，不将解码成功等同于所有素材的精度通过。

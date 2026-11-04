# MiniMax-H3 在线服务

从仓库根目录执行。先按 [权重下载](../README.md#h3-权重下载) 准备模型和素材，并用 `python tools/collect_env.py` 确认当前环境能看到八张卡。以下示例绑定本机，默认端口为 5001。每次选择一个服务启动。

## 选择服务

| 服务 | 任务 | 分辨率 | 请求时长 |
|---|---|---|---|
| NPU Base | T2VA、I2VA、R2VA | 768P | 5 至 15 秒，接受整数 |
| NPU Refiner | T2VA、I2VA、R2VA | 2K | 5、10 或 15 秒 |
| CUDA Base BF16 | I2VA。T2VA、R2VA 为待实机验证的候选入口 | 768P | 5 至 15 秒，接受整数 |
| CUDA Refiner | I2VA | 2K | 接口接受 5、10 或 15 秒，已有 5 和 10 秒结果 |
| CUDA MaaS | I2VA，单首帧 | 768P、16:9 | 固定 5 秒 |

Base 服务生成 15 秒视频时，在请求里设 `"duration":15`。模型按 VAE 要求生成 362 帧，服务保存 360 帧和对应的音频。历史性能表里的 15 秒在线 768P 来自独立批测程序。

## NPU Base

先按 [模型准备](H3-DEPLOYMENT.md#4-模型与目录准备) 生成 Base Transformer 和 AdaLN 表。下面的服务接收 T2VA 和 I2VA。

```bash
XGEN_DEVICE=npu torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 \
  server/video/video_server.py --init --service-mode t2va \
  --config example/diffusers/configs/h3_npu_base.example.json --host 127.0.0.1 --port 5001
```

R2VA 单独启动。复制上述 JSON，修改 `task_type` 为 `ref2va`，`transformer_path` 为 `model-cache/base-ref2va`，`adaln_cache_file` 为 `model-cache/ref2va-51step.safetensors`。启动时将 `--config` 指向新文件，将 `--service-mode` 改为 `ref2va`。

## CUDA Base

以下为 BF16 稠密配置。先准备包含完整 BF16 Transformer 的 `weights/base`。它可用于 FL2VA 请求。需要验证 T2VA 或 R2VA 时，将 JSON 的 `task_type` 和启动时的 `--service-mode` 设为相应工作流。R2VA 使用包含 `transformer_ref/` 的完整模型目录。

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode fl2va \
  --config example/diffusers/configs/h3_cuda_base.example.json --host 127.0.0.1 --port 5001
```

### 5090 上的 15 秒 I2VA

八卡实测使用打包 NVFP4 权重和分层加载。复制 [CUDA MaaS 模型配置](../example/diffusers/configs/h3_maas.example.json) 为 `config-cuda-base-15s.json`，把 `h3_nvfp4_resident` 改成 `false`。在新终端执行下列命令，使用原 Base API。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XGEN_H3_CUDA_VAE_TILES=1 XGEN_PHAA_FP8=1 XGEN_PHAA_GATHER_PRIORITY=1
export XGEN_PHAA_DEVICE_JOIN=1 XGEN_PHAA_ASYNC_EXCHANGE=1
export XGEN_PHAA_DIRECT_RETURN=1 XGEN_PHAA_FP8_ROW_TILING=1 NCCL_MAX_CTAS=2
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --service-mode fl2va \
  --config config-cuda-base-15s.json --host 127.0.0.1 --port 5001
```

用 I2VA 请求示例，将 `duration` 设为 15。设 `num_inference_steps` 为 2 可以跑 1 次 DiT，适合检查生成和下载链路。本配置已通过 1 次 DiT 的 15 秒音视频验证。全驻留配置在同一用例的 VAE 解码阶段显存不足。

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

选择下面对应任务的 JSON，保存为 `request.json`。本地路径相对于服务的启动目录，远程客户端需要用服务可读取的素材。2K 请求把 `resolution` 改为 `2k`。显式传入 51 个调度点可运行 50 次 Base DiT。Base HTTP 在没有 AdaLN 表时，省略该字段会使用旧接口的默认值 50。

**T2VA**

```json
{"task_type":"t2va","prompt":"A small robot walks through a garden.","duration":5,"resolution":"768p","num_inference_steps":51}
```

**I2VA**

```json
{"task_type":"fl2va","prompt":"The camera follows the moving subject.","duration":5,"resolution":"768p","num_inference_steps":51,"first_frame":"inputs/first-frame.png"}
```

**R2VA**

```json
{"task_type":"ref2va","prompt":"The characters walk together.","duration":5,"resolution":"768p","num_inference_steps":51,"ref_path":["inputs/ref01.png","inputs/ref02.png","inputs/ref03.png","inputs/ref04.png","inputs/ref05.png"]}
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

准备 [CUDA 模型组件与 NVFP4 权重](H3-CUDA.md#环境和模型)，然后启动：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas-fl2va-input \
  --config example/diffusers/configs/h3_maas.example.json --host 127.0.0.1 --port 5001
```

MaaS 使用 `POST /v1/tasks`，首帧要放在 JSON 的 `input.first_frame` 中，格式为 JPEG 或 PNG 的 Base64 data URI。先直连实例提交一条预热请求，等 `/ready` 返回 HTTP 200 再接入平台流量。任务状态通过 `GET /v1/tasks/<id>` 查询，成功状态为 `succeeded`。 [预热和高性能开关](H3-CUDA.md#maas-在线服务) · [MaaS API](../server/video/MAAS-FL2VA-API.md)

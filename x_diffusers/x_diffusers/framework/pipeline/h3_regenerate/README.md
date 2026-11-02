# H3 Refiner：原 XGEN 架构中的可选超分后端

环境、资产准备、在线/离线命令和匿名性能表见 [完整部署教程](../../../../../docs/H3-DEPLOYMENT.md)。

**Experimental，默认关闭。** 开启后，H3 Base 生成 1344×768 的原始归一化视频/音频 latent，latent upscaler 与 Refiner 生成 2560×1440 视频，再复用 XGEN 的 VAE 时序解码和 `SaveAVStream` 编码、音频封装。不开启时不导入或加载 Refiner，不改变原来的推理、KV/DiT 缓存、SeedVR2、VAE 或设备调度。

没有独立 HTTP 服务、请求循环、请求条件缓存或 latent 恢复。CLI 使用原 `InferenceManager`；HTTP 使用原 `VideoInferenceManager` / `VideoGenerationService`。两者共享 `adaptor/h3_refiner.py`，Base 仍是 Manager 的 `pipe`，超分后端位于 `sr_pipe`。Refiner 内不创建 Manager。

## 参数与回退

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--h3_refiner` | false | 显式开启 experimental 后端 |
| `--h3_refiner_weights` | 空 | Regenerate 源模型包，含 `base_model/` 和 `latent_upsampler/` |
| `--h3_refiner_prepared` | 空 | 预处理后的 Refiner transformer、AdaLN 资产和 manifest |
| `--h3_refiner_seconds` | 5 | 5 / 10 / 15 秒 |
| `--h3_refiner_loading` | prefetch | 双槽分层预取；`sync` 为同步加载诊断路径，仍使用同一模型计算 |

这些参数只在启用后生效。在线配置也可使用同名 JSON 字段。开关、模型路径、工作流和量化/并行设置是启动级配置；HTTP 请求不能修改。关闭时在启动配置中设置 `"h3_refiner": false`、移除启动命令的 `--h3_refiner` 并重启进程，即恢复旧路径。暂不支持同一实例热切换新旧后端。同步加载只降低重叠复杂度，不承诺其时延与预取相同。

## 已实现范围与约束

| 项目 | 范围 |
|---|---|
| 硬件 | 昇腾 910B，8 ranks / SP8；其他硬件未验证 |
| 输入 | 原 `t2va`；`fl2va` 单张首帧；`ref2va` 五张不同参考图 |
| 输出 | 横屏 2560×1440，24fps；5 / 10 / 15 秒 |
| 输出/生成帧数 | 120/124、240/243、360/362；多出的帧是模型原生时序对齐，输出时裁剪 |
| Base | BF16 或 INT8，21/51 个调度点对应 20/50 次实际 DiT；AdaLN 表必须匹配 |
| Refiner | BF16，固定 3 步；预处理合并发布组合 LoRA 和 Sharp 组件 |
| VAE | BF16 解码、归一化权重 FP32；分布式空间 tile + 原时序解码逻辑 |
| 图片 | 可解码的本地图片；单图不超过 4096×4096 的像素总量；视觉 token 仍由原 H3 processor 处理 |
| 并发 | 原服务队列串行派发 GPU 请求；没有新增并行 GPU 请求调度 |
| 常驻复用 | 同一工作流、同一模型设置的不同请求/时长共用模型和 CPU master，不重复加载、量化 |

暂不支持最后一帧、多模态音视频参考、竖屏、任意输出尺寸或时长。`adopt_sr`（SeedVR2）、`cache_dit`、额外 LoRA、Turbo/SLA、插帧、FSDP、旧 infinite-vram hook、旧 VAE lightning 与此后端互斥；不能让两套 offload hook 接管同一模型。关闭 Refiner 时，这些原有能力的行为不变。

## 模型准备

复用现有 Python、PyTorch/torch_npu、Diffusers、Transformers、safetensors 与 FFmpeg，不增加推理框架、ComfyUI 或新的编解码库。

```bash
python -m x_diffusers.framework.pipeline.h3_regenerate.prepare \
  --weights ./Open-H3-Regenerate-2K \
  --output ./prepared-refiner
```

该工具需要 NPU 进行 AdaLN 预计算，处理结果在仓库外保存。它不修改源权重。Base 的匹配调度/剥离权重沿用现有 AdaLN 工具，或使用：

```bash
python -m x_diffusers.framework.pipeline.h3_regenerate.prepare_base_cache \
  --source ./base-transformer --eval-steps 50 --model-variant T2VA \
  --output ./base-51step.safetensors \
  --stripped-output ./base-transformer-stripped
```

Ref2VA 使用 `--model-variant Ref2VA` 和对应 transformer；T2VA/FL2VA 使用对应原生模型与工作流。不要混用不同模型、步数的预计算资产。资产清单和 LoRA 来源保留在模型目录；实际模型和张量不进 Git。

## 原 CLI 与离线常驻

运行新增示例 `example/diffusers/scripts/infer_minimax_h3_refiner.sh`，其调用的仍是原 `example/diffusers/infer.py`。使用者通过环境变量提供本地模型路径和输入，示例不含私人 prompt 或素材。

连续离线请求可以复用同一个原 Manager：

```python
from copy import deepcopy
from x_diffusers import init_env, parse_args, InferenceManager

args = parse_args()  # 同示例参数，明确带 --h3_refiner
init_env()
manager = InferenceManager()
try:
    for seconds in (5, 10, 15):
        request = deepcopy(args)
        request.h3_refiner_seconds = seconds
        request.save_path = f"output-{seconds}s.mp4"
        manager.infer(request)
finally:
    if manager.sr_pipe is not None:
        manager.sr_pipe.close()
```

这是原 Manager 的连续调用，没有额外批调度器。不同工作流分别启动；修改模型或量化配置需重启。

## 原 HTTP 服务

复制 `example/diffusers/configs/h3_refiner.example.json`，替换路径，按原服务部署方式启动八卡 `server/video/video_server.py --init --model minimax-h3 --config ...`。沿用原服务路径与依赖。不要使用原来少步数的 `--warmup`：已配置 AdaLN 的步数不可随意缩减。

向原 `/generate` 提交：

```json
{"prompt":"A small robot walks through a sunlit garden.","duration":5,"resolution":"2k","aspect_ratio":"16:9","seed":42,"enable_audio":true}
```

I2VA/R2VA 沿用原 `first_frame` / `ref_path` 字段，但工作流须匹配启动配置。任务 ID、排队、查询、结果下载沿用原 API。仅启用后端时，`/params` 才显示新规格。

编码、音频封装完成并发布最终文件后才标记请求成功；不会在 GPU 算完但文件未写完时提前成功。普通参数错误由原 API 返回；GPU/collective 失败后不重试，也不继续复用失败 worker，必须重新启动服务。不会静默返回 768P 伪装为 2K。

## 资源、性能与可观测性

Base、Refiner、VAE、upscaler 的 CPU master 常驻 RAM，HBM 分阶段使用。Refiner 预取只使用两个层缓冲槽；`sync` 不分配双槽预取缓冲。CPU master 的实际字节量通过 `H3_REFINER_ASSETS_READY` 日志输出；不能仅按 BF16 模型参数量估计整个进程 RAM。

相对 Base-only，新增 Refiner/upscaler 权重、CPU master、预取槽和 2K VAE 激活开销；CPU 权重、页缓存、RSS/PSS 和驱动 HBM 是不同口径。不能承诺所有请求都在固定显存阈值内。原生时间分块仍返回完整解码张量，15 秒的激活峰值需专项检查。

日志：`H3_REFINER_READY`、`H3_REFINER_COMPLETE`、`H3_REFINER_FAILED`。每个结果旁有 `.metrics.rankN.json`，记录初始化、实际 DiT 次数、各阶段事件、H2D 等待、分层计算、RGB8、D2H/队列、编码封装等待、请求周期、allocated/reserved 与峰值。跨 stream 的跨度不能简单相加当作 E2E。

本次优先复用原 `SaveAVStream`，保留原编码器默认参数；目前 RGB8 → CPU 传输为同步路径。旧独立服务的异步 FrameExporter/编码重叠和性能表不能直接套用。正确性确认后，再通过独立可选优化逐项恢复重叠能力。

## 验证

```bash
python x_diffusers/tests/test_h3_refiner_opt_in.py
python x_diffusers/tests/test_h3_refiner_runtime_cpu.py
```

CPU 测试覆盖默认 CLI、关闭开关后的原代码路径、规格/帧数、输入边界、实例级时序 padding、共享解码的张量一致性、真实 FFmpeg 成功/失败与原编码模式。Torch/FFmpeg 未安装时，对应运行时测试会明确跳过，不能视为通过。

NPU 回归仍需覆盖三个工作流、时长上限、连续复用、输出音视频和资源释放。旧版本 36 项技术验收不构成本次重构的全矩阵证明；视频主观画面、音频质量需要单独复核。保持 experimental，直到这些门槛满足。

第三方来源与本地修改见 `vendor/PROVENANCE.md`，upscaler 代码许可证见 `vendor/LICENSE-upscaler`；模型权重受各自许可证约束。

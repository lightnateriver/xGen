# NPU 离线 H3 benchmark

v0.6 推荐 Base 768P 默认 VC＋Ascend 融合、INT8 Linear、Native-stream 和 full-parallel，需先[安装相应算子](H3-NPU-FUSIONS.md)。Benchmark 配置记录所请求的 attention／融合列表；实际命中与回退仍需检查日志。运行前还需设置`H3_VAE_MODEL_PATH`指向可信原生`video_vae`目录，以及`H3_VAE_TRUST_REMOTE_CODE=1`。Native阶段含H2D、解码、RGB8/D2H及编码封装；每rank另记录`native_vae`子项，不把它当作旧INT8 VAE的同精度对照。2K Refiner仍使用独立旧后端。见[默认方案](H3-NPU-DEFAULT.md)。

从仓库根目录执行。使用 Linux、8 张空闲昇腾卡、`npu-smi`、FFmpeg 和 FFprobe。按 [NPU 部署教程](H3-DEPLOYMENT.md) 准备 Base 和 Refiner。分别为 FL2VA、Ref2VA 用 `--steps 20` 生成 AdaLN 表，对应 20 次实际 Base DiT。已有的 21 点表可直接复用。2K 另跑 3 次 Refiner。

建议使用 `--manifest` 读取素材包的 `cases.json`，为每条用例指定自己的 prompt 和按顺序排列的参考文件。相对路径从清单目录解析。`--plan-only` 检查矩阵和输入哈希，不启动八卡进程。简单试跑也可以使用 `--inputs` 目录，按下表命名文件。

| 文件 | 用途 |
|---|---|
| `t2va.txt`、`i2va.txt`、`r2va.txt` | 三种任务各自的 prompt |
| `first-frame.png` | I2VA 首帧，基线建议 1280×720 |
| `ref01.png` 至 `ref09.png` | 同一主题的九张 2560×1440 图，按清单顺序取前 1/5/9 张 |
| `video-{2,5,8}s.mp4`、`video-2s-with-audio.mp4` | 三个静音参考片段和与 2 秒静音版画面相同的有声版 |
| `audio-{2,5,8}s.wav` | 三个独立音频参考片段 |

准备好 `BASE_MODEL`、`BASE_TRANSFORMER`、`REFINER_WEIGHTS`、`REFINER_PREPARED` 等现有启动变量。默认从 `model-cache/base-t2va` 和 `model-cache/base-ref2va` 分别读取两种 Transformer。全矩阵如需单独覆盖 Ref2VA Transformer 路径，用 `BENCH_REF_TRANSFORMER`。

```bash
python tools/bench_h3_npu.py --manifest inputs/bench/cases.json --output outputs/h3-smoke \
  --suite acceptance \
  --base-adaln model-cache/t2va-21step.safetensors --ref-adaln model-cache/ref2va-21step.safetensors
```

默认只测 T2VA 5 秒 768P。`--suite acceptance` 用 6 条请求覆盖 5 个进程组。`--suite images` 跑 30 条主矩阵，`--suite multimodal` 跑五条探针，`--suite all` 跑完整 35 条。`--output` 指定结果父目录。每次运行会新建一个北京时间子目录，格式为 `YYYYMMDD-HHMMSS-微秒`，可以重复使用同一条命令。

完整 35 项、每项预热一次后正式测量：

```bash
python tools/bench_h3_npu.py --manifest inputs/bench/cases.json --output outputs/h3-benchmark \
  --suite all --warmup-per-case --request-timeout 3600 \
  --base-adaln model-cache/t2va-21step.safetensors --ref-adaln model-cache/ref2va-21step.safetensors
```

只检查链路、每项只跑一次时，去掉 `--warmup-per-case`。工具不会自动重试失败请求。测试报告示例见 [2026-11-04 至 05 的性能与内存数据](performance/h3-npu-offline-20261105.md)；那批使用的阶段性清理已移除，当前请求结束回收策略尚待完整设备验证。

## 多媒体探针与素材清单

| ID | 参考输入 | 输出 | 检查目的 |
|---|---|---|---|
| `video-silent` | 一段 2 秒静音视频 | 5 秒 768P | 视频参考 |
| `video-sound` | 同画面的 2 秒有声视频 | 5 秒 768P | 视频音轨参与条件编码 |
| `image-audio` | 一图、一段 5 秒音频 | 5 秒 768P | 图与独立音频混合参考 |
| `max-mixed` | 六图、三视频、三音频；视频与音频各为 2/5/8 秒 | 15 秒 768P | 12 文件总上限、两类媒体累计时长各 15 秒 |
| `mixed-2k` | 一图、一段 2 秒视频、一段 2 秒音频 | 5 秒 2K | 混合参考到 Refiner 完整链路 |

`max-mixed` 的 max 指文件总上限，不是九图。各媒体按清单顺序加载。图像主体与视频、音频应属于同一场景，prompt 按 H3 的任务格式编写，并包含轮流说话、运动与切镜等待检查内容。

`cases.json` 是包含全部 35 个用例的数组。每项格式如下，文件相对清单目录解析；ID、任务、时长、分辨率和有序输入文件名须与上述矩阵一致。提示词文件可以按用例独立指定。

```json
{
  "case_id": "mixed-2k",
  "task": "r2va",
  "duration_s": 5,
  "resolution": "2k",
  "prompt": "prompts/mixed-2k.txt",
  "inputs": ["inputs/ref01.png", "inputs/video-2s.mp4", "inputs/audio-2s.wav"],
  "status": "ready"
}
```

先检查输入及哈希，不启动 worker：

```bash
python tools/bench_h3_npu.py --manifest inputs/bench/cases.json --suite all \
  --output outputs/h3-preflight --plan-only
```

仓库不附本次实验的完整图片、音视频及成片。使用自己的合法素材即可运行；更换素材后的结果不能作为同输入性能复现。后续公开素材包适合放 Release 等下载位置，附来源、许可和 SHA-256，避免将大媒体重复写入 Git 历史。

每个工作组启动一次八卡进程，并复用组内权重。运行顺序为 **T2VA/I2VA 768P → T2VA 2K → I2VA 2K → R2VA 768P → R2VA 2K**。768P 的 T2VA/I2VA 共用 FL2VA 权重和进程，切换到 I2VA 时选择对应的 AdaLN 条件。2K 的 Refiner 在启动时固定工作流，T2VA 和 I2VA 分成两组。每组退出并核对进程残留后才启动下一组。

正式热态对比加 `--warmup-per-case`。每个用例先在同一进程内按相同配置、时长、分辨率和参考媒体预热一次，再执行计时请求。Markdown 和 CSV 分别标记「预热」「热态测量」或「形状首跑」。默认每个用例只跑一次，用于链路检查和观察首次遇到该输入的成本。完整矩阵开启预热会执行 35 条预热和 35 条正式请求。

## 内存观测

每组的 `groups/<组名>/memory.jsonl` 默认每 10 秒发起一次采样，记录可见物理卡的 HBM、torchrun 及其进程树的 RSS/PSS、系统 MemAvailable 和文件页缓存。另记录启动前、每条请求结束和组退出后的样本。HBM 峰值来自驱动采样，可能漏掉短于采样间隔的尖峰。RSS 会重复计算共享页。PSS 在进程采样不完整时留空，覆盖范围写入 `pss_coverage`。请求结束时单独核对八个实际 rank PID 的 PSS。CSV 同时记录实际采样间隔和单次采样耗时的最大值。

常规 benchmark 在上述请求中完成内存观测，矩阵结束并记录资源回收情况后结束任务。每条请求生成、保存和设备同步完成后，采集各 rank 的 allocated/reserved 与主机 PSS，再执行下一条。比较数值时同时查看时长、分辨率和参考媒体数量，首次分配、allocator 保留、页缓存和其他进程会影响样本。

启动时终端会打印 `report-<时间戳>.md` 的完整路径，可以在运行期间打开。每组先写实际配置，每条完成媒体检查后追加性能、HBM、RSS/PSS、结束显存和可播放的 MP4 链接。组结束后追加退出码、残留 PID 和各卡 HBM 的回落值。`summary-<时间戳>.csv` 同步更新，`run-<时间戳>.json` 记录启动时间和运行计划，各用例的 `result.json` 保留 rank 快照和媒体检查结果。

组首条请求 wall 包含初始化，初始化也单列。768P 和 2K 均核对八个 rank 的实际 DiT 次数。DiT 阶段采用设备事件计时，768P 的 VAE 时段包含流式编码的重叠部分，封装等待与 VAE 时段可能重叠。各阶段不能直接相加作为总时长。

768P 的原生 VAE 帧数为 124/243/362，2K 输出为 120/240/360。技术检查覆盖尺寸、帧率、帧数、音轨、音画时长差和完整解码。画面质量、参考一致性和听音需要人工复核。组启动前需要八卡空闲。请求默认最长 1800 秒，首次请求包含模型初始化。B2 全矩阵运行使用 `--request-timeout 3600`，为 12 文件多模态用例的条件编码、20 步 DiT 和成片导出留出时间。超时及 worker 异常会记录在报告中。请求不会自动重试，最终通过数和缺项写入 `final-status.json`。

## 每条视频结束后回收显存

H3 的在线、离线入口均在视频保存结束、编码和音频封装排空、请求输出引用释放后执行 Python 垃圾回收、设备同步和 PyTorch `empty_cache()`。每个 rank 都执行，保留常驻模型和 CPU 权重，不重新读盘、量化或加载模型。CUDA 与 NPU 使用各自的设备接口，其他模型的流程不变。

日志 `H3_REQUEST_MEMORY_RELEASE` 记录回收耗时和前后 allocated/reserved。离线「推理及保存」包含这部分耗时；在线现有 inference 字段仍表示生成阶段，回收耗时单列在日志中。benchmark 的结束显存快照在回收后采集，2K 模型自身 metrics 中的 `allocated_after`/`reserved_after` 是模型返回前的快照，早于入口回收。

此前的解码前、上采样前缓存清理逻辑及其环境开关已经删除。`empty_cache()` 归还未使用的 allocator 缓存，不卸载仍被引用的张量或模型，也不保证消除单条请求内部的峰值 OOM。首次请求和不同输入规模仍须实测。常规矩阵结束后不自动追加独立的内存重复专项。

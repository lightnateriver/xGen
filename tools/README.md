# 工具集

本目录包含模型转换、环境配置和第三方模型支持等工具。

## 目录结构

```
tools/
├── adaln_cache/          # AdaLN 预计算缓存
│   ├── precompute_adaln_cache.py  # 预计算脚本
│   ├── run_adaln_cache.sh          # 启动脚本
│   └── README.md                   # 使用文档
├── conversion/           # 模型转换
├── setup/                # 环境配置
│   ├── ffmpeg/          # FFmpeg 配置
│   └── usp/             # USP 配置
└── thirdparty_model/    # 第三方模型
    └── seedvr2/         # SeedVR2 超分模型
```

## AdaLN 预计算缓存

MiniMax-H3 AdaLN 参数预计算与权重剥离工具，详见 [adaln_cache/README.md](adaln_cache/README.md)。

```shell
# 预计算缓存 + 剥离权重（修改脚本内路径后运行）
bash tools/adaln_cache/run_adaln_cache.sh
```

## 模型转换

### Wan2.2 权重转换

```shell
python tools/conversion/wan2.2_convert_safetensors.py \
    --input_path ./original_weights \
    --output_path ./converted_weights
```

用于将原始 safetensors 格式转换为 X_GEN 兼容格式。

## 环境配置

### CUDA 和 NPU 环境采集

在实际运行 xGen 的容器和 Python 环境中，从仓库根目录运行：

```bash
python tools/collect_env.py
```

脚本自动选择当前 Python 能看到设备的后端。若设置了 `XGEN_DEVICE=cuda` 或 `XGEN_DEVICE=npu`，则检查指定后端。自动模式遇到两个可用后端时，有 Ascend 工具或安装环境线索则优先 NPU，否则优先 CUDA。驱动工具可见但 PyTorch 看不到卡时，仍会输出对应诊断信息。

两种报告均包含系统、Python、xGen 相关包、FFmpeg、FFprobe 和 PyTorch 可见卡数。CUDA 报告额外显示 NVIDIA 设备、驱动、显存、拓扑、`nvcc` 及 PyTorch CUDA 构建版本，有可见 GPU 时查询 cuDNN 和 NCCL。NPU 报告额外显示 `npu-smi info` 和 CANN 信息，CANN 通过 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 或 `PATH` 上的 `atc` 定位。

Linux 下的 `CPU and NUMA` 区块显示采集进程允许使用的 CPU、允许分配内存的 NUMA 节点，以及系统可见的每个 NUMA 节点对应的 CPU 列表。它反映运行脚本时的约束。CUDA 报告还保留 `nvidia-smi topo -m` 给出的 GPU 亲和性。

`Environment variables` 区块显示采集进程中白名单内的 xGen 开关、设备可见性、线程及 NCCL/HCCL 配置。路径和 `MASTER_ADDR` 只显示是否设置，CUDA 设备 UUID 脱敏，凭据变量不采集。除后端选择和设备可见性变量外，未设置的配置会省略。在同一个容器中沿用服务的环境配置运行脚本，即可看到对应快照。

PyTorch 至少看到一张所选后端的设备时返回 `0`，否则返回 `1`。缺失的诊断工具会单独提示。检查八卡部署时还需核对可见卡数与预期 SP，通信和模型推理通过冒烟测试验证。

采集方式参考 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py) 和 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend/blob/1e3c1e76bf9e129bbb49878fa8614886965e418d/collect_env.py)，使用 [Apache-2.0 许可](../x_base/x_base/operator/h3_attention/LICENSE.upstream)。脚本无需安装 vLLM 系列包。

### FFmpeg

```shell
bash tools/setup/ffmpeg/install.sh
```

### USP

```shell
bash tools/setup/usp/install.sh
```

## 第三方模型

### SeedVR2 超分模型

SeedVR2 用于视频超分辨率增强。

```shell
bash tools/thirdparty_model/seedvr2/seedvr2.sh
```

#### Patch 说明

| 文件 | 说明 |
|------|------|
| `seedvr2.patch` | 主 patch |
| `seedvr2_frame.patch` | 帧处理 patch |
| `seedvr2_server.patch` | 服务端 patch |

#### 使用方式

1. 克隆 SeedVR2 仓库
2. 应用对应 patch
3. 运行推理脚本

```shell
# 示例
git clone https://github.com/xxx/seedvr2.git
cd seedvr2
git apply ./seedvr2.patch
bash seedvr2.sh
```

## NPU 离线 H3 benchmark

`bench_h3_npu.py` 复用原有八卡推理入口，覆盖 T2VA、I2VA、R2VA 的 5/10/15 秒、768P/2K 和多媒体参考。每个工作组只加载一次模型，逐项追加性能与内存采样到带时间戳的报告。素材准备、全矩阵命令、预热与计时口径见 [benchmark 教程](../docs/H3-NPU-BENCHMARK.md)，历史结果见 [20 步性能表](../docs/performance/h3-npu-offline-20261105.md)。

## 插帧模型

插帧功能使用 IFRNet_S_Vimeo90K 模型，下载地址：

https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/tag/models

下载 `IFRNet_S_Vimeo90K.pth`，通过 `--frame_model_path` 指定路径。

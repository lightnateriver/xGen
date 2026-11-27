# 工具集

本目录维护环境采集、权重转换、离线模型资产准备和独立评测。模型机制、组件装配和请求执行分别由 x_base、framework、adaptor 负责，见 [架构](../docs/ARCHITECTURE.md)。

## CUDA 和 NPU 环境采集

在实际部署的容器和 Python 环境中执行：

```bash
python tools/collect_env.py
```

可通过 `XGEN_DEVICE=cuda/npu` 选择后端，否则自动探测。报告包含系统、Python、依赖、FFmpeg/FFprobe、设备可见性、驱动/工具链及通信环境。环境变量按白名单采集，路径只显示是否设置，设备 UUID 脱敏；凭据不采集。

至少有一张所选设备对 PyTorch 可见时返回 0，否则返回 1。多卡通信与模型运行需另行验证。采集实现参考 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py) 和 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend/blob/1e3c1e76bf9e129bbb49878fa8614886965e418d/collect_env.py)，来源许可保留于 [NOTICE 对应目录](../x_base/x_base/operator/h3_attention/LICENSE.upstream)。

## H3 资产准备

| 工具 | 职责 |
|---|---|
| `adaln_cache/prepare_cuda_adaln.py` | CUDA AdaLN 表与 stripped Transformer |
| `adaln_cache/prepare_h3_base_cache.py` | NPU Base 配对表、stripped 权重及可选 Turbo8 合并 |
| `prepare_h3_refiner.py` | Refiner LoRA/Sharp 合并、调制表及 manifest |
| `adaln_cache/precompute_adaln_cache.py` | 原始 AdaLN 缓存和权重剥离工具 |
| `adaln_cache/compare_adaln_weights.py` | 已准备权重/缓存的局部核对 |

推荐资产布局和命令集中在 [模型准备](../docs/H3-MODELS.md)，低层工具参数见 [AdaLN 工具](adaln_cache/README.md)。这些操作在正式推理前执行，不由逐请求入口自动生成大文件。

## 构建、转换与第三方适配

| 位置 | 用途 |
|---|---|
| `build_h3_optimizations.sh` | 目标 NPU 环境构建融合库，说明见 [NPU](../docs/H3-NPU-DEFAULT.md#operators) |
| `lora_merge/merge_lora.py` | LoRA 权重合并，参数见脚本 `--help` |
| `thirdpart_model/lightx2v_convert/` | LightX2V/Wan 权重转换与补丁 |
| `thirdpart_model/seedvr2/` | SeedVR2 集成补丁与脚本 |
| `thirdpart_model/usp/install_usp.sh` | USP 环境准备 |
| `setup/install_ffmpeg.sh` | FFmpeg 安装辅助脚本 |

第三方脚本应按其固定源码和目标运行环境使用；模型准备产物、素材和输出文件不提交到源码仓库。

## VAE 加速评测

[evaluate_vae.py](evaluate_vae.py) 将 Wan2.x VAE 的速度矩阵、输出对照和报告集中在工具入口，通过 `InferenceManager.infer_batch` 复用正常推理。使用已安装 xGen 的 CUDA/NPU 环境，在仓库根目录运行，例如：

```bash
torchrun --standalone --nproc_per_node=1 tools/evaluate_vae.py \
  --model Wan2.2-I2V-A14B --task_type i2v \
  --pretrained_model_name_or_path /path/to/Wan2.2-I2V-A14B-Diffusers \
  --batch_input_dir /path/to/cases --batch_sample_size 1 --seed 42 \
  --eval_mode speed --eval_report_dir /path/to/results \
  --eval_resolutions 832x480 --eval_frames 17 --eval_targets none,decoder
```

每个用例目录包含 `image.jpg` 和 `prompt.txt`。一次评测只抽样一次，所有组合使用相同用例及 seed。速度模式保留 VAE 内部的两次预热、五次重复计时，生成 `report_speed.csv`；这是编码/解码耗时，不能作为端到端推理耗时。

`--eval_mode accuracy` 生成对照视频、`prompt.txt` 提示词清单和 `任务描述.txt`，不自动计算精度指标。视频按分辨率、帧数和加速对象分目录保存，避免多个组合覆盖；未加速输出位于 `original/`，加速输出位于 `target-*/`。速度模式保留 `lighting-*/` 视频目录。

评测参数只由本工具解析。旧 `--batch_test_mode/resolutions/frames/targets/report_dir` 已替换为对应的 `--eval_*` 参数；输入与抽样统一使用 `--batch_input_dir`、`--batch_sample_size`。普通推理入口不再接受评测参数或重复计时开关。原有两份 [速度](../example/diffusers/scripts_self/vae_batch_test/batch_speed_test_wan2.2_14b_i2v.sh)、[输出对照](../example/diffusers/scripts_self/vae_batch_test/batch_accuracy_test_wan2.2_14b_i2v.sh) 启动示例已更新。

## 开发检查

```bash
python -m pip install -r requirements-quality.txt
python tools/check_quality.py
```

检查范围和完整运行环境测试的区别见 [开发检查](../docs/ARCHITECTURE.md#validation)。

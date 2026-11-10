# CUDA H3 精度参考配方：fp8-bf16

> v0.5 默认已改为 [Sage-FP8 全并行 Streaming + AdaLN 表](H3-V05-CUDA-UPGRADE.md)。本文保留旧标准方案的配置与历史验收；显式选择对应精度仍可使用。


这是保留的 CUDA 标准方案一，显式使用 `fp8-bf16`。历史 legacy decoder 的 5 秒、20 步固定样例已有人工精度认可；更换 decoder 的精度需独立判断。它复用现有 InferenceManager、SP、PHAA、文本 TP 和 VAE 接口，不新增服务或调度体系。更快、人工精度待审核的基础方案见 [Sage-FP8](H3-CUDA-SAGE.md)。

| 模块 | 当前精度 / 策略 |
|---|---|
| 主干 QKV、输出、FFN Linear | FP8 W8A8，200 个投影 |
| 文本编码器 TP Linear | FP8 W8A8、TP8 |
| attention / All-to-All | BF16 cuDNN dense / BF16 |
| AdaLN | 原 BF16 owner 投影、逐层广播；仅完全相同的最近输入可精确复用 |
| DiT 权重 | 全部 50 层常驻，VAE 阶段也保留 |
| 视频 decoder | 默认 native-stream：Linear 预存 FP16，其余参数 FP32、FP16 autocast；可显式选 legacy/lightning |
| 音频 VAE | FP32，整模块按需上卡，结束或异常后回 CPU |

本方案默认不包含 NVFP4 attention、FP8 通信、AdaLN gather、部分层常驻或区域编译。历史 NVFP4 / Refiner 路线仍可显式使用，不能将其精度结论套用到本配方。

## 模型与环境

需要八张支持 FP8 GEMM 的 CUDA GPU；本轮验证硬件为 RTX 5090。准备兼容 H3 的 Diffusers、Transformers、PyTorch CUDA、Triton 和支持 H264/AAC 的 FFmpeg / FFprobe。此配方不要求 FlashInfer、Sage、Sparge 或 Comfy-Kitchen。

模型目录须含完整原始 FL2VA/Base Transformer、文本编码器、视频 VAE、音频 VAE、处理器、调度器和 `modular_model_index.json`。不能用仅含 packed NVFP4 的目录或去掉 AdaLN 的 stripped Transformer。模型下载入口见 [README](../README.md#h3-权重下载)。

## 离线启动

```bash
export BASE_MODEL=./weights/h3
export PROMPT_FILE=./inputs/prompt.txt
export H3_VAE_MODEL_PATH=./weights/h3-native/video_vae
export H3_VAE_TRUST_REMOTE_CODE=1
H3_CUDA_PRECISION=fp8-bf16 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
H3_CUDA_PRECISION=fp8-bf16 bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p inputs/first-frame.png
```

每条命令各生成一个视频。脚本默认八卡、SP8、文本 TP8、PHAA2、50 次实际 DiT、seed42、1344×768、124 帧 / 24 FPS。快速复测可在命令前加 `H3_BASE_STEPS=20`。模型初始化不计入常驻请求性能。

直接 CLI 的 `--h3_cuda_precision auto` 按设备选择默认；可显式指定 `fp8-bf16`。该 CUDA 配方默认选择 native-stream，还需传 `--h3_vae_model_path`、`--h3_vae_trust_remote_code`；显式 `--h3_vae_backend legacy` 或 `lightning` 可覆盖默认。通用 CLI 仍需填写模型路径、任务、SP8 和文本 TP8 等参数。若已有显式 packed NVFP4、Refiner、LoRA、AdaLN 表或其他 attention 配方，auto 保留其旧路线；强制指定 fp8-bf16 时，对不兼容组合明确报错。

本页历史精度认可与性能使用 legacy decoder；更换 native-stream 后不继承该人工认可。新后端组件精度、初始化和显存数据见 [VAE 后端教程](H3-OMNI-VAE.md)。

回到原 CUDA 路径：

```bash
H3_CUDA_PRECISION=none bash example/diffusers/scripts/run_h3.sh cuda i2va 5 768p
```

或直接 CLI 加 `--h3_cuda_precision none`。NPU 的 auto 始终保留原路径；在 NPU 上显式指定 fp8-bf16 会在加载权重前报错。

## 在线部署

[精度参考 JSON](../example/diffusers/configs/h3_cuda_bf16.example.json) 显式使用 fp8-bf16 和 owner AdaLN；Base / MaaS 默认示例已改为新全并行方案。修改模型路径后启动本页参考方案：

```bash
XGEN_DEVICE=cuda torchrun --standalone --nproc_per_node=8 \
  server/video/video_server.py --init --maas \
  --config example/diffusers/configs/h3_cuda_bf16.example.json --host 127.0.0.1 --port 5001
```

普通 Base HTTP 去掉 `--maas`，并使用 Base 示例配置。MaaS 接受 768P、5～15 秒的 T2VA/I2VA 和独立部署的 R2VA，步数由启动配置决定；原队列、超时、健康探针、原子发布和请求结构不变。precision 只能作为启动配置；常驻 worker 不允许在请求间静默切换配方，改变时须重启服务。

使用干净的启动终端，清除旧配方的 `XGEN_PHAA_FP8=1` 等环境变量。新配方要求 BF16 通信，会拒绝该冲突，不会悄悄改成 FP8 通信。服务也采用同样的 RGB8 / veryfast 保存策略。接口和预热见 [在线教程](H3-ONLINE.md#cuda-maas)。

## 可选输出 AdaLN 融合

`XGEN_H3_FUSED_OUTPUT_ADALN=1` 是默认关闭的 CUDA fp8-bf16 实验开关。它复用现有 indexed affine 算子处理输出 AdaLN，避免在 SP gather 后展开完整序列的 scale / shift 和乘加中间张量。RMSNorm、投影、AdaLN 缓存、BF16 各次舍入、SP / PHAA 及权重常驻策略保留；不增加依赖。

```bash
XGEN_H3_FUSED_OUTPUT_ADALN=1 H3_BASE_STEPS=20 \
  bash example/diffusers/scripts/run_h3.sh cuda t2va 15 768p
```

I2VA 改为 `cuda i2va 15 768p inputs/first-frame.png`，并使用对应 prompt。关闭时取消设置该变量，再启动进程。该开关仅在 fp8-bf16 配方安装；NPU 及其他旧路径不安装输出融合。启动日志出现 `output AdaLN indexed affine fusion enabled` 表示已接入。它只降低该运算的临时显存，不卸载常驻权重，不保证任意形状均无 OOM。

算子回归见 `x_base/tests/test_h3_output_adaln.py`：CUDA BF16 逐位对照覆盖普通投影及缓存命中路径，另核验临时显存峰值。整条链路的可用性和新时长的人工画面 / 听音结论须分开记录。

标准方案二与独立启动方法见 [CUDA Sage-FP8](H3-CUDA-SAGE.md)。它显式选择，复用本配方的 FP8 Linear、常驻权重和 VAE 链路；Sage attention 及 FP8 通信的成片精度仍需单独人工审核。

## 内存与后处理

decoder 复用 Diffusers ModuleGroup 保存约 4.52 GiB 锁页 CPU 权重 / rank，八卡约 36.1 GiB；这是该 CPU 副本的大小，并非相对旧方案净增内存的实测值。DiT 阶段 decoder 不驻留 GPU，解码阶段加载其设备副本，结束后切回已有 CPU master，不回传未修改的权重。DiT 的 50 层始终保留在 GPU。

完成的浮点时间块移至 CPU 暂存，空间 tile 与重叠融合仍按原 GPU / dtype 执行，随后按八帧小段完成像素处理。GPU 上先转 RGB8 再 D2H，最后用 CPU libx264 的 veryfast / CRF10 编码并封装音频。RGB8 是像素格式；换编码 preset 会改变码流。

视频 decoder 前同步 / GC / empty_cache 仅回收空闲分配，不释放活跃模型，也不保证任意长视频无 OOM。已有每条成片后的显存回收保留。日志记录 precision、投影数、常驻块数及清理前后 allocated / reserved。

## 验证范围与性能

用户人工认可的历史样例：八卡 RTX 5090、5 秒档、768P、实际 20 次 DiT，T2VA 56.752 s、I2VA 72.669 s；取八 rank 最慢常驻请求至 MP4 保存，不含初始化和客户端下载。一条完整 1 步 T2VA 预热后测两项，I2VA 为首次图像条件请求。不是 HTTP SLA，也不能外推为所有素材的质量保证。

这条精度可接受路径尚未达到 T2VA <50 秒。新默认没有包含快于 50 秒但人工精度不可接受的 NVFP4 实验。本轮 10/15 秒、R2VA、2K 和不同 GPU 没有继承上述精度结论；R2VA / 2K 继续使用既有路线。整理后代码的回归结果单独记录，不冒充新性能数据。

算子、VAE 和默认配置回归不需要下载模型权重，在已安装 xGen / H3 运行依赖和 pytest 的环境执行：

```bash
XGEN_DEVICE=cuda PYTHONPATH=./server/video:./x_base:./x_diffusers \
  python -m pytest --noconftest --import-mode=importlib -q \
  x_base/tests/test_h3_cuda_precision_ops.py \
  x_diffusers/tests/test_h3_cuda_precision_config.py \
  x_diffusers/tests/test_h3_cuda_decode_chunks.py server/video/test_cuda_precision.py
```

无 CUDA 时部分算子测试会跳过；CPU 测试不能替代真实八卡出片。整理后离线及 MaaS 验证结果见 [性能与回归记录](performance/h3-cuda.md#整理后代码的默认路径回归)。

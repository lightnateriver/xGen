# H3 硬件无关调度

基线为 GitHub v0.4 `bc1209305717762419ba89a117db21b876c289a3`。适配本身不替换已有 CUDA/NPU 算子；v0.5 NPU Base推荐入口默认开启以下通用调度，CUDA维持独立预设。

## 开关与边界

T2VA/I2VA/R2VA Base 768P共用`--h3_features full-parallel`。推荐NPU入口同时默认Native-stream、AdaLN和Laser INT8；详见[NPU默认方案](H3-NPU-DEFAULT.md)。以下R2VA示例另指定FFN chunk2048，T2VA/I2VA保持0：

```bash
--h3_features full-parallel --h3_refiner_ff_chunk_size 2048
```

CUDA 原有 `--h3_cuda_preset sage-fp8-full-parallel` 继续有效。希望使用原生 BF16 CUDA Linear 时显式选择 `--h3_cuda_precision none`；NPU 保留原有 `--matmul_quant int8 --atten laser` 等配置。

| 能力 | CUDA / NPU 共用逻辑 | 保留的边界 |
|---|---|---|
| 本地输入投影、本地输出头后 gather | 是 | 保留原有输出头精度 |
| 请求内固定条件投影复用 | 是 | 每个请求重新计算，DiT 后释放；不是跨请求缓存 |
| 输入 Token Refiner SP | 是 | SP2/4/8，BF16 通信；NPU 保留原有 attention 选择 |
| 输入 Token Refiner FFN 分块 | 是 | 不改权重或精度，按 token 分块 |
| Qwen3-VL 视觉编码 SP | 是 | SP2/4/8，按图像边界做 attention，保留 DeepStack |
| 参考图／视频 VAE tile 编码并行 | 是 | 不规则或小 tile 回落原生编码；保留原始拼接 |
| 参考音频 halo 编码、长音频解码并行 | 是 | 解码沿用 10/15 秒 latent 范围；5 秒回落原生 |
| DiT 常驻／条件期卸载／参考编码后恢复 | 是 | `--h3_dit_residency resident\|conditioning\|reference` |
| DiT TP/CP、同组 TP8/CP8、TP2/SP8 | 是，原生 BF16 路径 | NPU DiT TP 目前不支持 INT8 Linear 分片；TP 布局仍为实验性 |
| 进程组预热、stream、同步与显存统计 | 是 | 通过 `torch.get_device_module` 选择运行时 |
| FP8 PHAA 打包、cuDNN attention、FP8/FP4 GEMM、Triton 融合 | 否 | 属于硬件算子，继续由 CUDA 路径实现；NPU 不静默开启 |
| 2K Refiner 链路中的 Base 条件优化 | 共用计算逻辑 | 继续由原有协调器管理权重，禁用第二套 offload；全链路性能待实测 |

`full-parallel` 开启本地输入／输出、条件复用、视觉／VAE／音频编码并行、输入 Refiner SP 和长音频解码。默认 DiT 常驻、输入 Refiner 使用 BF16 FFN；**不自动改变 DiT TP/CP 布局、量化精度、attention 算子、VAE decoder 或 FFN chunk 大小**。后者需要单独指定。

通用参数名如 `--h3_local_input`、`--no-h3_vision_sp`、`--h3_input_refiner_ffn bf16` 与已有 `h3_cuda_*` 拼写共用 argparse 目标。HTTP 启动配置 JSON 同样接受通用拼写；冲突别名会报错。服务启动后不能按请求改变模型布局、精度或这些优化开关。

## 复现与验证

组件数学验证不读取模型权重：

```bash
export XGEN_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/x_base:$PWD/x_diffusers"
torchrun --master_addr=127.0.0.1 --master_port=29761 --nproc-per-node=4 --max-restarts=0 \
  tools/validate_h3_portable.py --device npu --output outputs/portable-components
```

`tools/validate_h3_tp_cp.py --device npu --cp 2 --output ...` 可用于四卡 TP2/CP2 的 BF16 分片验证。CUDA 继续保留原有 FP8 验证分支。组件误差通过不能代替训练权重全链路或人工视频质量验收。

2026-11-10 已完成：177 项 CPU 回归通过，1 项跳过；11 项纯配置检查通过；NPU 四卡小模型的输入 Refiner（3/33/259 token，含尾部 padding 和 FFN 分块）与视觉编码（两张图、DeepStack、尾部 padding）对照通过。另有 NPU 双卡的 BF16 TP2、训练权重 Audio VAE 编解码、Video VAE tile 编码对照通过，覆盖短视频和小图回落；样本误差不代替全负载验证。实际视频性能证据单独保存在本地实验目录，不能用历史八卡数据代替本轮四卡测量。

回退：停止自己的 worker，去掉新增开关，再以原有配置启动。权重和原有模型目录不变；不同策略使用独立 manager，不热切换已构建模型。

NPU INT8 VAE decoder 的权重迁移在参考编码根 offload hook 注册前完成，避免 pipeline.to() 跳过整个 VAE 后 decoder 仍留在 CPU。可用 `tools/validate_h3_portable_vae.py --device npu --quant-decoder ...` 单独验证编码到解码的设备状态与数值一致性。

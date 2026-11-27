# H3 CUDA 部署与回退

[文档目录](README.md) · [资产准备](H3-MODELS.md#adaln-cuda) · [视频命令](H3-VIDEO-QUICKSTART.md#cuda)

<a id="defaults"></a>
## 默认方案

八卡 Base 768P 推荐入口 `cuda` 和 `cuda-5090` 使用 `sage-fp8-full-parallel`、native-stream 和 BF16 AdaLN 表，支持 T2VA/I2VA/Ref2VA 的配套模型。

| 组件 | 设置 |
|---|---|
| 主 DiT | FP8 Linear，Sage attention，SP8，50 层常驻 |
| 文本 | FP8 TP8 |
| 通信与输出 | PHAA2、同步 FP8、peer Align16、直接打包、本地输入/输出投影 |
| 条件 | 请求内复用，Token Refiner SP/cuDNN/FP8 FFN，视觉/VAE/音频组件并行 |
| AdaLN | 20 次实际计算的配对 BF16 表与 stripped Transformer |
| 视频 VAE | native-stream，Linear FP16 存储，其余原生参数 FP32、FP16 autocast |
| 音频 VAE | FP32 加载、输入和计算，不随 DiT 量化降低精度 |

使用匹配的 PyTorch CUDA、Triton、SageAttention、H3 Diffusers 与 Transformers；环境检查见 [模型文档](H3-MODELS.md#environment-cuda)。完整命令只在 [视频](H3-VIDEO-QUICKSTART.md) 和 [在线](H3-ONLINE.md) 维护。

<a id="precision"></a>
## 精度选择

| `h3_cuda_precision` | 主要行为 |
|---|---|
| `fp8-sage` | FP8 Linear、Sage attention，与全并行预设分别控制 |
| `fp8-bf16` | FP8 Linear、BF16 cuDNN attention 与 BF16 通信，作为显式精度参考 |
| `none` | 保留原低层 CUDA 路径；packed/Refiner 等组合有各自要求 |
| `auto` | 按设备和已有配置解析，不强行覆盖显式 packed/LoRA/Refiner 路线 |

全并行预设是多个能力的配置集合，不是另一种权重格式。直接指定旧精度时也应显式取消预设，避免残留其他能力。

<a id="rollback"></a>
## 独立回退

保留默认 Sage 全并行，仅改用原始 AdaLN 权重：

```bash
H3_BASE_STEPS=20 H3_CUDA_ADALN_MODE=owner \
  bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

关闭全并行预设，使用完整原始 Transformer 和 BF16 attention 参考方案：

```bash
H3_BASE_STEPS=20 H3_CUDA_PRESET=none H3_CUDA_PRECISION=fp8-bf16 H3_CUDA_ADALN_MODE=owner \
  bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

owner 需要保留未剔除 AdaLN 的原始权重，服务配置中同时移除表目录或固定表路径。`h3_cuda_bf16.example.json` 提供相应服务配置。保留 Sage 基础能力而取消全并行时，用 `H3_CUDA_PRESET=none H3_CUDA_PRECISION=fp8-sage`。

单项布尔能力可通过 `--` 后的 `--no-h3_cuda_*` 关闭，服务用启动 JSON 的布尔值；并行、驻留、组件编译和后端改动后重启。详见 [统一配置](H3-CONFIGURATION.md#switches)。

<a id="optional"></a>
## 可选布局和后端

TP/CP、共享 TP8/CP8、TP2-SP8 均为显式实验布局，不在默认 SP8 方案内。配置前核对 [并行约束](H3-CONFIGURATION.md#parallel)；表模式只支持其规定的原生 SP 路径。

Packed NVFP4 和 CUDA 2K 使用独立资产及依赖，见 [模型](H3-MODELS.md#optional-assets) 和 [视频 2K](H3-VIDEO-QUICKSTART.md#refiner)。不会在 OOM 后自动改用这些路径，也不会自动裁图或改变精度。

原生 VAE 可显式选择 legacy/lightning；Torch Compiler 只编译可支持的 decoder tile 子图，见 [VAE](H3-CONFIGURATION.md#vae)。Sage、精度、PHAA、驻留和 decoder 应作为独立设置核对，不能由一个开关推断其他状态。

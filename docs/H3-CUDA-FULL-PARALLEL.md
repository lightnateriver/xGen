# Sage-FP8 全并行

v0.4 的完整配置、独立开关、在线／离线启动和回退方法统一到 [RTX 5090 R2VA 全并行教程](H3-CUDA-COMPOSABLE.md)。

优选入口为 `run_h3.sh cuda-5090 r2va`，对应 `--h3_cuda_preset sage-fp8-full-parallel`。主 DiT 使用 SP8、Sage 和 FP8 Linear，50层权重在条件编码、DiT和VAE解码阶段均保留在GPU。全程常驻是新版默认；原条件阶段offload仅作为显式备选，不是此预设的必要条件。

旧 `cuda`／`npu` 启动和默认精度不变。TP/CP为独立实验选项，不包含在全并行SP8预设内。

CUDA 推荐 T2VA/I2VA/R2VA 的视频 decoder 默认已改为 native-stream，需可信原生 `video_vae` 目录和显式授权。旧精度配方继续保留，后端可用 `--h3_vae_backend legacy|lightning|native-stream` 覆盖；原 NPU 默认不变。历史整网表使用 legacy decoder，新默认不继承其性能和精度结论。

[性能与验证范围](performance/h3-cuda5090-full-parallel.md) · [在线服务](H3-ONLINE.md)

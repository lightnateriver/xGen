# Sage-FP8 全并行

本页保留旧链接入口。完整配置、独立开关、在线／离线启动和回退方法统一到 [RTX 5090 全并行教程](H3-CUDA-COMPOSABLE.md)，当前 20 步命令见[视频教程](H3-VIDEO-QUICKSTART.md)。

优选入口为 `run_h3.sh cuda-5090 r2va`，对应 `--h3_cuda_preset sage-fp8-full-parallel`。主 DiT 使用 SP8、Sage 和 FP8 Linear，50层权重在条件编码、DiT和VAE解码阶段均保留在GPU。全程常驻是新版默认；原条件阶段offload仅作为显式备选，不是此预设的必要条件。

自 v0.5 起，八卡 CUDA Base 的 T2VA/I2VA/R2VA 推荐入口统一采用该预设；`cuda` 与 `cuda-5090` 的 Base 768P 默认相同。TP/CP 为独立实验选项，不包含在全并行 SP8 预设内。

CUDA 推荐 T2VA/I2VA/R2VA 的视频 decoder 默认 native-stream，需可信原生 `video_vae` 目录和显式授权。旧精度配方继续保留，后端可用 `--h3_vae_backend legacy|lightning|native-stream` 覆盖；NPU 当前推荐见 [VC＋Ascend 融合 INT8 全并行 Streaming](H3-NPU-DEFAULT.md)。历史整网表使用 legacy decoder，新默认不继承其性能和精度结论。

[性能与验证范围](performance/h3-cuda5090-full-parallel.md) · [在线服务](H3-ONLINE.md)

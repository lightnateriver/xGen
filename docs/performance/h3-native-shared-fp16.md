# Stream backend：CUDA/NPU 统一 Linear FP16 存储

2026-11-11。继续使用已有 `native-stream`，仅去掉 Linear FP16 存储对 CUDA 和 DiT 精度配方的限制；Conv、Norm 等保留 FP32。不恢复已回退的旧 decoder CPU 放置修改，不新增组件或后端。Legacy/Lightning 保持原行为。

环境：8×Ascend 910B1，64 GiB/卡；driver25.5.1，CANN9.1.0，torch2.10.0+cpu，torch-npu2.10.0.post4，diffusers0.40.0，transformers5.14.1，mindiesd3.1.0。镜像与权重不变。

R2VA 单张原始2560×1440图片；1344×768，124帧/24fps，seed42；SP8/text TP8/PHAA2；DiT INT8+Laser+fused RoPE；Base实际20步，无2K Refiner。保留full-parallel与输入Token Refiner FFN chunk2048。新旧两组均使用Native-stream、vae_matmul_quant none，相同输入和编码参数；各一次同形状预热、一次正式请求，初始化单独记录，无重试。

| 指标 | 原FP32存储 | 共享FP16 Linear存储 |
|---|---:|---:|
| 请求E2E | 78.189 s | 77.644 s |
| DiT | 61.993 s | 63.274 s |
| VAE H2D | 1.080 s | 0.390 s |
| VAE decode/output | 4.897 s | 4.504 s |
| allocated峰值 | 40.137 GiB/卡 | 31.121 GiB/卡 |
| reserved峰值 | 43.010 GiB/卡 | 34.219 GiB/卡 |
| 进程组采样PSS峰值 | 241.751 GiB | 243.130 GiB |

各耗时独立取八rank最大值，不能相加还原某个rank的E2E。allocated峰值由请求内PyTorch高水位取得，PSS/HBM为周期采样。

allocated降低约9.016 GiB/卡（22.5%）；E2E只减少约0.545s（0.7%），每组仅一条正式样本，不宣称稳定的吞吐或时延加速。146个Linear使用FP16存储；原生decoder参数/buffer字节从9,694,132,608降至4,847,686,016。旧decoder在两组中均保持约4.51GiB设备权重且dtype相同，因此不与上轮保留CPU放置修复时的35.62GiB峰值混算。权重字节减少不代表进程PSS一定同步减少。

117项CPU/配置回归通过，1项checkpoint设备冒烟跳过。四个生成请求均由八rank记录确认20次实际DiT、每组model_builds=1、后续请求复用。四条MP4的视频与音频完整解码通过；人工画质、音频验收待完成。真实复测为NPU，不外推CUDA性能或长期无泄漏。两组进程均退出，结束检查八卡无计算进程。

原始启动命令、环境、逐rank指标、内存监控及视频随实验记录保留。基线为本版通用适配、FP32原生decoder存储，未改变旧decoder驻留策略；共享入口的核心改动仅为`fp16_linears=True`。不需要新增命令行参数，继续选择`--h3_vae_backend native-stream --vae_matmul_quant none`并提供既有video_vae路径及trust参数。

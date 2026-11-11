# Sage-FP8 全并行 Streaming：768P 历史矩阵

新增统一BF16 AdaLN表模式的[在线Align16性能对比](h3-cuda-phaa-align16.md)，覆盖同样12项组合。该表记录服务端推理和HTTP指标，本页继续保留离线历史口径。

采集日期：2026-11-11。8×RTX 5090、原生 SP8、文本 FP8 TP8、PHAA2、50 层常驻、Sage attention、FP8 DiT Linear／双向通信、SAP、本地输入、条件复用及组件并行；视频后端 native-stream，seed42。每项完整预热20步，再正式20步，客户端串行执行。

这些数据来自本次合并前的 CUDA 开发树，包含当时未提交的表和 FL2VA 适配，不能冒充合并后提交的完整性能重测。Torch2.12／CUDA13、Diffusers0.40、Transformers5.15、Sage2.2；原生 VAE Linear FP16存储／FP16 autocast。

12项正式视频均通过全片视频／音频解码、帧数／尺寸／音轨检查；本地媒体哈希和8rank记录完整。人工完整动态、口型及听音审核待完成；单图15秒抽帧提示身份漂移，技术通过不代表精度认可。原始素材和过程日志保留在本地，不进入公开仓库。

## 正式性能

单位秒；峰值为每卡 PyTorch allocated 的最大值（GiB），不是驱动 HBM。E2E 为常驻 Manager 从输入处理到保存及回收／同步，不含初始化、预热、HTTP排队／网络或事后哈希／FFmpeg审计。首9项owner，后三项五图为table-only，**不能跨行归因查表收益**。

| 用例 | AdaLN | 常驻 E2E s | DiT s | 峰值 allocated GiB/卡 |
|---|---|---:|---:|---:|
| t2va-5s | owner | 43.23 | 34.90 | 29.07 |
| t2va-10s | owner | 71.06 | 60.90 | 29.12 |
| t2va-15s | owner | 134.09 | 118.86 | 29.16 |
| i2va-5s | owner | 33.23 | 26.09 | 29.08 |
| i2va-10s | owner | 81.98 | 71.92 | 29.13 |
| i2va-15s | owner | 130.13 | 118.65 | 29.18 |
| r2va-1img-5s | owner | 57.20 | 47.16 | 29.15 |
| r2va-1img-10s | owner | 92.05 | 80.49 | 29.20 |
| r2va-1img-15s | owner | 157.21 | 142.21 | 29.25 |
| r2va-5img-5s | table | 133.69 | 114.07 | 26.50 |
| r2va-5img-10s | table | 214.60 | 192.03 | 26.55 |
| r2va-5img-15s | table | 298.69 | 273.80 | 26.60 |

原始精度：[精度参考配方](../H3-CUDA-PRECISION.md)；表的精度与空间边界：[AdaLN说明](../H3-CUDA-ADALN.md)。历史 legacy decoder 数据继续保留在[原全并行表](h3-cuda5090-full-parallel.md)。[CSV](h3-cuda-streaming-matrix.csv)

## Max 容量边界

另测直接离线输入7000个UTF-16码元、9张320×1280图片及3段各5秒视频（总12文件），short-edge保持2048。配置1步，**进入DiT前已OOM，实际DiT完成0步**；没有成片或E2E性能。

9张图片归一化到2048×8192，视觉patch589824行；视觉SP之前的位置插值构造全量FP32临时张量，单次申请约10.125GiB。错误时已分配约28.87GiB、空闲1.57GiB、reserved未用约49MiB；已清空卡上旧任务，不是上一个请求残留缓存导致。OOM日志见rank0/3/4/5/6，其他rank被终止，不能宣称全部rank独立OOM。

后续DiT／VAE容量没有得到验证，本次没有加入临时降分辨率或offload规避。需要独立优化位置插值临时张量。1:4图片也不在当前MaaS允许的0.4～2.5宽高比内，此项是离线压力测试，不改变在线输入约束。10秒间隔驱动采样可能漏掉瞬时峰值，不用于代替PyTorch峰值。

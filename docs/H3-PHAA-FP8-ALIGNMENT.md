# PHAA FP8 peer 包对齐

GPU 的 `sage-fp8-full-parallel` 预设默认包含 Align16，离线和在线共用这一配置。CUDA 同步 PHAA FP8 通信将每个 peer 的包长补齐到16字节，有效FP8数据和FP32 scale保持原布局，接收端按相同步长读取，补齐字节写零。

```bash
# Sage-FP8全并行Streaming默认启用Align16
H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p

# 保留该预设，独立关闭Align16
XGEN_PHAA_FP8_ALIGN16=0 H3_BASE_STEPS=20 bash example/diffusers/scripts/run_h3.sh cuda t2va 5 768p
```

T2VA、I2VA和R2VA都随该预设自动开启。准备模型与AdaLN表的方法见[CUDA全并行指南](H3-CUDA-COMPOSABLE.md)。在线服务采用同一预设时同样自动开启，关闭时先设置 `export XGEN_PHAA_FP8_ALIGN16=0` 再启动服务。修改后重启进程，所有rank必须使用相同代码和开关值。日志包含 `PHAA FP8 per-peer 16-byte alignment enabled`，`peer_alignment_metrics()` 提供调用次数、补齐调用次数和新增传输字节数。

该特性仅开放给GPU的同步CUDA FP8 transport，底层入口要求所有输入/输出张量位于CUDA设备。BF16通信、NPU和独立的异步FP8实现沿用各自路径。预设之外的transport默认关闭；兼容的同步CUDA FP8配方可以显式设置 `XGEN_PHAA_FP8_ALIGN16=1`。设置为0可恢复原wire布局，取消环境变量会恢复当前预设的默认值。

每次 collective 最多增加 15 字节/peer，发送和接收 buffer 均按新包长分配。收益取决于原始包长；已经对齐的 shape 不增加 padding，也不保证有加速。测试和性能结果须区分纯通信与完整模型；原始精度和长视频能力不能直接由此推断。

## 验证结果

八卡 RTX 5090、768P T2VA 5秒、20步、Sage-FP8全并行、native-stream、BF16 AdaLN表、50层常驻。同一服务和模型内关闭/开启开关，各完整预热一次，再正式测一次；未启用 profiler，初始化不计入正式请求。

| 指标 | 关闭 | 开启 | 耗时降低 |
|---|---:|---:|---:|
| 在线推理 | 41.590s | 32.187s | 22.61% |
| DiT | 33.330s | 25.145s | 24.56% |
| HTTP至成功 | 42.029s | 32.552s | 22.55% |
| 峰值 allocated/卡 | 25.914GiB | 25.914GiB | — |

发布前传输回归21项通过，包括19项CUDA小张量测试和2项非CUDA入口拒绝测试。另有6种真实八卡传输布局、合计48个rank场景，有效FP8字节、scale及解码输出均逐位一致，补齐字节为零。显式关闭时保留原wire布局，覆盖混合stride、batch=2、非连续输入、3/4头分块及单路回程。

两条正式视频和音轨完整解码通过。音频PCM SHA-256相同，视频PSNR为57.821dB、SSIM为0.999199，视频像素差异原因尚未单独归因。整条生成链路的bitwise等价尚未证明，人工精度验收待完成。

后续在线矩阵覆盖T2VA、I2VA、单图及五图R2VA各5/10/15秒，12项正式与12次预热技术通过，192份rank快照和12个下载视频SHA验证通过。当前总表保留推理/DiT耗时、峰值allocated/reserved及输入图尺寸，详见[性能文档](performance/README.md)。该矩阵使用跨节点历史基线，独立于本节同机A/B。

默认绑定与配套在线发布修复在发布前追加回归，配置/解码/跨设备适配127项通过，另1项CUDA专项在CPU环境跳过。在线发布与接口18项及13个子测试通过。三种任务的自动默认选择、环境变量设0回退、其他CUDA配方和NPU默认隔离均有对应测试。

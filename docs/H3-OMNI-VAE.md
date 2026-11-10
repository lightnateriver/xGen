# H3 Native-stream VAE Decode（实验性）

H3 解码统一通过 `--h3_vae_backend` 选择，CUDA 与 NPU 共用三种后端。CUDA 推荐的 `fp8-bf16`、`fp8-sage`（T2VA/I2VA/R2VA）及 `sage-fp8-full-parallel` 默认选择 `native-stream`；其他离线配置默认 `legacy`。显式选择优先。选择后端需要重启常驻服务；不能在请求之间热切换权重或后端。

`legacy` 表示沿用旧链路，并不自动启用 Lightning。常用配置如下：

| 解码方式 | CUDA | NPU | 配置 |
|---|---|---|---|
| 旧链路，不启用 Lightning | 支持 | 支持 | `--h3_vae_backend legacy` |
| 原 Lightning decoder | 支持 | 支持 | `--h3_vae_backend lightning`，需要 `--sp > 1` |
| 新原生流式 decoder | 支持 | 支持 | `--h3_vae_backend native-stream`，另加下述模型路径和授权参数 |

H3 不再用 `--vae_lightning` 开启 Lightning；旧命令中的 `--vae_lightning decoder` 应替换成 `--h3_vae_backend lightning`。其他模型（如 Wan）继续保留原参数。内部复用原 VAEManager 和资源调度，不另建 Lightning 实现。

在线服务可用同名 CLI 参数或配置 JSON 设置后端、模型路径及授权；显式 CLI 覆盖 JSON。NPU 在线预设仍用 `lightning`；CUDA 推荐示例已改为 `native-stream`。原 `--vae_matmul_quant none` 现在可显式覆盖 NPU 预设中的 INT8。修改后端需重启服务。

## 启动

在原有 **CUDA 或昇腾 H3 Base** 启动命令后追加以下参数：

```bash
--h3_vae_backend native-stream \
--h3_vae_model_path "$H3_MODEL_PATH/video_vae" \
--h3_vae_trust_remote_code \
--vae_matmul_quant none
```

`H3_MODEL_PATH` 指向原生 H3 模型目录。`video_vae` 必须包含原生 `config.json`、Python 实现及权重；这里不接受转换后的 Diffusers `vae` 子目录。第三个开关授权执行该模型目录中的 Python 代码，仅用于可信权重。

使用 `run_h3.sh` 时可设置 `H3_VAE_MODEL_PATH`、`H3_VAE_TRUST_REMOTE_CODE=1` 和可选 `H3_VAE_BACKEND`。默认切换不会自动授权模型代码；缺少路径或授权时，在加载权重前明确报错，不静默降级。

三种 decoder 互斥。保留现有 DiT、文本、PHAA 及独立 encoder 并行配置；`--h3_cuda_vae_sp` 属于旧 CUDA 解码器配置，不控制 native-stream 的 tile 分派。新后端使用当前分布式进程组的全部 rank，单卡也可运行。目前要求 `world_size == sp`，一组 rank 处理同一个请求；不支持把独立 DP 请求混入同一 VAE collective。

关闭新后端：移除原生模型路径与授权参数，并指定 `--h3_vae_backend legacy`。无需修改模型文件。

脚本回退示例：

```bash
unset H3_VAE_MODEL_PATH H3_VAE_TRUST_REMOTE_CODE
H3_VAE_BACKEND=legacy bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png
```

把 `legacy` 换为 `lightning` 即选择原 Lightning。NPU 三种后端也使用同一选择器；native-stream 须关闭旧 VAE INT8，旧两路可继续使用其原量化配置。三后端均支持 CUDA/NPU，不代表任意精度、常驻层数和输入长度组合都已完成整网验收。

## 迁移范围与边界

| 部分 | 行为 |
|---|---|
| 原生模型 | 加载 H3 原生 decoder、post-quant 层及 checkpoint 的 tile 调度，不套在 Lightning 的双层切块上 |
| 空间并行 | 整幅画面的 tile 按 rank 轮转分派；tile 数少于进程数时使用上游的本地退化路径 |
| 时间维 | 保留原生 overlap、blend、首尾特殊帧与尾部裁切；只发布已经提交的帧 |
| 输出 | 设备上归一化与 RGB8 round；独立 copy stream、两个可复用锁页槽位、256 MiB 在途预算；有界后台 MP4 编码 |
| 音频 | 沿用 xGen Audio VAE；先解码音频，再通过迁移的 mux 封装 AAC |
| 生命周期 | 沿用 Diffusers `ModuleGroup` CPU master 暂存，模型只加载一次；请求结束恢复 CPU master 引用 |
| 失败 | 编码启动、回调和输出发布失败传递给参与 rank；不自动重试或恢复 latent |
| 可观测 | 日志包含 H2D、解码与输出、offload 时间，以及 copy/slot 等待和在途峰值 |

onload、编码器清理或设备同步异常后仍尝试恢复 CPU master；清理错误不覆盖原始推理错误。设备故障时不承诺恢复成功，metrics 的 `success=false`，由现有服务处理失败。

`pil` 文件输出直接写 MP4，由现有 H3 保存标志通知上层无需再次编码；目前只支持单请求单视频。`latent` 直接返回 latent，不加载 decoder 到设备。`pt` / `np` 输出使用 Omni 原生完整输出路径（先量化为 RGB8，再还原到 0–1 做张量后处理），会物化整段输出，因此不具有流式文件路径的内存优势，也不保留未量化的浮点像素精度。

本版本支持 **CUDA 和昇腾原生 H3 VAE**，未开放 Refiner、替换式 NVFP4/Lite VAE 权重。它不引入 vLLM 运行时，复用已有 PyTorch、Transformers、PyAV、NumPy 和 Diffusers；可选 Triton 算子保留上游硬件判断。5090/SM120 不在上游 VAE 融合算子启用列表中，使用原生 FP32 权重与 FP16 autocast；其他 CUDA 硬件上的 SiLU reference fallback 不等同于 vLLM 的编译算子性能。

CUDA 推荐 FP8 配方额外复用原 decoder 的 Linear 存储策略：仅 `nn.Linear` 权重和 bias 预存为 FP16（与 FP16 autocast 的操作数类型一致），归一化、卷积和其他参数保留原生 FP32。它不把 VAE 改成 FP8；NPU 和未选 CUDA FP8 配方的原生 decoder 仍保留 FP32 权重。原始 FP32 整块 decoder 上卡与全常驻 DiT 组合曾在 32 GiB 显卡 OOM，不能直接套用单独 VAE 的容量结果。

两路 CUDA decoder 共用原有解码前准备回调：释放已完成 DiT 的请求内条件结果、回收空闲缓存，再暂存 decoder 权重；不会卸载 DiT。metrics 分列 `prepare_seconds`、`h2d_seconds`、`decode_output_seconds`、`offload_seconds` 和 `success`。

昇腾路径使用当前 `torch_npu` 算子与设备 stream/event，保留原生 tile 和时间块算法；不启用 CUDA Triton 融合。已验证环境为 Torch 2.10.0、torch_npu 2.10.0.post4、Diffusers 0.40.0：完成八卡 910B3 组件矩阵及单卡 CPU master 暂存复用测试，另完成 910B1 小尺寸探针；尚未验证包含 DiT 的新后端整网容量。原 `--vae_matmul_quant int8` 仅适用于旧 VAE，不能作用于新原生权重，选择 native-stream 时须设为 `none`；DiT 的量化选择独立于此开关。

原生 FP32 decoder 的显存比现有半精度暂存方案大。管线仍保留原 Diffusers VAE 供 encoder 使用，新 decoder 的 CPU master 是额外的主机内存占用。组件峰值、CPU 锁页内存和模型加载时间应单独报告；现有 DiT 常驻层数不自动调整。不能把组件通过当作任意常驻配置的整网容量保证。

两种 tile 几何不同，tile 内 self-attention 不等同于全画面全局 attention；overlap 不能恢复缺失的远距离 K/V。新后端与 Lightning 不承诺逐像素相同，需要对同一 latent 核对主体、接缝和时间连续性。

## 来源与验证

迁移来源固定为 [vLLM-Omni b92856c](https://github.com/vllm-project/vllm-omni/tree/b92856cb540958713fee5d99a1d6c8e68fd48755/vllm_omni/diffusion)。归属、Apache-2.0 许可证、逐文件源哈希和适配说明位于 `framework/vae/omni_h3/`。原生 checkpoint 代码仍从用户提供的模型目录加载。

隔离组件测试对齐八卡、latent、FP32 decoder、FP16 autocast、24 fps、x264 fast/CRF10 及同一 PCM 音频；每个形状预热一次、正式一次。计时范围为 latent H2D 至 MP4 完成，模型初始化及技术验收另计，不能直接作为整网加速比。精度使用同一清晰成片经 VAE 重新编码后的 latent，并比较编码前 RGB8 和抽帧；明确区别于原始 DiT 最终 latent。

测试入口：

```bash
pytest x_diffusers/tests/test_h3_omni_decode.py -q
```

设置 `XGEN_TEST_NATIVE_VAE` 为可信原生 `video_vae` 路径，可运行 CUDA 小尺寸解码、连续两次请求权重复用和失败后 CPU master 恢复测试。测试结果见 [CUDA 组件性能与精度](performance/h3-omni-vae.md)和 [910B3 三后端性能](performance/h3-vae-npu.md)。两份报告的音频与环境口径不同，不直接计算跨硬件加速比。

默认切换后的回归、两次真实 CUDA 请求及实际 OOM 处理见 [提交验证记录](performance/h3-vae-release.md)。小尺寸 FP16 Linear 存储对比的 RGB8 最大差为 1，不宣称逐像素等价或继承旧 decoder 的人工精度认可。

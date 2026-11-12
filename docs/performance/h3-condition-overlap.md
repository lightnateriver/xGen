# R2VA 条件准备重叠：验证范围

此功能默认关闭。统一 CUDA/NPU 开关和启动方式见[使用说明](../H3-CONDITION-OVERLAP.md)。下列历史测量来自基于公开基线 `ca6b6f0` 的私有原型，不能当作最新 v0.6 在所有设备上的性能承诺。

## CUDA 组件与容量结果

8×RTX 5090、SP8、文本 FP8 TP8、原 dense Sage、主 DiT FP8 Linear、PHAA2＋Align16、50 层常驻、native-stream、BF16 AdaLN 表且相应权重缺席。五张输入原图均为 2560×1440，参考短边 2048；输出 1344×768；固定输入、顺序及 seed42。

预热后按 A→B→B→A，每条路径两次正式测量，另做条件张量和 RNG 的逐位检查。计时从初始化之后开始，到第一步 DiT 输入齐备为止，**不包括 DiT、解码和输出保存**。

| 输出时长 | 原串行 s | 重叠 s | 节省 s | 条件耗时降低 | 串行峰值 allocated GiB/卡 | 重叠峰值 GiB/卡 |
|---|---:|---:|---:|---:|---:|---:|
| 5s | 14.043 | 12.792 | 1.251 | 8.91% | 25.711 | 26.805 |
| 10s | 14.549 | 13.876 | 0.673 | 4.62% | 25.334 | 26.428 |

上述原型显存采样保留 DiT 输入回放副本。两次候选样本跨度分别约 13.1%／10.1%，统计显著性未建立，不能把这些百分比当作稳态收益或整段视频推理收益。

另做同输入的完整 **1 实际 DiT step** 容量测试：每项仅一次，无预热，两条请求复用同组八卡模型。

| 场景 | 实际完成 DiT/每rank | 结果 | 全路径峰值 allocated GiB/卡 | 全路径峰值 reserved GiB/卡 |
|---|---:|---|---:|---:|
| 五图5s | 1 | 八rank生成成功，124帧及音轨完整解码 | 25.974 | 28.529 |
| 五图10s | 1 | 八rank生成成功，243帧及音轨完整解码 | 26.016 | 28.668 |

1 step 仅确认容量和技术链路，不能评价画面、动态、口型或音频质量，也不能证明20/50步或15秒场景稳定。

## 满载限制

7,000 字符、九张 320×1280 图片（短边2048后为2048×8192）、三段各5秒且带音轨的视频，目标15秒768P：原串行历史测试与开启重叠的单步探针都在 Qwen3-VL 视觉位置编码插值 OOM。该计算在 SP 切分前临时申请约10.12 GiB，两次实际 DiT 完成次数均为0。

两次输入文件哈希一致，但节点与代码快照不同，不把显存差全部归因于重叠；该满载输入不能通过开启此功能解决。

该满载为直接离线引擎的容量探针，1:4 素材不代表当前 MaaS 输入校验接受此比例。

## v0.6 接入验证

发布接入使用实例级原生 modular blocks，VAE 汇总复用原适配器；没有原型中的全局 Diffusers 类替换或函数代码复制。默认关闭、串行回退、状态隔离、CPU 传输字节、两种设备调度接口和错误收尾由 CPU 契约测试覆盖。

基于最新 `v0.6` 基线 `97a969e`，本次验证结果如下：

| 检查 | 结果与边界 |
|---|---|
| CPU 回归 | 251 passed、7 skipped；覆盖新开关及超时、配置 JSON/CLI 覆盖顺序、启动配置隔离、CUDA/NPU 调度契约和相关原有适配器 |
| 八卡 CUDA 接入 | 五图 5s／10s，两条请求复用同一组模型，每 rank 实际完成 1 步 DiT，16/16 记录成功 |
| 原串行与重叠条件对照 | 两个场景各八 rank 的文本 embedding、text token tags、参考视频／音频 latent 逐位一致，CPU/CUDA RNG 不变；图像用例没有独立音频参考输入 |
| 视频技术检查 | 1344×768、124／243 帧，均包含音轨，完整解码通过；不作为人工画面或听音验收 |
| 退出清理 | 本批模型 worker 已退出，八卡占卡恢复；不作为长期泄露结论 |

该 CUDA 接入检查额外执行原串行条件分支作对照，**耗时包含校验开销，不用于性能对比**。包含该校验的全路径峰值如下，不能与上文原型的纯容量测量直接作显存收益对比：

| 场景 | peak allocated GiB/卡（八rank最大值） | peak reserved GiB/卡（八rank最大值） |
|---|---:|---:|
| 五图5s | 25.974 | 29.604 |
| 五图10s | 26.016 | 29.391 |

CPU 回归命令（在已配置好项目依赖的仓库根目录执行）：

```bash
python -m pytest -q x_diffusers/tests/test_h3_condition_overlap.py x_diffusers/tests/test_h3_portable_features.py x_diffusers/tests/test_h3_cuda_options.py x_diffusers/tests/test_h3_cuda_precision_config.py x_diffusers/tests/test_h3_cuda_precision_legacy.py x_diffusers/tests/test_h3_launch_profiles.py x_diffusers/tests/test_h3_reference_image_size.py x_diffusers/tests/test_h3_condition_parallel.py x_diffusers/tests/test_h3_omni_decode.py server/video/test_condition_overlap.py
```

这不是全仓库测试通过声明。额外运行旧 HTTP 精度测试时，有两条既存断言在未修改的 `97a969e` 上同样失败：示例 JSON 默认精度期望值、4 秒输入拒绝期望值；本次不调整这些业务规则或旧断言。

NPU 使用相同调度和 CPU/Gloo 路径，保留原精度和 HCCL 策略。**NPU 真机性能、显存和数值结果尚未验收**；不将 CUDA 结果推广为 NPU 结果。

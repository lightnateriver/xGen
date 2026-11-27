# H3 NPU 部署与算子

[文档目录](README.md) · [资产准备](H3-MODELS.md#adaln-npu) · [视频命令](H3-VIDEO-QUICKSTART.md#npu)

<a id="defaults"></a>
## Base 默认方案

NPU 768P 推荐启动器和 Base/Ref2VA 服务 JSON 使用 VC attention、Ascend 融合、INT8 DiT、通用 full-parallel、BF16 AdaLN 表和 native-stream。推荐入口显式设置默认值，低层 CLI 和 2K/Image 入口有各自配置。

| 配置 | 值 |
|---|---|
| 进程、DiT SP、文本 TP | 8 / 8 / 8 |
| Base 步数与表 | 20 次实际计算，对应 21 个调度点 |
| Linear / attention | INT8 / VC ExpCast |
| AdaLN | BF16 表；完整权重的主块/输出 AdaLN 投影保留 BF16 |
| 调度 | full-parallel、DiT 常驻，请求内条件复用 |
| Token Refiner | SP8、BF16 FFN；Ref2VA chunk2048，T2VA/I2VA chunk0 |
| 视频 decoder | native-stream，`vae_matmul_quant=none`；Linear FP16 存储，其余原生参数 FP32 |
| 音频 VAE | FP32，按条件和形状选择编码/长音频解码并行 |

有图/视频时才执行视觉与 VAE 编码；有音轨时才执行音频编码。full-parallel 不自动把 DiT 改为 TP/CP，也不改变输入分辨率。

<a id="operators"></a>
## Ascend 融合算子安装

在目标 Ascend 910B（`dav-2201`）Linux aarch64 环境构建，使用匹配的 CANN、PyTorch/torch_npu、AscendC、CMake 和 C++17 工具链。源码位于 `x_base/x_base/operator/npu_fusion/`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install --no-deps -e ./x_base -e ./x_diffusers
PYTHON=python BUILD_JOBS=4 bash tools/build_h3_optimizations.sh
XGEN_NPU_META=1 python x_base/tests/test_npu_fusion_meta.py
```

按实际 CANN 安装位置调整路径。Meta 检查只确认算子注册、shape 和 dtype，设备计算仍需实际验证。

| 能力 | 默认库，相对于 `operator/npu_fusion/` | 覆盖变量 |
|---|---|---|
| `swiglu` | `f1b/build/libh3_swiglu_quant_reuse.so` | `H3_SWIGLU_QUANT_REUSE_LIBRARY` |
| `gate_residual` | `f2a/build/libh3_gate_residual.so` | `H3_GATE_RESIDUAL_LIBRARY` |
| `norm_quant` | `f2_nq2/build/libh3_norm_affine_quant2_cached.so` | `H3_NQ2_CACHED_LIBRARY` |
| `rope` | `f3/build/libh3_partial_rope.so` | `H3_PARTIAL_ROPE_LIBRARY` |
| `phaa_layout` | `f4/build/libh3_phaa_layout.so` | `H3_PHAA_LAYOUT_LIBRARY` |
| `laser_preprocess` | `f4_preprocess/build/libh3_phaa_preprocess.so` | `H3_PHAA_PREPROCESS_LIBRARY` |
| `merge_quant` | `f4_merge_quant/build/libh3_merge_quant.so` | `H3_MERGE_QUANT_LIBRARY` |

`shared_qkv` 和 `text_cache` 复用 PyTorch/torch_npu。包含编译库的 wheel 按对应平台和 ABI 构建，不能跨环境复用未经确认的二进制。

<a id="vc"></a>
## VC attention 安装

使用部署配方固定的源码版本，在同一目标环境构建：

```bash
git clone --branch codex/ascend-fp16-expcast-optimize https://gitee.com/czf1234/vc-attention.git
git -C vc-attention checkout 161108e910d86b4db954d5fb18b221ce876e2bed
cd vc-attention
VC_ATTENTION_BUILD_DEVICE=0 VC_ATTENTION_BUILD_JOBS=4 \
  python -m pip install --no-deps --no-build-isolation -e .
```

回到 xGen 根目录后启动。VC 使用 INT8 QK、FP16 ExpCast 和 FP16 PV，支持其声明的 dense D64/D128 FP16/BF16 布局。mask、causal、dropout、梯度输入或初始化失败会使用原生 SDPA；实际执行期错误不会自动重试。

<a id="switches"></a>
## 融合选择与回退

`--h3-opts all` 请求九项能力，`none` 清空列表，逗号列表和 `-名称` 可单项选择。显式列表优先于 `H3_OPT_FEATURES`，`--h3-opt-rope` / `--no-h3-opt-rope` 等单项参数再覆盖列表。服务在启动 JSON 中配置 `h3_opts`，不传离线 CLI 参数给服务外层。

推荐入口已有显式 `all`；覆盖时放在额外参数中：

```bash
bash example/diffusers/scripts/run_h3.sh npu r2va 5 768p inputs/ref01.png -- \
  --h3-opts none
```

保留原 attention 与融合时可选择 `--atten laser --h3-opts none`。原生 VAE 和组件并行分别控制，见 [统一配置](H3-CONFIGURATION.md)。

量化融合需要匹配的 INT8 投影；部分 PHAA 布局融合要求 SP4/PHAA7/56 头/D128，常用 SP8/PHAA2 使用其原布局。`laser_preprocess` 需要匹配的 MindIE Laser。`all` 是请求列表，具体 shape、后端、钩子和库决定实际执行。

检查 `[VC_ATTENTION] enabled=expcast`、`[H3_OPT] enabled=`、`summary=`、`fallback=` 确认命中及回退。NPU 2K 和 H3-Image 使用独立配方，不继承 Base 的所有默认值；可选 Spark/Veda 见 [attention 配置](H3-CONFIGURATION.md#attention)。

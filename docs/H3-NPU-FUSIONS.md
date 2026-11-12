# MiniMax-H3 NPU 融合算子与 VC-Attention

## 编译与安装

融合算子来自 `xgen_optimized_full_source_20261107_v5.zip`。在目标 Ascend 910B（`dav-2201`）的 Linux aarch64 环境中构建。压缩包使用 CANN 9.1、Python 3.12、PyTorch 2.10 及匹配的 `torch_npu`。构建还需要 CMake、C++17 工具链和 CANN 的 AscendC 开发组件。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install --no-deps -e ./x_base -e ./x_diffusers
PYTHON=python BUILD_JOBS=4 bash tools/build_h3_optimizations.sh
XGEN_NPU_META=1 python x_base/tests/test_npu_fusion_meta.py
```

在仓库根目录执行。按实际安装位置调整 `set_env.sh` 路径。Meta 检查会加载算子库并验证注册、shape 和 dtype。实际命中、数值精度和性能需在设备上验证。

| 开关 | 默认库，相对于 `x_base/x_base/operator/npu_fusion/` | 库路径覆盖变量 |
|---|---|---|
| `swiglu` | `f1b/build/libh3_swiglu_quant_reuse.so` | `H3_SWIGLU_QUANT_REUSE_LIBRARY` |
| `gate_residual` | `f2a/build/libh3_gate_residual.so` | `H3_GATE_RESIDUAL_LIBRARY` |
| `norm_quant` | `f2_nq2/build/libh3_norm_affine_quant2_cached.so` | `H3_NQ2_CACHED_LIBRARY` |
| `rope` | `f3/build/libh3_partial_rope.so` | `H3_PARTIAL_ROPE_LIBRARY` |
| `phaa_layout` | `f4/build/libh3_phaa_layout.so` | `H3_PHAA_LAYOUT_LIBRARY` |
| `laser_preprocess` | `f4_preprocess/build/libh3_phaa_preprocess.so` | `H3_PHAA_PREPROCESS_LIBRARY` |
| `merge_quant` | `f4_merge_quant/build/libh3_merge_quant.so` | `H3_MERGE_QUANT_LIBRARY` |

`shared_qkv` 和 `text_cache` 复用 PyTorch 与 `torch_npu`。构建完成后可运行 `python -m pip wheel --no-deps --no-build-isolation -w dist ./x_base ./x_diffusers` 打包。包含已编译库的 `x_base` wheel 带有平台标签。

## 启用融合

```bash
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p -- --h3-opts rope,swiglu,gate_residual,norm_quant
```

融合默认关闭。`--h3-opts all` 请求全部九项，`--h3-opts none` 清空列表。也可用 `H3_OPT_FEATURES=all,-text_cache` 设置默认列表。CLI 列表覆盖环境变量，`--h3-opt-rope`、`--no-h3-opt-rope` 等单项参数覆盖列表。在线服务在启动 JSON 中用 `"h3_opts": "rope,swiglu"` 配置。

`swiglu`、`norm_quant`、`shared_qkv` 和 `merge_quant` 需要匹配的 INT8 投影。PHAA 布局和合并量化需要 SP4 / PHAA7 / 56 头 / D128，常用的 SP8 / PHAA2 会使用原生布局。`laser_preprocess` 需要兼容的 MindIE Laser 版本及匹配的 PHAA 分块。启用了 v0.5 请求内条件预计算的配方直接复用现有缓存。

在 INFO 日志中查找 `[H3_OPT] enabled=` 和每次请求的 `summary=`。张量几何、钩子、后端或库不满足约束时，`fallback=` 会记录原因。

## VC ExpCast Attention

VC 使用第三方 `vc_attention` 包。压缩包固定的安装版本如下。在同一目标 NPU 环境中安装，安装钩子会构建 VC 原生组件。

```bash
git clone --branch codex/ascend-fp16-expcast-optimize https://gitee.com/czf1234/vc-attention.git
cd vc-attention
git checkout 161108e910d86b4db954d5fb18b221ce876e2bed
VC_ATTENTION_BUILD_DEVICE=0 VC_ATTENTION_BUILD_JOBS=4 python -m pip install --no-deps --no-build-isolation -e .
```

回到 xGen 根目录，在 768P Base 推理命令中追加 `--atten vc`。可同时选择融合开关。在线服务的对应配置是 `"atten": "vc"`。

```bash
bash example/diffusers/scripts/run_h3.sh npu t2va 5 768p -- --atten vc --h3-opts rope,swiglu
```

VC 使用 INT8 QK、FP16 ExpCast 和 FP16 PV。支持 D64/D128 的 FP16 或 BF16 BNSD dense attention。mask、causal、dropout、梯度输入或 VC 初始化失败时使用原生 SDPA。VC 自带预处理，`laser_preprocess` 会转交给 VC。检查 `[VC_ATTENTION] enabled=expcast` 确认实际命中，并在部署前核对输出画面和音频质量。

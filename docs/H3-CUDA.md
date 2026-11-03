# MiniMax-H3 CUDA 部署与启动

使用 `XGEN_DEVICE=cuda` 启动现有 CLI 或 HTTP 服务。省略该变量时选择 NPU。设备由启动环境决定，HTTP 请求沿用现有生成参数。

## 环境和模型

准备支持 RTX 5090 的 PyTorch CUDA、Triton、兼容 MiniMax-H3 的 Diffusers 和 Transformers。NVFP4 需要 Comfy-Kitchen CUDA 扩展，Sage 和 Sparge 分别需要对应 CUDA 扩展。Triton 首次编译需要与运行时 Python 版本匹配的开发头文件和可写的缓存目录。安装支持 H.264 和 AAC 的 FFmpeg。部署示例使用本地模型组件和权重。

```bash
export PYTHONPATH="./x_base:./x_diffusers${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=./weights/h3
export NVFP4_CHECKPOINT=./weights/h3-fl2va-nvfp4.safetensors
export FIRST_FRAME=./inputs/first-frame.png
export PROMPT_FILE=./inputs/prompt.txt
```

`MODEL_PATH` 指向含 `modular_model_index.json` 的 FL2VA 组件目录。NVFP4 checkpoint 需与该目录的 Transformer 配置匹配。CUDA 打包权重已验证的工作流为 FL2VA。

## 768P 离线

```bash
GPUS=8 ATTENTION_BACKEND=sparge XGEN_SPARGE_TOPK=0.5 XGEN_PHAA_FP8=1 \
  bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh \
  --h3_nvfp4_resident --h3_cuda_vae_sp --phaa_num 2
```

默认请求 124 帧，24 fps，51 个调度点对应 50 次 DiT。可选 `ATTENTION_BACKEND=sage` 使用 5090 上的 INT8 Q/K 和 FP8 P×V。FP8 PHAA 同时控制去程和回程。关闭该环境开关即可恢复常规通信。

C4 预设复用 xGen PHAA。下面的两档参数已检查，整套 TP8 配方仍待八卡端到端验证。

| 预设 | 运行方式 | 帧数 | 分块顺序 | NCCL_MAX_CTAS |
| --- | --- | ---: | --- | ---: |
| 5 秒 | `GPUS=8 XGEN_FASTVIDEO_RECIPE=5s bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh` | 124 | 0,1,2,3 | 2 |
| 15 秒 | `GPUS=8 XGEN_FASTVIDEO_RECIPE=15s bash example/diffusers/scripts/infer_minimax_h3_cuda_nvfp4.sh` | 345 | 3,0,1,2 | 1 |

预设同时选择文本 TP8、DiT 常驻、Sparge、FP8 PHAA 和 gather-priority。CUDA 文本 TP8 要求八个分布式 rank。没有选择预设时，启动脚本使用文本 TP1。

## 在线与可选 2K

[README 的 CUDA 在线启动示例](../README.md#实验性-cuda-2k-refiner)给出了 JSON 和 torchrun 命令。使用 `/generate` 和 `/status/<task_id>` 提交及查询任务。2K CUDA 需要 SP8、NVFP4 FL2VA Base 权重及预处理的 Refiner 资产。按 [H3 部署教程的模型准备章节](H3-DEPLOYMENT.md#43-一次性预处理)准备 Refiner，在 CUDA 环境执行 `h3_regenerate.prepare`，将结果输出到独立目录。普通 768P 请求可直接使用 Base 配置。

[实测性能和验证范围](performance/h3-cuda.md)

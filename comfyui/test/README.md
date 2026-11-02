# ComfyUI 测试

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_comfyui_nodes.py` | 节点配置、W8A8 量化、Attention 分发、SPWanLayerNorm、PHAA 开关、FSDP 签名、VAE 解码签名、Patch 文件完整性 |
| `test_comfyui_inference.py` | 分布式后端选择、环境变量、工作流 JSON、目录结构、源码关键特征、序列并行、README 内容校验 |

## 运行

```shell
cd comfyui/test
python -m pytest -v
```

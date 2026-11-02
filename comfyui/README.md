# ComfyUI 集成

X_GEN 提供与 ComfyUI 的集成，支持在 ComfyUI 中使用昇腾加速特性。

## 目录结构

```
comfyui/
├── custom_nodes/         # ComfyUI 自定义节点
├── main_multi.py         # 多卡启动入口
├── README.md             # 本文档
└── Wan2.2首尾帧720P加速版.json  # 示例工作流
```

## 支持功能

| 功能 | 说明 |
|------|------|
| W8A8 线性层量化 | 权重激活量化加速 |
| Sage Attention | 高效 Attention 实现 |
| 序列并行 (SP) | 分布式序列并行 |
| Laser Attention | 可选的 Attention 优化 |
| VAE 并行 | VAE 分布式处理 |
| FSDP | Fully Sharded Data Parallel |
| 通算并行 | 计算通信重叠 |

## 安装步骤

### 1. 安装依赖

```shell
pip install ascendx_video-0.1.0-py3-none-any.whl
pip install seal-6.5.913-cp311-cp311-linux_aarch64.whl
```

### 2. 设置环境

```shell
source /path/to/user-home/operate/AscendTurbo/set_env.bash
source /path/to/user-home/operate/vendors/customize/bin/set_env.bash
source /path/to/user-home/operate/vendors/customize_cloud/bin/set_env.bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/user-home/anaconda3/envs/PyTorch-2.5.1/lib/python3.11/site-packages/torch_npu/lib:/path/to/user-home/anaconda3/envs/PyTorch-2.5.1/lib/python3.11/site-packages/torch/lib/
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

### 3. 安装 ComfyUI-WanVideoWrapper

```shell
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
cd ComfyUI-WanVideoWrapper
git reset --hard 37365817e82c3d6057fcb0a5d7f952f7376f096a
```

将 `custom_nodes/ComfyUI-WanVideoWrapper/comfyui_ascendx` 放入对应目录，执行 `patch_apply.sh`。

### 4. 安装 VideoHelperSuite

```shell
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite
git checkout 6e7f63867584bb1fb4944d36172e7f98436da9a1
pip install -r requirements.txt
git apply video_save_adaptor.patch
```

### 5. 安装 MindIE（可选，支持 Laser Attention）

```shell
# 下载 MindIE
# https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann&ie=2.1.RC1&pt=7.1.0&cann=8.2.RC1

# 安装
./Ascend-mindie_2.0.RC2_linux-aarch64.run --install

# 设置环境
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
```

> **注意**: MindIE 需要 PyTorch 2.1.0 基础镜像

## 启动方式

### 单卡启动

```shell
python main.py --port 8080 --listen 0.0.0.0
```

### 多卡启动

```shell
torchrun --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29505 \
    main_multi.py --port 8080 --listen 0.0.0.0
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `WAN_FSDP_ENABLE=true` | 启用 FSDP |
| `PHAA_ENABLE=true` | 启用通算并行 |

## 示例工作流

目录下提供了示例工作流 JSON：

- `Wan2.2首尾帧720P加速版.json`

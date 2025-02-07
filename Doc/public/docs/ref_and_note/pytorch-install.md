---
title: PyTorch 安装
keywords: PyTorch
desc: PyTorch 安装
date: 2025-02-07
id: pytorch
---

Reference: [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)

## PyTorch 基本环境搭建

1. 创建并激活环境

```bat
python -m venv [venv-name]
[venv-name]\Scripts\activate
```

2. 安装 Pytorch

```bat
pip install torch torchvision torchaudio
```

验证 Pytorch 安装，出现版本号则为正常。

## PyTorch-GPU 环境搭建

在搭建虚拟环境后，如果需要在 GPU 上运行，需要安装 PyTorch-GPU 版本。

1. 确定自己的 GPU CUDA 版本。

```bat
nvidia-smi
```

2. 下载对应的 PyTorch-GPU 版本。[官方引导下载](https://pytorch.org/get-started/locally/)

附镜像页面链接：

- PyTorch官方镜像
    - [Torch](https://download.pytorch.org/whl/torch/)
    - [TorchVision](https://download.pytorch.org/whl/torchvision/)
    - [TorchAudio](https://download.pytorch.org/whl/torchaudio/)
- [阿里云镜像源](https://mirrors.aliyun.com/pytorch-wheels/)
    - 支持的 CUDA：10.0、10.1、10.2、11.0、11.1、11.3、11.5、11.6、11.7、11.8、12.1

3. 检测是否可用。

```python
import torch
print(torch.cuda.is_available())
```
---
title: 日志记录
keywords: 日志
desc: 日志
date: 2024-12-29
class: heading_no_counter
---

### TODO

1. 《Image-to-Image Translation with Conditional Adversarial Networks》提出 GANs 统一框架、基础 PatchGAN
2. 《High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs》提出 3 级金字塔判别器（PatchGAN）
3. CycleGAN
4. 学习 PyTorch 框架(2)
	- [ ] 迁移学习
    - [ ] 实验跟踪
    - [ ] 论文复制
    - [ ] 模型部署
5. ......

### 2025.03.05

- [x] 学习 VGG 网络架构，完成阅读论文《Very Deep Convolutional Networks for Large-Scale Image Recognition》
    - 首次系统性地研究了卷积神经网络（CNN）深度对大规模图像识别任务的影响，并提出了经典的 VGG 网络架构，其中 VGG-16、VGG-19 效果比较好。
    - 使用 3 × 3 小卷积核堆叠替代大卷积核，减少参数量并引入更多非线性因素
    - 通过浅层网络预训练初始化深层网络的前几层和全连接层，加速收敛并缓解梯度不稳定问题
    - [VGG 笔记](../ref_and_notes/vgg.md)

### 2025.03.02~2025.03.04

- [x] 学习残差网络，完成阅读论文《Deep Residual Learning for Image Recognition》
    - 提出了深度残差学习框架（ResNet），通过引入残差块（Residual Block）和快捷连接（Shortcut Connection），解决了深度神经网络训练中的退化问题（随着深度增加，训练误差不降反升）
    - ResNet-9、ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152
    - [ResNet 笔记](../ref_and_notes/resnet.md)

### 2025.02.26~2025.02.28

- [x] 学习 U-Net 网络，完成阅读论文《U-Net Convolutional Networks for Biomedical Image Segmentation》
    - 提出了一种名为 U-Net 的卷积神经网络架构，专为生物医学图像分割任务设计
    - 收缩路径（卷积）、扩展路径（上卷积）
    - 权重图与权重初始策略
    - [U-Net 笔记](../ref_and_notes/unet.md)

### 2025.02.07~2025.02.13

- [x] 学习 PyTorch 框架 (1)
    - [x] PyTorch 张量操作、模型搭建
    - [x] 基础分类模型
    - [x] 计算机视觉基础
    - [x] 自定义数据集
    - [x] 模块化

### 2025.02.01~2025.02.04

- [x] 完成阅读论文《Generative Adversarial Nets》，做了个[笔记和实验](/ref_and_note/GAN.html)。
    - 生成器、判别器
    - 在 $k$ 步优化判别器 D 和 $1$ 步优化生成器 G 之间交替进行
    - 最大化 $\log D(G(z))$ 代替最小化 $\log(1−D(G(z)))$ 训练生成器 G
    - [GAN 笔记](../ref_and_notes/gan.md)

### 2025.01.06~2025.01.13

- [x] 完成 & 提交开题报告。
- [x] 完成预期方案及成果。

### 2024.12.29

- [x] 选定网页框架，新建文件夹。


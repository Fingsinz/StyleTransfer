---
title: 日志记录
keywords: 日志
desc: 日志
date: 2024-12-29
class: heading_no_counter
---

### 2025.04.02~

- 对改进的 MetaNet 网络进行消融实验和对比实验（实验记录平台：[Swanlab](https://swanlab.cn/@Fingsinz/StyleTransfer/overview)），尝试在不同的部分进行改进
- 实验等待过程做撰写毕业设计论文
    - [x] 04.04 拟全文目录，起草论文第一章绪论

### 2025.03.31~2025.04.01

- [x] 了解特征变换（白化和着色）的基本机制，准备用于作比较
- [x] 了解快速基于 Patch 的风格迁移，准备用于作比较
- [x] 训练了增加注意力机制的 MetaNet，比较效果

### 2025.03.27~2025.03.30

- [x] 了解风格迁移结果评分机制，阅读《Evaluation of Painting Artistic Style Transfer Based on Generative Adversarial Network》
- [x] 编码项目，地址：[Github/Fingsinz/StyleTransfer-PyTorch](https://github.com/Fingsinz/StyleTransfer-PyTorch/)
- [x] 添加基本 MetaNet 结构并训练网络
- [x] 添加推演程序、模型结果打分程序

### 2025.03.19~2025.03.26

- [x] 03.19 阅读《Meta Networks for Neural Style Transfer》
- [x] 03.23 了解 MetaNet 思想
    - 通过网络生成另一网络的参数
- [x] 03.24 复现 MetaNet
- [x] 03.25 ~ 03.26 代码实验 demo，代码分析，弄清 MetaNet 结构
- [MetaNet](../ref_and_notes/metanet.md)

### 2025.03.16~2025.03.17

- [x] 阅读《Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks》
- [x] 了解 CycleGAN 的基本架构
    - 两个生成器 $G_A$ 和 $G_B$，满足 $F(G(x))\approx x$ 和 $G(F(y))\approx y$
    - 两个判别器 $D_X$ 和 $D_Y$，$D_X$ 用于区分图像 $\{x\}$ 和生成图像 $\{F(y)\}$，$D_Y$ 用于区分图像 $\{y\}$ 和生成图像 $\{G(x)\}$
- [x] 复现论文
- [CycleGAN：循环 GAN](../ref_and_notes/cyclegan.md)

### 2025.03.11~2025.03.15

- [x] 阅读《High-Resolution Image Synthesis and Semantic Manipulation with cGANs》
- [x] 了解 pix2pixHD 中的三级判别器和 Coarse-to-Fine Generator
    - 全局生成器 $G_1$ 和局部增强器 $G_2$
    - 三级判别器 $D_1$、$D_2$ 和 $D_3$，分别对应原始分辨率图像、下采样 1/2 的图像和下采样 1/4 的图像
- [x] 理解 pix2pixHD 仓库中的部分代码
- [PatchGAN 到 Multi-Scale PatchGAN](../ref_and_notes/patchgan.md)

### 2025.03.06~2025.03.10

- [x] 了解 cGAN：$D(x|c)$ 或 $D(G(z|c)|c)$
- [x] 了解 pix2pix 中的 70×70 PatchGAN
    - 将输入图像分割为多个局部小块（Patch），每个 Patch 独立判断真伪，最终取所有 Patch 的平均结果作为整体判别输出
    - 论文中感受野最后为 70，即 Patch 为 70×70
- [x] 理解 pytorch-CycleGAN-and-pix2pix 仓库中 pip2pix model 部分
- [cGAN：条件 GAN](../ref_and_notes/cgan.md)

### 2025.03.05

- [x] 学习 VGG 网络架构，完成阅读论文《Very Deep Convolutional Networks for Large-Scale Image Recognition》
    - 首次系统性地研究了卷积神经网络（CNN）深度对大规模图像识别任务的影响，并提出了经典的 VGG 网络架构，其中 VGG-16、VGG-19 效果比较好
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

- [x] 完成阅读论文《Generative Adversarial Nets》，做了个[笔记和实验](/ref_and_note/GAN.html)
    - 生成器、判别器
    - 在 $k$ 步优化判别器 D 和 $1$ 步优化生成器 G 之间交替进行
    - 最大化 $\log D(G(z))$ 代替最小化 $\log(1−D(G(z)))$ 训练生成器 G
    - [GAN 笔记](../ref_and_notes/gan.md)

### 2025.01.06~2025.01.13

- [x] 完成 & 提交开题报告
- [x] 完成预期方案及成果

### 2024.12.29

- [x] 选定网页框架，新建文件夹


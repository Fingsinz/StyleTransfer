---
title: Multi-style Generative Network for Real-time Transfer
keywords: msgnet
desc: Multi-style Generative Network for Real-time Transfer
date: 2025-04-09
id: msgnet
---

[Multi-style Generative Network for Real-time Transfer](https://arxiv.org/abs/1703.06953)

*ZHANG H, DANA K. Multi-style Generative Network for Real-time Transfer[M/OL]//Lecture Notes in Computer Science,Computer Vision – ECCV 2018 Workshops. 2019: 349-365. http://dx.doi.org/10.1007/978-3-030-11018-5_32. DOI:10.1007/978-3-030-11018-5_32.*

> Despite the rapid progress in style transfer, existing approaches using feed-forward generative network for multi-style or arbitrary-style transfer are usually compromised of image quality and model flexibility. We find it is fundamentally difficult to achieve comprehensive style modeling using 1-dimensional style embedding. Motivated by this, we introduce CoMatch Layer that learns to match the second order feature statistics with the target styles. With the CoMatch Layer, we build a Multi-style Generative Network (MSG-Net), which achieves real-time performance. We also employ an specific strategy of upsampled convolution which avoids checkerboard artifacts caused by fractionally-strided convolution. Our method has achieved superior image quality comparing to state-of-the-art approaches. The proposed MSG-Net as a general approach for real-time style transfer is compatible with most existing techniques including content-style interpolation, color-preserving, spatial control and brush stroke size control. MSG-Net is the first to achieve real-time brush-size control in a purely feed-forward manner for style transfer. Our implementations and pre-trained models for Torch, PyTorch and MXNet frameworks will be publicly available.

**摘要**：尽管风格迁移的研究进展迅速，但现有的使用前馈生成网络进行多风格或任意风格迁移的方法通常会损害图像质量和模型灵活性。我们发现使用一维样式嵌入实现全面的样式建模从根本上是困难的。受此启发，我们引入了 CoMatch Layer，该层学习将二阶特征统计与目标风格进行匹配。利用 CoMatch 层构建了多风格生成网络（MSG-Net），实现了实时性能。我们还采用了一种特殊的上采样卷积策略，避免了由分数阶卷积引起的棋盘伪影。与最先进的方法相比，我们的方法取得了卓越的图像质量。所提出的 MSG-Net 作为实时风格转换的通用方法，与大多数现有技术兼容，包括内容风格插值、色彩保持、空间控制和笔触大小控制。MSG-Net 是第一个以纯粹的前馈方式实现实时笔刷大小控制的风格转移。我们对 Torch、PyTorch 和 MXNet 框架的实现和预训练模型将公开可用。

## 主要内容

### CoMatch 层  

通过匹配目标风格的二阶特征统计量（ Gram 矩阵），取代传统的一维风格嵌入（均值和方差），更全面地捕捉风格特征。  
- 解决了因一维嵌入导致的风格表达能力不足的问题，显著提升了生成图像的质量。

### MSG-Net架构

- 上采样卷积：采用整数步长卷积结合上采样操作，替代传统的反卷积（分数步长卷积），避免了棋盘格伪影。  
- 上采样残差块：扩展残差网络结构，支持恒等映射，加速收敛并提升深层网络的稳定性。  
- 实时笔触控制：通过动态调整风格图像的分辨率（训练时随机采样尺寸，推理时用户自定义），实现笔触大小的实时调整。

## 关键技术细节

### 训练策略 

1. 使用预训练的 VGG 网络作为损失网络，通过联合优化内容损失（特征匹配）和风格损失（Gram 矩阵匹配）。  

2. 风格图像在训练时动态调整尺寸（256、512、768），使模型适应不同笔触大小。  

### 网络结构 

- 编码器-解码器框架：Siamese 网络提取风格特征，生成网络通过 CoMatch 层融合内容与风格信息。  
- 反射填充：减少边界伪影。  
- 实例归一化：提升生成图像对比度鲁棒性。

## 实现

PyTorch-Github：https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer

---
title: 快速基于补丁的任意风格的风格转移
keywords: Fast Patch-based
desc: Fast Patch-based Style Transfer of Arbitrary Style
date: 2025-04-01
id: fast_patch_based
---

[Fast Patch-based Style Transfer of Arbitrary Style](https://arxiv.org/abs/1612.04337)

*CHEN T, SCHMIDT M. Fast Patch-based Style Transfer of Arbitrary Style[J]. Cornell University - arXiv,Cornell University - arXiv, 2016.*

> Artistic style transfer is an image synthesis problem where the content of an image is reproduced with the style of another. Recent works show that a visually appealing style transfer can be achieved by using the hidden activations of a pretrained convolutional neural network. However, existing methods either apply (i) an optimization procedure that works for any style image but is very expensive, or (ii) an efficient feedforward network that only allows a limited number of trained styles. In this work we propose a simpler optimization objective based on local matching that combines the content structure and style textures in a single layer of the pretrained network. We show that our objective has desirable properties such as a simpler optimization landscape, intuitive parameter tuning, and consistent frame-by-frame performance on video. Furthermore, we use 80,000 natural images and 80,000 paintings to train an inverse network that approximates the result of the optimization. This results in a procedure for artistic style transfer that is efficient but also allows arbitrary content and style images.

**摘要**：艺术风格转移是一个图像合成问题，其中一个图像的内容与另一个图像的风格复制。最近的研究表明，视觉上吸引人的风格转移可以通过使用预训练卷积神经网络的隐藏激活来实现。然而，现有的方法要么应用一个适用于任何风格图像但非常昂贵的优化过程，要么一个只允许有限数量的训练风格的有效前馈网络。在这项工作中，我们提出了一个基于局部匹配的更简单的优化目标，将内容结构和风格纹理结合在预训练网络的单层中。我们展示了我们的目标具有理想的属性，例如更简单的优化场景、直观的参数调优以及视频上一致的逐帧性能。此外，我们使用 80,000 张自然图像和 80,000 幅绘画来训练一个近似优化结果的逆网络。结果是得到一种高效的艺术风格转移过程，允许任意内容和风格图像。

## 风格交换 Style Swap

设 $C$ 表示内容图像，$S$ 表示风格图像。$\Phi(\cdot)$ 表示预训练 CNN 模型的全卷积部分表示的函数，将图像从 RGB 映射到某个中间激活空间。计算激活值 $\Phi(C)$ 和 $\Phi(S)$ 后，**风格交换**如下：

1. 从内容和风格的激活中提取一组 Patches，表示为 $\{\phi_i (C)\}_{i\in n_c}$ 和 $\{\phi_j (S)\}_{j\in n_s}$，其中 $n_c$ 和 $n_s$ 为提取的 Patch 个数。提取的 Patch 应该有足够的重叠，并且包含所有的激活通道。

2. 对于每个内容激活的 Patch，根据归一化互相关度量确定最接近匹配的风格 Patch：

$$
\phi_i^{ss}(C, S) := \arg \mathop{\max}\limits_{\phi_j(S),j=1,...,n_s} \frac {<\phi_i (C), \phi_j (S)>} {\vert\vert \phi_i (C) \vert\vert \cdot \vert\vert \phi_j (S) \vert\vert}
$$

3. 将每个内容激活 Patch $\phi_i (C)$ 与其最匹配的风格 Patch $\phi_i^{ss}(C, S)$ 进行交换。

4. 通过对步骤 3 中可能具有不同值的重叠区域进行平均，重建完整的内容激活 $\Phi^{ss} (C, S)$。

## 优化目标

目标是**最小化合成图像激活与目标激活 $\Phi^{ss} (C, S)$ 的平方误差**，并加入总变差正则化（TV Loss）以平滑图像：

$$
I_{stylized}(C, S) = \arg \mathop{\min}\limits_{I\in \mathbb{R}^{h\times w\times d}} \vert\vert \Phi(I) - \Phi^{ss}(C, S) \vert\vert _F^2 + \lambda\mathcal{l}_{TV}(I)
$$

优化过程通过反向传播完成，但由于耗时，作者进一步提出逆网络。

## 逆网络

训练目标：学习从风格交换后的激活图 $\Phi^{ss} (C, S)$ 直接生成图像，绕过逐次优化。

关键设计：
- 使用混合数据集（8 万自然图像 + 8 万绘画）训练，增强泛化能力。
- 引入风格交换后的激活图作为训练数据，解决 CNN 的非满射性问题。
- 网络架构基于转置卷积层和实例归一化（InstanceNorm），提升生成质量。

## 实现

Torch-Github：https://github.com/rtqichen/style-swap

PyTorch-Github：https://github.com/irasin/Pytorch_Style_Swap

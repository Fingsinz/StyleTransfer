---
title: 自适应实例归一化 AdaIn
keywords: AdaIn
desc: 自适应实例归一化 AdaIn
date: 2025-04-02
id: adain
---

[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

*HUANG X, BELONGIE S. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization[C/OL]//2017 IEEE International Conference on Computer Vision (ICCV), Venice. 2017. http://dx.doi.org/10.1109/iccv.2017.167. DOI:10.1109/iccv.2017.167.*

> Gatys et al. recently introduced a neural algorithm that renders a content image in the style of another image, achieving so-called style transfer. However, their framework requires a slow iterative optimization process, which limits its practical application. Fast approximations with feed-forward neural networks have been proposed to speed up neural style transfer. Unfortunately, the speed improvement comes at a cost: the network is usually tied to a fixed set of styles and cannot adapt to arbitrary new styles. In this paper, we present a simple yet effective approach that for the first time enables arbitrary style transfer in real-time. At the heart of our method is a novel adaptive instance normalization (AdaIN) layer that aligns the mean and variance of the content features with those of the style features. Our method achieves speed comparable to the fastest existing approach, without the restriction to a pre-defined set of styles. In addition, our approach allows flexible user controls such as content-style trade-off, style interpolation, color & spatial controls, all using a single feed-forward neural network.

**摘要**：Gatys 等人最近介绍了一种神经算法，该算法将内容图像呈现为另一图像的风格，实现了所谓的风格迁移。然而，他们的框架需要一个缓慢的迭代优化过程，这限制了其实际应用。目前已经提出了前馈神经网络的快速近似来加速神经风格的转换。不幸的是，速度的提高是有代价的：网络通常与一组固定的风格相关联，无法适应任意的新风格。在本文中，我们提出了一种简单而有效的方法，首次实现了任意风格的实时转换。我们方法的核心是一个新颖的自适应实例归一化（AdaIN）层，它将内容特征的均值和方差与样式特征的均值和方差对齐。我们的方法与最快的现有方法速度相当，且不受预定义样式集的限制。此外，我们的方法允许灵活的用户控制，如内容风格权衡，风格插值，颜色和空间控制，所有这些都使用单个前馈神经网络。


## 自适应实例归一化（AdaIN）

提出了一种新颖的归一化层，通过对**齐内容特征和风格特征的均值和方差（统计量）**，直接在特征空间实现风格迁移。AdaIN 的公式为：

$$
\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)
$$

其中，$x$ 为内容特征，$y$ 为风格特征。AdaIN 无需可学习参数，仅通过风格特征的统计量调整内容特征，实现高效风格对齐。

结合预训练的 VGG 编码器和轻量级解码器，构建了一个端到端的前馈网络。该网络支持对任意未见过的风格进行实时处理（如 512 × 512 图像达 15 FPS），无需针对新风格重新训练。

## 方法框架

### 编码器-解码器架构  

编码器：固定使用 VGG-19 的前几层（至 relu4_1），提取内容和风格图像的高层特征。  

AdaIN层：将内容特征的均值和方差对齐到风格特征，生成目标特征。  

解码器：随机初始化，通过反卷积将AdaIN输出的特征逆映射到图像空间。解码器未使用归一化层以避免风格固化。

### 损失函数：  

使用预训练 VGG 计算内容损失和风格损失：  
- 内容损失：目标特征（AdaIN 输出）与生成图像特征的 L2 距离。  
- 风格损失：生成图像与风格图像在各 VGG 层上的均值和方差差异的 L2 距离。

## 关键创新点

从特征统计量视角解释 IN 的作用：
- 作者通过实验验证，实例归一化（IN）的有效性源于其对图像风格的归一化，而非仅对比度调整。IN通过消除内容图像的原始风格信息，使网络更易学习目标风格。

对比现有方法：
- 优化方法（如 Gatys）：灵活但速度慢（分钟级）。
- 单风格前馈网络（如 Ulyanov）：速度快（毫秒级）但风格受限。
- 风格交换（如 Chen）：支持任意风格但计算量大（95% 时间用于风格交换）。
- 本文方法：结合前馈速度（接近单风格方法）与任意风格灵活性，且无计算瓶颈。

## 实现

Torch-Github：https://github.com/xunhuang1995/AdaIN-style

PyTorch-Github：https://github.com/naoto0804/pytorch-AdaIN

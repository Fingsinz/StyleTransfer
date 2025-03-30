---
title: 风格迁移评价
keywords: 迁移评价
desc: 风格迁移评价
date: 2025-03-27
id: evaluation
---

*仅讨论评估方法*

## ResNet 18 预训练模型和结构相似性

[Evaluation of Painting Artistic Style Transfer Based on Generative Adversarial Network](https://ieeexplore.ieee.org/document/10154714)

*TANG Z, WU C, XIAO Y, et al. Evaluation of Painting Artistic Style Transfer Based on Generative Adversarial Network[J].*

> Unlike most image style transfer tasks, which are to convert a photograph into an image with a certain artist’s style, this paper focuses on artistic style transfer in Monet painting, which is transferring a Monet painting into an image in another artist’s style. In our experiments, we successfully implemented the Cycle-Consistent GAN model and applied Neural Style Transfer (NST) model for contrasting effects. In order to evaluate the result of artistic style transfer quantitatively and effectively with a low requirement for computational resources, we also proposed a quantitative method called style transfer indicator to make the comparison more obvious as the comparisons of the effect of image style transfer were mostly done by subjective analysis previously. This method takes both the style and content of the transferred image into account because whether the transferred image belongs to the new style is as important as whether the content of the image is saved. A ResNet18 pre-trained model and structural similarity index are used for the evaluation of style and content respectively. The human survey that we conducted also proved the validity of our style transfer indicator. Moreover, our proposed indicator could also be applied for the evaluation of other image style transfer tasks.

**摘要**：大多数图像风格转换任务都是将一张照片转换成具有某个艺术家风格的图像，而本文主要研究的是莫奈绘画中的艺术风格转换，即将莫奈的一幅画转换成另一个艺术家风格的图像。在我们的实验中，我们成功地实现了循环一致的 GAN 模型，并应用了神经风格迁移（NST）模型来对比效果。为了在对计算资源要求较低的情况下，对艺术风格迁移的效果进行定量有效的评价，我们还提出了一种称为风格迁移指标的定量方法，使图像风格迁移效果的比较更加明显。该方法同时考虑了转换图像的样式和内容，因为转换的图像是否应用新风格与图像的内容是否保持一致同样重要。ResNet 18 预训练模型和结构相似性指数分别用于风格和内容的评估。我们在人们中调查也证明了我们的风格转移指标的有效性。此外，我们提出的指标也可以应用于其他图像风格迁移任务的评估。

*在 IV. Evaluation 部分*

### 风格评分

在风格的定量分析方面，采用了风格分类器的方法。
- 经过训练的分类器能够区分图像的风格，可以判断风格迁移模型是否将图像迁移到正确的风格以及风格转移的程度。

其中，ResNet 是一种深度卷积神经网络模型，经过大型图像数据集彻底训练，可以学习图像分类和识别所需的许多特征。

- 使用预训练的 ResNet18 模型（需要微调）对生成图像进行风格分类。
- 通过 SoftMax 输出概率值作为风格得分（0-1），衡量生成图像与目标风格的匹配度。

### 内容评分

在内容的定量分析方面，采用结构相似性指数 SSIM 指标。

将 SSIM 的三个组成部分，**亮度**、**对比度** 和 **结构** 相乘生成 SSIM：

$$
S(x,y) = f(l(x, y), c(x, y), s(x, y)) = l(x, y) \cdot c(x, y) \cdot s(x, y)
$$

其中，$l$, $c$, $s$ 代表评估亮度、对比度和结构；$x$ 和 $y$ 分别是 2 张图像。

#### 亮度测量

亮度通过图像的均值 $\mu$ 衡量：

$$
l(x,y) = \frac{2 \mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}
$$

其中：

- $\mu_x$ 和 $\mu_y$ 是图像 $x$ 和 $y$ 的均值。
- $C_1$ 是一个常数，用于防止分母为零的情况，通常 $C_1=(k_1L)^2$，$L$ 为像素动态范围，$k_1\ll 1$。

#### 对比度测量

对比度通过标准差 $\sigma$ 衡量：

$$
c(x,y) = \frac{2 \sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_1}
$$

其中：

- $\sigma_x$ 和 $\sigma_y$ 是图像 $x$ 和 $y$ 的标准差。
- $C_2$ 是一个常数，用于防止分母为零的情况，通常 $C_2=(k_2L)^2$，$L$ 为像素动态范围，$k_2\ll 1$。

#### 结构测量

结构通过协方差 $\sigma_{xy}$ 衡量：

$$
s(x,y) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}
$$

其中：

- $\sigma_{xy}$ 是图像 $x$ 和 $y$ 的协方差。
- $C_3$ 是一个常数，通常取 $C_2/2$。



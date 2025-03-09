---
title: cGAN 简介
keywords: cGAN
desc: cGAN 简介
date: 2025-03-06
id: cGAN
---

[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

*Mirza M , Osindero S .Conditional Generative Adversarial Nets[J].Computer Science, 2014:2672-2680.DOI:10.48550/arXiv.1411.1784.*

[Image-to-Image Translation with Conditional Adversarial Networks](https://ieeexplore.ieee.org/document/8100115)

*Isola P , Zhu J Y , Zhou T ,et al.Image-to-Image Translation with Conditional Adversarial Networks[C]//IEEE Conference on Computer Vision & Pattern Recognition.IEEE, 2016.DOI:10.1109/CVPR.2017.632.*

## cGAN

条件生成对抗网络（Conditional Generative Adversarial Networks, cGAN）是生成对抗网络（GAN）的一种扩展形式，通过引入 **条件信息**（如标签、文本、图像等），使生成器和判别器能够根据特定条件生成或判别数据。

- 核心思想是通过条件约束，控制生成内容的属性和结构，从而解决普通 GAN 生成结果不可控的问题。

### cGAN 的核心原理

条件信息的引入： 

- 生成器（Generator）：输入不仅包含随机噪声 $z$，还包括条件信息 $c$（如类别标签、另一张图像等）。生成器需根据 $c$ 生成对应的数据 $G(z|c)$。  
- 判别器（Discriminator）：输入包含真实数据 $x$ 或生成数据 $G(z|c)$，同时结合条件信息 $c$。判别器的任务是判断数据是否真实且与条件匹配，即 $D(x|c)$ 或 $D(G(z|c)|c)$。

cGAN 的损失函数在普通 GAN 的基础上加入了条件约束：  

$$
\mathcal{L}_{cGAN}(G,D) = \mathbb{E}_{x,c}[\log D(x|c)] + \mathbb{E}_{z,c}[\log(1 - D(G(z|c)|c)]]
$$  

- 生成器 $G$ 的目标：生成与条件 $c$ 匹配的逼真数据，使 $D(G(z|c)|c)$ 趋近于1。  
- 判别器 $D$ 的目标：区分真实数据 $x|c$ 和生成数据 $G(z|c)|c$。

### cGAN 对比普通 GAN

| 特性 | 普通GAN | 条件GAN（cGAN） |
|:-:|:-:|:-:|
| 输入 | 随机噪声 $z$ | 随机噪声 $z$ + 条件信息 $c$ |
| 生成控制 | 完全随机 | 通过条件 $c$ 控制生成内容 |
| 应用场景 | 无约束生成（如随机图像生成）| 需特定条件生成（如根据文本生成图像） |
| 典型任务 | 生成随机人脸、艺术品 | 图像到图像转换（pix2pix）、文本到图像生成、可控生成（如风格迁移）、图像修复、图像翻译（如黑白→彩色） |

### 代码示例

```python
# 生成器（U-Net结构为例）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入：噪声z + 条件图像c
        self.encoder = Encoder()  # 下采样层
        self.decoder = Decoder()  # 上采样层（含跳跃连接）

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)  # 拼接噪声和条件
        return self.decoder(self.encoder(x))

# 判别器（PatchGAN结构为例）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入：真实/生成图像 + 条件图像c
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3 + 3, 64, kernel_size=4, stride=2),  # 假设条件c是3通道图像
            nn.LeakyReLU(0.2),
            # 更多卷积层...
        )

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)  # 拼接图像和条件
        return self.conv_blocks(x)
```

## Image-to-Image Translation with cGAN

本文提出使用条件 GANs 作为通用解决方案，通过对抗训练自动学习任务相关的损失函数，避免人工设计损失函数的复杂性。

> *条件生成对抗网络（Conditional Generative Adversarial Networks, cGANs）最初由 Mehdi Mirza 和 Simon Osindero 在 2014 年的论文 《Conditional Generative Adversarial Nets》 中提出。这篇论文首次将条件信息（如类别标签或辅助数据）引入 GAN 框架，使生成器和判别器能够基于特定条件进行训练和生成。*

条件 GANs 的优势：
- 条件输入：生成器和判别器均以输入图像为条件，确保输出与输入的结构对齐（*如下图输入边缘图生成对应照片案例中，生成器和判别器都观察输入的边缘*）。

![](../static/images/cGAN/fig1.png)

- 结合L1损失：在对抗损失基础上引入 L1 损失，保留低频信息（如整体布局），而对抗损失负责高频细节（如纹理和锐度），解决传统 L2 损失导致的模糊问题。

### 方法细节

#### 目标函数

cGAN 的目标可以表示为：

$$
\mathcal{L}_{cGAN}(G,D) = \mathbb{E}_{x,c}[\log D(x|c)] + \mathbb{E}_{z,c}[\log(1 - D(G(z|c)|c)]]
$$

目标函数上，总损失函数为对抗损失与 L1 损失的加权和：

$$
G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)
$$

- 对抗损失：迫使生成器输出逼真的图像，判别器区分生成图像与真实图像。  
- L1 损失：约束生成图像与真实图像在像素级的一致性，减少模糊（所以不使用 L2 损失）。

随机性的引入：生成器的输入包含随机噪声（通过 Dropout 实现），但实验表明生成结果仍具有较低随机性。这表明当前方法在建模条件分布的多样性方面仍有改进空间。

#### 网络架构

在网络架构上：

- 生成器：带跳跃连接。
- 判别器（马尔可夫随机场）：PatchGAN，尝试对图像中的每个 N × N 块进行真假分类。在图像上卷积运行这个鉴别器，平均所有响应来提供 D 的最终输出。

#### 训练与推演过程

训练中：
- 遵循 GAN 中的优化算法，交替 $D$ 和 $G$ 的 step 训练。
- 在优化 $D$ 时将目标函数除以 2，减慢 $D$ 相对于 $G$ 学习的速率。
- 使用小批量 SGD 并应用 Adam 求解器，学习率为 0.0002，动量参数 $\beta_1 = 0.5$，$\beta_2 = 0.999$。

推演时：
- 与训练阶段相同的方式运行生成器。

### 实验与验证

1. 多任务测试：在语义标签 → 照片、地图→航拍图、图像着色等任务中，cGANs均能生成高质量结果。例如：  
   - 地图→航拍图：AMT 实验显示，18.9% 的生成图像被误认为真实图像。  
   - 图像着色：cGANs 生成的色彩分布更接近真实数据（通过 Lab 空间直方图验证）。

2. 架构分析  
   - U-Net vs. 编码器-解码器：U-Net 因跳跃连接显著提升生成质量（如 Cityscapes 任务中，FCN 分数提高 30%）。  
   - PatchGAN尺寸：70 × 70 的局部判别器在清晰度和计算效率间取得平衡，优于全图判别器（ImageGAN）和极小的 PixelGAN。

3. 感知评估  
   - AMT 众包实验：量化生成图像的逼真程度。  
   - FCN 分数：使用预训练分割模型评估生成图像的语义可识别性，验证其结构与真实标签的一致性。

### 局限性与启示

- 随机性不足：生成结果偏向确定性，难以建模多模态输出（如同一输入对应多种合理输出）。  
- 复杂任务表现：在高度结构化任务（如语义分割）中，cGANs 效果不及纯 L1 回归，表明对抗训练更适用于需细节生成的图形任务。  
- 社区应用：开源代码（pix2pix）被广泛用于艺术创作（如草图转肖像、背景去除），验证了其易用性和扩展性。

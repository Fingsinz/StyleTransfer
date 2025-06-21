---
title: 摘要
keywords: Abstract
desc: 论文摘要部分
date: 
id: abstract
class: heading_no_counter
---

<style>
p {
    font-size: 20px;
}

.text {
    text-indent: 2em;
}
</style>

<div class="text">

随着信息技术的发展，图像风格迁移技术在艺术创作、影视特效、文创设计等领域展现出重要价值。近年来出现了很多基于深度学习的风格迁移方法，它们相比传统方法提高了图像风格迁移的质量，但仍存在风格化不明显以及细节丢失或模糊的问题。 

本文基于元学习理念和现有的MetaNet模型展开研究，旨在通过改进算法，提升图像风格迁移的效果与效率。改进方法包括优化特征提取、改良图像转换网络结构，以及引入注意力机制。在特征提取中，采用预训练的VGG-19替代VGG-16，增强对深层风格特征的捕捉能力；在图像转换网络中，改进下采样和上采样的操作确保输入输出图像尺寸一致，并引入实例归一化层以保留图像细节特征。在元学习器中，分别引入通道注意力、自注意力和Transformer 模块，验证对通道间依赖、长距离特征关联的建模能力的提升效果。 

实验过程中采用MS COCO 2017测试集与WikiArt数据集的子集展开训练，并通过客观指标和主观的人工打分方式综合评估模型。实验纵向对比分析了多个超参数对迁移效果产生的影响，从而确定最优模型。横向方面，与AdaIN、MSG-Net、StyleID等主流风格迁移算法展开对比。本文改进模型在风格相似度评估上优于其他对比模型，得分最高达0.733，并且在推演效率上具有优势，尤其在实时交互场景中表现突出，充分证实了本文方法的有效性。 

</div>

**关键词**：深度学习；风格迁移；元学习；注意力机制

# Abstract

<div class="text">

With the development of information technology, image style transfer technology shows important value in the fields of art creation, film and television special effects, and cultural and creative design. In recent years, many deep learning-based style transfer methods have appeared, which improve the quality of image style transfer compared with the traditional methods, but still have the problems of inconspicuous stylization and loss or blurring of details. 

In this paper, based on the concept of meta-learning and the existing MetaNet model, we aim to improve the effectiveness and efficiency of image style transfer by improving the algorithm. The improvement methods include optimizing feature extraction, improving the image transformation network structure, and introducing the attention mechanism. In feature extraction, pre-trained VGG-19 is used instead of VGG-16 to enhance the ability to capture deep style features; in the image conversion network, the operations of down-sampling and up-sampling are improved  to ensure that the input and output image sizes are the same, and an instance normalization layer is introduced to preserve image detail features. In the meta-learner, channel attention, self-attention and Transformer modules are introduced respectively to verify the effect of improving the modeling ability for inter-channel dependency and long-range feature association.

A subset of MS COCO 2017 test set and WikiArt dataset is used to start the training during the experiment, and the model is evaluated comprehensively by objective metrics and subjective manual scoring. The experiment vertically compares and analyzes the impact of multiple hyperparameters on the transfer effect to determine the optimal model. Horizontally, the model is compared with mainstream style transfer algorithms such as AdaIN, MSG-Net, and StyleID. The improved model in this paper outperforms other comparative models in style similarity evaluation, with a score of up to 0.733, and has an advantage in deduction efficiency, especially in real-time interaction scenarios, which fully confirms the effectiveness of this paper's method. 

</div>

**Keywords**：Deep Learning; Style Transfer; Meta Learning; Attention Mechanism
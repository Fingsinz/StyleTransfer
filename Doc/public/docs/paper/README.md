---
title: 基于深度学习的图像风格迁移算法研究
keywords: 
desc: 基于深度学习的图像风格迁移算法研究
date: 
---

## 元网络风格迁移改进

### 迁移后图像保持原图像大小一致

1. 迁移时摒弃原来的强制缩放。
2. 调整 TransformNet 的下采样逻辑：移除 `stride=2`，改用卷积核实现特征提取。
3. 调整 TransformNet 的上采样逻辑：使用 bilinear 插值，替代固定 `scale_factor`，同时在 `forward` 使用 `F.interpolate`，动态恢复原始尺寸。

### 在 MetaNet 的全连接层之间添加注意力机制

1. 自实现注意力机制类。
2. 在 MetaNet 的 `forward` 中添加。
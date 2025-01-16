---
title: 生成对抗网络
keywords: GAN
desc: GAN文献及笔记
date: 2025-01-16
---

> [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
> Goodfellow I , Pouget-Abadie J , Mirza M ,et al.Generative Adversarial Nets[J].MIT Press, 2014.DOI:10.3156/JSOFT.29.5_177_2.

## 论文笔记

$$
\min_{G} \max_{D} \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z)))]
$$

## 代码测试
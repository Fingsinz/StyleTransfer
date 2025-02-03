'''
Created on 2025.02.01

@Author: Fingsinz (fingsinz@foxmail.com)
@Reference: 
    1. https://arxiv.org/abs/1406.2661
'''

import time
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# 配置参数
class Config():
    data_folder: str = './data' # 数据集路径, 此处用 MNIST 做测试
    batch_size: int = 128       # batch 大小
    epochs: int = 10            # 训练轮数
    lr: float = 0.0002          # 学习率
    betas: tuple = (0.5, 0.999) # Adam 的超参数
    k_steps: int = 5            # k 值
    latent_dim: int = 100       # 隐变量维度
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

# 判别器  
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
 
# GAN 模型
class GAN():
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config.latent_dim).to(config.device)
        self.discriminator = Discriminator().to(config.device)
        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config.lr, betas=config.betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=config.betas)
        self.real_label = 1
        self.fake_label = 0
        
    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=self.config.data_folder, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        return train_loader
    
    def train(self):
        train_loader = self.get_data()
        epochs = self.config.epochs
        g_loss = 0
        d_real_loss = 0
        d_fake_loss = 0
        
        for epoch in range(epochs):
            for i, (images, _) in enumerate(train_loader):
                batch_size = images.size(0)
                images = images.to(self.config.device)
                    
                # ---判别器训练-----------------------------
                if (i + 1) % self.config.k_steps != 0:
                    self.d_optimizer.zero_grad()
                    # 训练真实数据
                    labels = torch.full((batch_size,), self.real_label, device=self.config.device).float()
                    output = self.discriminator(images)
                    loss_real = self.criterion(output.view(-1), labels)
                    loss_real.backward()
                    # 训练假数据
                    z = torch.randn(batch_size, self.config.latent_dim, device=self.config.device)
                    fake_images = self.generator(z)
                    labels.fill_(self.fake_label).float()
                    output = self.discriminator(fake_images.detach())
                    loss_fake = self.criterion(output.view(-1), labels)
                    loss_fake.backward()
                            
                    self.d_optimizer.step()
                    d_real_loss = loss_real.item()
                    d_fake_loss = loss_fake.item()
                # ---判别器训练-----------------------------
                    
                # ---生成器训练-----------------------------
                else:
                    self.g_optimizer.zero_grad()
                    labels.fill_(self.real_label).float()
                    output = self.discriminator(fake_images)
                    loss_g = self.criterion(output.view(-1), labels)
                    loss_g.backward()
                    self.g_optimizer.step()
                    g_loss = loss_g.item()
                # ---生成器训练-----------------------------
                
                if i % 100 == 0:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] " +
                        f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], "
                        f"D Loss: {d_real_loss:.4f} + {d_fake_loss:.4f}, G Loss: {g_loss:.4f}")
                    
            self.save_generated_images(epoch + 1)
    
    def save_generated_images(self, epoch):
        """
        保存训练效果图片
        
        参数:
        - epoch (int): 当前轮数
        """
        z = torch.randn(64, self.config.latent_dim, device=self.config.device)
        fake_images = self.generator(z)
        fake_images = fake_images.cpu().detach().numpy()
        fake_images = np.transpose(fake_images, (0, 2, 3, 1))
        
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        
        for i in range(8):
            for j in range(8):
                axes[i, j].imshow(fake_images[i * 8 + j, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
        
        if not os.path.exists('gan_generated_images'):
            os.makedirs('gan_generated_images')
        
        plt.savefig(f'./gan_generated_images/epoch_{epoch}.png')
        plt.close()

if __name__ == '__main__':
    config = Config()
    gan = GAN(config)
    gan.train()
    
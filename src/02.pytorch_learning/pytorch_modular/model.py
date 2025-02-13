"""
@file: model.py
@brief: TinyVGG 定义
@author: -
@date: 2025-02-13
"""

import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    """
    一个简单的 VGG 类神经网络图像分类任务。
    
    参数:
        input_channels (int): 输入通道数 (e.g., 3 for RGB images)。
        hidden_units (int): 卷积层中隐藏单元的数量。
        output_shape (int): 输出类的数量。
    """
    def __init__(self, input_channels: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状：(batch_size, input_channels, height, width)。
        
        返回:
            torch.Tensor: 输出张量，形状：(batch_size, output_shape)。
        """
        return self.classifier(self.block_2(self.block_1(x)))

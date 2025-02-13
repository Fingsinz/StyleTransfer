"""
@file: data_setup.py
@brief: 包含从图像文件夹创建 PyTorch dataloader 的函数。
@author: -
@date: 2025-02-13
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 3

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    从图像文件夹创建PyTorch dataloader。

    参数:
        train_dir (str): 训练图像文件夹的路径。
        test_dir (str): 测试图像文件夹的路径。
        transform (transforms.Compose): 应用的图像变换。
        batch_size (int): DataLoader 的批量大小。
        num_workers (int): 用于加载数据的工作线程数。

    返回:
        包含训练 DataLoader、测试 DataLoader 和类名列表的元组。
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader, class_names

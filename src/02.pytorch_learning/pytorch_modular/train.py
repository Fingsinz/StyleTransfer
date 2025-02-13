"""
@file: train.py
@brief: 训练模型
@author: -
@date: 2025-02-13
"""

import os
import torch
import data_setup, engine, model, utils

from torchvision import transforms

# 训练超参数
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# 数据路径
train_dir = "../data/pizza_steak_sushi/train"
test_dir = "../data/pizza_steak_sushi/test"

# 训练设备
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 创建数据加载器
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=train_transform,
        batch_size=BATCH_SIZE,
    )

    # 创建模型
    model = model.TinyVGG(
        input_channels=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names),
    ).to(device)

    # 设置损失函数和优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=LEARNING_RATE)

    # 启动训练
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device)

    # 保存模型
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="tinyvgg_model.pth")
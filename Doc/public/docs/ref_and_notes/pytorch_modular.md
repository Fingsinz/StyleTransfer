---
title: PyTorch 模块化
keywords: PyTorch
desc: PyTorch 模块化
date: 2025-02-13
id: pytorch_modular
---

Reference: [PyTorch Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/)


## 前言

模块化后的 Python 文件：

- `data_setup.py`：用于准备和下载数据的文件。
- `engine.py`：包含各种训练函数的文件。
- `model_builder.py` 或 `model.py`：创建 PyTorch 模型的文件。
- `train.py`：利用所有其他文件并训练目标 PyTorch 模型的文件
- `utils.py`：专用于有用的实用程序功能的文件。

然后就可以使用类似于下面的命令调用脚本进行模型训练：

```bat
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

模块化后的目录：

```
root/
├── data_setup.py
├── engine.py
├── model.py
├── train.py
└── utils.py
└── models/
│   ├── xxx.pth
└── data/
    ├── train/
    │   └── xxx.jpg
    └── test/
        └── xxx.jpg
```

## data_setup.py：建立 Datasets 和 DataLoaders

获得数据后，将其转换为 PyTorch `Dataset` 和 `DataLoader` 。

- 将 `Dataset` 和 `DataLoader` 创建代码转换为一个名为 `create_dataloaders()` 的函数。

<details>
<summary>data_setup.py</summary>

```python
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

```

</details>

## model.py: 构建模型

将模型放入其文件中使得可以一次又一次地重用它。

<details>
<summary>data_setup.py</summary>

```python
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

```

</details>

## engine.py：训练和测试模型

在前面编写了几个训练函数：

- `train_step()`：接受一个模型、一个 DataLoader、一个损失函数和一个优化器，并在 DataLoader 上训练一个 step。
- `test_step`：接受一个模型、一个 DataLoader 和一个损失函数，并在 DataLoader 上测试模型。
- `train()`：执行 `train_step()` 和 `test_step()`。并返回一个结果字典。

<details>
<summary>engine.py</summary>

*需要下载 tqdm 包。*

```python
"""
@file: engine.py
@brief: PyTorch模型的训练和测试函数
@author: -
@date: 2025-02-13
"""

import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    模型逐步训练

    参数:
        model: 要训练的模型
        dataloader: 用于训练的 DataLoader
        loss_fn: 损失函数
        optimizer: 优化器
        device: 设备 (e.g. GPU or CPU)

    返回值:
        包含训练损失和正确率的元组
    """
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    模型逐步测试

    参数:
        model: 要测试的模型
        dataloader: 用于测试数据的 DataLoader
        loss_fn: 损失函数
        device: 设备 (e.g. GPU or CPU)

    返回值:
        包含训练损失和正确率的元组
    """
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """
    训练模型多个 epoch

    参数:
        model: 训练的模型
        train_dataloader: 训练数据的 DataLoader
        test_dataloader: 测试数据的 DataLoader
        optimizer: 优化器
        loss_fn: 损失函数
        epochs: 训练轮数
        device: 设备 (e.g. GPU or CPU)

    返回值:
        包含训练和测试损失和正确率的字典
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
            f"\nEpoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

```

</details>

## utils.py：实用函数集合

通常情况下，在训练期间或训练后需要保存模型。

- 将 helper 函数存储在名为 `utils.py` （utilities的缩写）的文件中。

<details>
<summary>utils.py</summary>

```python
"""
@file: utils.py
@brief: 包含实用函数
@author: -
@date: 2025-02-13
"""

import torch
from pathlib import Path
from typing import Union


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    保存模型

    参数:
        model (torch.nn.Module): 要保存的模型。
        target_dir (str): 保存模型的目标目录。
        model_name (str): 保存的模型文件的名称。

    异常:
        断言错误: 如果 model_name 不以.pt或. pth结尾。
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

```

</details>

## train.py：训练、评估和保存模型

在其它项目里，经常会遇到将其所有功能组合在一个 `train.py` 文件中。

在这里的 `train.py` 文件中，将结合我们创建的其他 Python脚本的所有功能，并使用它来训练模型。

有以下步骤：

1. 从目录中导入 `data_setup`、`engine`、`model`、`utils` 等各种依赖项。
2. 设置各种超参数，如批量大小，epoch数，学习率和隐藏单元数（这些可以在将来通过 Python 的 argparse 设置）。
3. 设置训练和测试目录。
4. 设置使用设备。
5. 创建必要的数据转换。
6. 使用 `data_setup.py` 创建 DataLoader。
7. 使用 `model.py` 创建模型。
8. 设置损失函数和优化器。
9. 使用 `engine.py` 训练模型。
10. 使用 `utils.py` 保存模型。

<details>
<summary>train.py</summary>

```python
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

```

</details>

*可研究 `argparse` 的使用优化。*

测试调用：

```bat
python .\train.py
  0%|                                                                                 | 0/5 [00:00<?, ?it/s] 
Epoch: 1 | train_loss: 1.1113 | train_acc: 0.3047 | test_loss: 1.0953 | test_acc: 0.3400
 20%|██████████████▌                                                          | 1/5 [00:18<01:14, 18.60s/it]
Epoch: 2 | train_loss: 1.1072 | train_acc: 0.2969 | test_loss: 1.0744 | test_acc: 0.4233
 40%|█████████████████████████████▏                                           | 2/5 [00:36<00:55, 18.45s/it]
Epoch: 3 | train_loss: 1.0931 | train_acc: 0.4141 | test_loss: 1.0908 | test_acc: 0.3617
 60%|███████████████████████████████████████████▊                             | 3/5 [00:55<00:36, 18.45s/it] 
Epoch: 4 | train_loss: 1.0891 | train_acc: 0.4180 | test_loss: 1.0931 | test_acc: 0.3722
 80%|██████████████████████████████████████████████████████████▍              | 4/5 [01:13<00:18, 18.46s/it] 
Epoch: 5 | train_loss: 1.0622 | train_acc: 0.4766 | test_loss: 1.0636 | test_acc: 0.4621
100%|█████████████████████████████████████████████████████████████████████████| 5/5 [01:32<00:00, 18.43s/it] 
[INFO] Saving model to: models\tinyvgg_model.pth
```
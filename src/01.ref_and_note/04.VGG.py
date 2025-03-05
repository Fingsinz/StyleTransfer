'''
Created on 2025.03.05

@Author: Fingsinz (fingsinz@foxmail.com)
'''

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg19

import swanlab
from torchsummary import summary

def load_data(data_path, batch_size=128, input_size=32):
    # 数据增强与归一化（默认CIFAR-10参数）
    train_transform = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载数据集（使用CIFAR-10）
    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=test_transform
    )

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader

class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
           self._make_conv_block(in_channels, 64, 2),   # conv3-64 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(64, 128, 2),           # conv3-128 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(128, 256, 3),          # conv3-256 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
        
           self._make_conv_block(256, 512, 3),          # conv3-512 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(512, 512, 3),          # conv3-512 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))     # 全局平均池化
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),               # 全连接层（FC 1）
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),                      # 全连接层（FC 2）
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),               # 全连接层（FC 3）
        )                                               # nn.CrossEntropyLoss() 计算损失时隐式 soft-max
            
    def _make_conv_block(self, in_channels, out_channels, num_blocks, kernel_size=3, padding=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))    # 每个卷积层后加入nn.BatchNorm2d，显著提升训练稳定性
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
           self._make_conv_block(in_channels, 64, 2),   # conv3-64 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(64, 128, 2),           # conv3-128 × 2
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(128, 256, 4),          # conv3-256 × 4
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(256, 512, 4),          # conv3-512 × 4
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
           
           self._make_conv_block(512, 512, 4),          # conv3-512 × 4
           nn.MaxPool2d(kernel_size=2, stride=2),       # maxpool
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))     # 全局平均池化
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),               # FC 1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),                      # FC 2
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),               # FC 3
        )
            
    def _make_conv_block(self, in_channels, out_channels, num_blocks, kernel_size=3, padding=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))     # 每个卷积层后加入nn.BatchNorm2d，显著提升训练稳定性
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(epochs + 1):
        model.train()
        loss_num = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            loss_num += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')
        scheduler.step()    
        
        loss_num /= len(train_loader)
        swanlab.log({"vgg_train_loss": loss_num})
        
        if epoch % 5 == 0:
            test(model, device, test_loader, criterion)
            
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.inference_mode():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += criterion(output, label).item()
            # 计算top-1和top-5准确率
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred))
            correct_top1 += correct[:1].reshape(-1).float().sum().item()
            correct_top5 += correct[:5].reshape(-1).float().sum().item()
    test_loss /= len(test_loader)
    top1_acc = 100.0 * correct_top1 / len(test_loader.dataset)
    top5_acc = 100.0 * correct_top5 / len(test_loader.dataset)
    print(f'Test set: vgg_test_loss: {test_loss:.4f}, Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')
    swanlab.log({"vgg_test_loss": test_loss, "vgg_top1_acc": top1_acc, "vgg_top5_acc": top5_acc})

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "classes": 10,
    "epochs": 50,
    "data_path": './data',
    "batch_size": 128
}

if __name__ == '__main__':
    train_loader, test_loader = load_data(config["data_path"], config["batch_size"])
    
    model = VGG16(num_classes=config["classes"]).to(config["device"])
    #model = VGG19(num_classes=config["classes"]).to(config["device"])
    #model = vgg19(num_classes=config["classes"]).to(config["device"])

    summary(model, (3, 32, 32))

    run = swanlab.init(
        project="StyleTransfer",
        experiment_name="VGG19_my",
        description="VGG19 on CIFAR-10",
        config=config
    )
    
    train(model, train_loader,test_loader, config["epochs"], config["device"])

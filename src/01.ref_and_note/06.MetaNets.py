'''
Created on 2025.03.23

@Author: Fingsinz (fingsinz@foxmail.com)
@Reference: 
    1. https://github.com/CortexFoundation/StyleTransferTrilogy/tree/master
'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pathlib
import random
from PIL import Image

import swanlab

def mean_std(features):
    """输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为 1920"""
    mean_std_features = []
    for x in features:
        batch, C, H, W = x.shape
        x_flat = x.view(batch, C, -1)
        mean = x_flat.mean(dim=-1)
        std = torch.sqrt(x_flat.var(dim=-1) + 1e-5)
        feature = torch.cat([mean, std], dim=1)
        mean_std_features.append(feature)
    return torch.cat(mean_std_features, dim=-1)

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros(out_channels, in_channels, kernel_size, kernel_size).to(config["device"])
        self.bias = torch.zeros(out_channels).to(config["device"])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, upsample=None,
              instance_norm=True, relu=True, trainable=False):
    """
    构造一个卷积层。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int, optional): 卷积核大小，默认为 3
        stride (int, optional): 卷积的步幅，默认为 1
        upsample (float or None, optional): 上采样的比例因子，None 表示不上采样。
        instance_norm (bool, optional): 是否在卷积后应用实例归一化，默认为 True。
        relu (bool, optional): 是否在规范化后应用 ReLU 激活，默认为 True。
        trainable (bool, optional): 参数是否可训练，默认为 False。

    Returns:
        list: 卷积层的列表。
    """
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))

    layers.append(nn.ReflectionPad2d(kernel_size // 2))  # 填充以保持空间维度

    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    else:
        layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride))

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        layers.append(nn.ReLU(inplace=True))

    return layers

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }
        
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )
        
    def forward(self, x):
        return self.conv(x) + x

class TransformNet(nn.Module):
    def __init__(self, base=8):
        super(TransformNet, self).__init__()
        self.base = base
        self.weights = []
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9, trainable=True),
            *ConvLayer(base, base * 2, kernel_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2)
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base*4) for _ in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True)
        )
        self.get_param_dict()

    def forward(self, x):
        y = self.downsampling(x)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y
    
    def get_param_dict(self):
        """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
        param_dict = defaultdict(int)
        def dfs(module, name):
            for _name, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, _name) if name != '' else _name)
            if isinstance(module, MyConv2D):
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))
        dfs(self, '')
        return param_dict
    
    def set_my_attr(self, name, value):
        """遍历字符串（如 residuals.0.conv.1）找到对应的权值"""
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)

    def set_weights(self, weights, i=0):
        """输入权值字典，对应网络所有的 MyConv2D 层进行设置"""
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])
    
class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128 * self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))
            
    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128 : (i + 1) * 128])
            
        return filters

class ImgDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*.jpg"))
        self.transform= transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.load_image(index)
        if self.transform:
            return self.transform(img)
        else:
            return img

def train(content_data_loader, model_vgg, model_transform, metanet):
    trainable_params = {}
    trainable_param_shapes = {}
    for model in [model_vgg, model_transform, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape
                
    optimizer = torch.optim.Adam(trainable_params.values(), 1e-3)
    metanet.train()
    model_transform.train()
    
    for epoch in range(config['epochs']):
        content_loss_sum = 0
        style_loss_sum = 0
        avg_max_value = 0
        batch = 0
        for content in tqdm(content_data_loader, desc=f"{epoch + 1} / {config['epochs']}"):
            if batch % 20 == 0:
                style_image = style_dataset[random.randint(0, len(style_dataset) - 1)].unsqueeze(0).to(config['device'])
                style_features = model_vgg(style_image)
                style_mean_std = mean_std(style_features)
                        
            x = content.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue
                    
            optimizer.zero_grad()
            weights = metanet(style_mean_std)
            model_transform.set_weights(weights, 0)
            content = content.to(config['device'])
            output = model_transform(content)
                    
            content_features = model_vgg(content)
            transformed_features = model_vgg(output)
            transformed_mean_std = mean_std(transformed_features)
                    
            content_loss = config['content_weight'] * F.mse_loss(transformed_features[2], content_features[2])
            style_loss = config['style_weight'] * F.mse_loss(transformed_mean_std,
                                                                 style_mean_std.expand_as(transformed_mean_std))
            y = output
            tv_loss = config['tv_weight'] * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                    
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
                    
            max_value = max([x.max().item() for x in weights.values()])
                    
            content_loss_sum += content_loss.item()
            style_loss_sum += style_loss.item()
            avg_max_value += max_value
            
            batch += 1
        
        swanlab.log({"content_loss": content_loss_sum / len(content_data_loader),
                     "style_loss": style_loss_sum / len(content_data_loader),
                     "max_value": avg_max_value / len(content_data_loader)})
        print(f"{epoch + 1} / {config['epochs']} | \
            content_loss: {content_loss_sum / len(content_data_loader)} | \
                style_loss: {style_loss_sum / len(content_data_loader)} | \
                    max_value: {avg_max_value / len(content_data_loader)} ")
        
        if (epoch + 1) % 3 == 0:
            test(content_dataset, style_dataset, model_vgg, model_transform, metanet)

            torch.save(metanet.state_dict(), './MetaNet_{}.pth'.format(epoch+1))
            torch.save(model_transform.state_dict(), './transform_net_{}.pth'.format(epoch+1))

def test(content_dataset, style_dataset, model_vgg, model_transform, metanet):
    # style_image = Image.open('./style.jpg').convert('RGB')
    # t = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # style_tensor = t(style_image).unsqueeze(0).to(config['device'])
    
    style_tensor = style_dataset[random.randint(0, len(style_dataset) - 1)].unsqueeze(0).to(config['device'])
    
    features = model_vgg(style_tensor)
    mean_std_features = mean_std(features)
    weights = metanet.forward(mean_std_features)
    model_transform.set_weights(weights)
    
    content_images = torch.stack([random.choice(content_dataset) 
                                  for _ in range(config["test_batch"])]).to(config['device'])
    transformed_images = model_transform(content_images)
    
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return torch.clamp((tensor * std + mean), 0, 1)

    style_denorm = denormalize(style_tensor).squeeze(0)
    style_vis = style_denorm.cpu().permute(1, 2, 0).numpy()
    style_images = np.repeat(style_vis[np.newaxis], 4, axis=0)

    content_vis = denormalize(content_images).cpu().permute(0, 2, 3, 1).numpy()
    transformed_vis = denormalize(transformed_images).cpu().detach().permute(0, 2, 3, 1).numpy()

    def create_grid(styles, contents, transformed):
        grid = []
        for s, c, t in zip(styles, contents, transformed):
            row = np.concatenate([s, c, t], axis=1)
            grid.append(row)
        full_grid = np.concatenate(grid, axis=0)
        return (full_grid * 255).astype(np.uint8)

    merged_image = create_grid(style_images, content_vis, transformed_vis)
    swanlab.log({'transformed_grid': swanlab.Image(merged_image)})

def one_image_test(content_path, model_vgg, model_transform, metanet, style_path='./style.jpg'):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),          # 强制缩放至256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    style_image = Image.open(style_path).convert('RGB')
    style_tensor = preprocess(style_image).unsqueeze(0).to(config['device'])

    content_image = Image.open(content_path).convert('RGB')
    content_tensor = preprocess(content_image).unsqueeze(0).to(config['device'])

    with torch.inference_mode():
        style_features = model_vgg(style_tensor)
        style_mean_std = mean_std(style_features)
        weights = metanet(style_mean_std)
        model_transform.set_weights(weights, 0)

    model_transform.eval()
    with torch.inference_mode():
        transformed_tensor = model_transform(content_tensor)
    model_transform.train()

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return torch.clamp((tensor * std + mean), 0, 1)

    content_vis = denormalize(content_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    transformed_vis = denormalize(transformed_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    style_vis = denormalize(style_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy() 

    comparison = np.concatenate([content_vis, style_vis, transformed_vis], axis=1)
    plt.imshow(comparison)
    plt.axis('off')
    plt.savefig(f"./{os.path.basename(content_path).split('.')[0]}+{os.path.basename(style_path).split('.')[0]}.png")
    plt.close()

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "base": 32,
    "style_weight": 50,
    "content_weight": 1,
    "tv_weight": 1e-6,
    "epochs": 30,
    "batch_size": 16,
    "width": 256,
    "test_batch": 4,
    
    "train_content_path": './coco2017/',
    "train_style_path": './WikiArt2/',
}

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(config['width'], scale=(256/480, 1), ratio=(1, 1)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

style_dataset = ImgDataset(config['train_style_path'], transform=data_transform)
content_dataset = ImgDataset(config['train_content_path'], transform=data_transform)

content_data_loader = torch.utils.data.DataLoader(content_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=True, num_workers=os.cpu_count())

if __name__ == '__main__':
    swanlab.init(
        project="StyleTransfer",
        experiment_name="MetaNet_demo_30",
        description="MetaNet demo",
        config=config
    )
    
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16 = VGG(vgg16.features).to(config['device']).eval()
    
    transform_net = TransformNet(config['base']).to(config['device'])
    transform_net.get_param_dict()
    
    metanet = MetaNet(transform_net.get_param_dict()).to(config['device'])
    
    train(content_data_loader, vgg16, transform_net, metanet)
    
    load_transform_net = TransformNet(config['base']).to(config['device'])
    load_transform_net.load_state_dict(torch.load('./transform_net_2.pth', map_location=config['device']))
    load_metanet = MetaNet(load_transform_net.get_param_dict()).to(config['device'])
    load_metanet.load_state_dict(torch.load('./MetaNet_2.pth', map_location=config['device']))
    one_image_test("content.JPEG", vgg16, load_transform_net, load_metanet)
    
'''
Created on 2025.02.27

@Author: Fingsinz (fingsinz@foxmail.com)

U-Net架构：
- 遵循原始论文设计：4个下采样块，每个块包含两次卷积
- 使用转置卷积进行上采样
- 通过中心裁剪实现跳跃连接的特征对齐

特征提取：
- 使用 VGG19 的 conv1_2 和 conv2_2 计算风格损失（捕捉纹理）
- 使用 conv3_4 和 conv4_4 计算内容损失（保留结构）

训练策略：
- 内容损失侧重高层语义特征
- 风格损失聚焦浅层纹理特征
- 使用 Adam 优化器，初始学习率 0.001

图像处理：
- 输入图像统一缩放到 512 * 512
- 使用 ImageNet 均值和标准差进行归一化
- 输出结果反归一化后保存

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path

import swanlab

# 图像预处理
def load_image(path, size=512):
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert('RGB')
    return loader(image).unsqueeze(0).to(config["device"])

# Gram矩阵计算
def gram_matrix(feature):
    b, c, h, w = feature.size()
    features = feature.view(b*c, h*w)
    return torch.mm(features, features.t()) / (c * h * w)

# 原始U-Net架构（与论文结构一致）
class OriginalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 收缩路径
        self.down1 = self._double_conv(in_channels, 64)
        self.down2 = self._double_conv(64, 128)
        self.down3 = self._double_conv(128, 256)
        self.down4 = self._double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 底部层
        self.bottleneck = self._double_conv(512, 1024)
        
        # 扩展路径
        self.up4 = self._up_conv(1024, 512)
        self.up_conv4 = self._double_conv(1024, 512)
        
        self.up3 = self._up_conv(512, 256)
        self.up_conv3 = self._double_conv(512, 256)
        
        self.up2 = self._up_conv(256, 128)
        self.up_conv2 = self._double_conv(256, 128)
        
        self.up1 = self._up_conv(128, 64)
        self.up_conv1 = self._double_conv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
    
    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _up_conv(self, in_c, out_c):
        return nn.ConvTranspose2d(
            in_c, out_c, 
            kernel_size=2, 
            stride=2,
            padding=0,    
            output_padding=0
        )     
    
    def forward(self, x):
        # 编码器
        c1 = self.down1(x)        # [B,64,H,W]
        p1 = self.pool(c1)
        
        c2 = self.down2(p1)       # [B,128,H/2,W/2]
        p2 = self.pool(c2)
        
        c3 = self.down3(p2)       # [B,256,H/4,W/4]
        p3 = self.pool(c3)
        
        c4 = self.down4(p3)       # [B,512,H/8,W/8]
        p4 = self.pool(c4)
        
        # 瓶颈层
        bn = self.bottleneck(p4)  # [B,1024,H/16,W/16]
        
        # 解码器（带中心裁剪跳跃连接）
        def _crop_concat(up, contract):
            _, _, h_up, w_up = up.size()
            h_cont, w_cont = contract.size()[2], contract.size()[3]
            delta_h = (h_cont - h_up) // 2
            delta_w = (w_cont - w_up) // 2
            cropped = contract[:, :, delta_h:delta_h+h_up, delta_w:delta_w+w_up]
            return torch.cat([up, cropped], dim=1)
        
        u4 = self.up4(bn)         # [B,512,H/8,W/8]
        u4 = _crop_concat(u4, c4)
        u4 = self.up_conv4(u4)
        
        u3 = self.up3(u4)         # [B,256,H/4,W/4]
        u3 = _crop_concat(u3, c3)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)         # [B,128,H/2,W/2]
        u2 = _crop_concat(u2, c2)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)         # [B,64,H,W]
        u1 = _crop_concat(u1, c1)
        u1 = self.up_conv1(u1)
        
        return torch.tanh(self.final(u1))  # 输出归一化到[-1,1]

# 特征提取器（VGG19）
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[:4]    # conv1_2
        self.slice2 = vgg[4:9]   # conv2_2
        self.slice3 = vgg[9:18]  # conv3_4
        self.slice4 = vgg[18:27] # conv4_4
        
        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        h = self.slice1(x)
        f1 = h  # 风格特征
        h = self.slice2(h)
        f2 = h  # 风格特征
        h = self.slice3(h)
        f3 = h  # 内容特征
        h = self.slice4(h)
        f4 = h  # 内容特征
        return [f1, f2, f3, f4]

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

def load_model(target_dir: str, model_name: str) -> nn.Module:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    model = UNetStyleGenerator()
    model.load_state_dict(torch.load(f=model_save_path, weights_only=True))
    return model

def train():
    # 初始化网络
    generator = OriginalUNet().to(config["device"])
    extractor = VGGFeatureExtractor().to(config["device"])
    optimizer = optim.Adam(generator.parameters(), lr=0.001)

    # 获取参考特征
    content_features = extractor(content_img)
    style_features = extractor(style_img)
    style_grams = [gram_matrix(f) for f in style_features]

    # 训练循环
    for step in range(config["epochs"]):
        # 生成图像
        generated = generator(content_img)
        gen_features = extractor(generated)
        
        # 内容损失（conv3_4和conv4_4）
        content_loss = 0
        for i in [2, 3]:  # 使用深层特征
            content_loss += torch.mean((gen_features[i] - content_features[i])**2)
        
        # 风格损失（conv1_2和conv2_2）
        style_loss = 0
        for i in [0, 1]:  # 使用浅层特征
            gen_gram = gram_matrix(gen_features[i])
            style_loss += torch.mean((gen_gram - style_grams[i])**2)
        
        # 总损失
        total_loss = config["content_weight"] * content_loss + config["style_weight"] * style_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 保存中间结果
        if step % 50 == 0 or step == config["epochs"]-1:
            swanlab.log({"content_loss": content_loss, "style_loss": style_loss, "total_loss": total_loss.item()})
            output = generated.detach().cpu().squeeze()
            output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            image = swanlab.Image(output.clamp(0,1), caption="example")
            swanlab.log({"Output": image})

            print(f"Step {step}: Total Loss={total_loss.item():.2f}")
            
        if step == config["epochs"]-1:
            save_image(output.clamp(0,1), f"run1_output_{step}.png")

    print("Training completed!")
    save_model(generator, target_dir="models", model_name="test_model1.pth")

def test():
    # 初始化网络
    generator = load_model(target_dir="models", model_name="test_model1.pth").to(config["device"])

    generated = generator(content_img)
    output = generated.clone().detach().cpu().squeeze()
    output = output * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    output += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    save_image(output.clamp(0,1), f"run1_test.png")

config= {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "content_path": "content1.JPEG",
    "style_path": "style1.jpg",
    "epochs": 500,
    "content_weight": 1,       # 内容损失权重
    "style_weight": 1e6       # 风格损失权重
}

if __name__ == "__main__":

    content_img = load_image(config["content_path"])  # 内容图像路径
    style_img = load_image(config["style_path"])      # 风格图像路径

    run = swanlab.init(
        project="StyleTransfer",
        experiment_name="vgg_origin_unet",
        description="vgg + 原始 U-Net 进行特定风格迁移",
        config=config
    )

    train()
    # test()

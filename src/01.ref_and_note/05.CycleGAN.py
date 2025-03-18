import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import swanlab

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def load_image(path: str):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = Image.new('RGB', image.size).paste(image)
    return image

class GeneratorResNet(nn.Module):
    def __init__(self, input_channels: int, n_residual_blocks: int):
        super().__init__()
        
        out_features = 64
        layers = [
            nn.Conv2d(input_channels, out_features, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        
        for _ in range(2):
            out_features *= 2
            layers += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(n_residual_blocks):
            layers += [ResidualBlock(out_features)]
            
        for _ in range(2):
            out_features //= 2
            layers += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        layers += [nn.Conv2d(out_features, input_channels, kernel_size=7, stride=1, padding=3), nn.Tanh()]
        
        self.model = nn.Sequential(*layers)
        self.apply(weight_init_normal)

    def forward(self, x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        self.model = nn.Sequential(
            DiscriminatorBlock(channels, 64, normalize=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )
        
        self.apply(weight_init_normal)
        
    def forward(self, img):
        return self.model(img)
    
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, dataset_name: str, transforms_, mode: str):
        self.transform = transforms.Compose(transforms_)
        
        root = Path(dataset_name)
        path_a = root / 'trainA' if mode == 'train' else root / 'testA'
        path_b = root / 'trainB' if mode == 'train' else root / 'testB'
        self.files_A = sorted(list(path_a.glob('*.jpg')))
        self.files_B = sorted(list(path_b.glob('*.jpg')))
        
    def __getitem__(self, index: int):
        return (
            self.transform(load_image(self.files_A[index % len(self.files_A)])),
            self.transform(load_image(self.files_B[index % len(self.files_B)]))
        )
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []
        
    def push_and_pop(self, data: torch.Tensor):
        data = data.detach()
        res = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                res.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    res.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    res.append(element)
        return torch.stack(res)

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = config["input_shape"]
        self.generator_xy = GeneratorResNet(input_channels=self.input_shape[0],
                                            n_residual_blocks=config["n_residual_blocks"]).to(config["device"])
        self.generator_yx = GeneratorResNet(input_channels=self.input_shape[0],
                                            n_residual_blocks=config["n_residual_blocks"]).to(config["device"])
        self.discriminator_x = Discriminator(input_shape=self.input_shape).to(config["device"])
        self.discriminator_y = Discriminator(input_shape=self.input_shape).to(config["device"])
        
        self.gan_loss = nn.MSELoss()
        self.identity_loss = nn.L1Loss()
        self.cycle_loss = nn.L1Loss()
        
        self.generator_optimizer = torch.optim.Adam(
            self.generator_xy.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"])
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator_x.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"])
        )

        decay_epochs = config["decay_epochs"]
        self.generator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer,
            lr_lambda=lambda epoch: 1 - max(0, epoch - decay_epochs) / (config["epochs"] - decay_epochs)
        )
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_optimizer,
            lr_lambda=lambda epoch: 1 - max(0, epoch - decay_epochs) / (config["epochs"] - decay_epochs)
        )

        transforms_ = [
            transforms.Resize(int(config["input_shape"][1] * 1.12), Image.BICUBIC),
            transforms.RandomCrop((config["input_shape"][1], config["input_shape"][2])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        self.train_dataloader = DataLoader(
            ImageDataset(config["dataset_name"], transforms_, mode='train'),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8
        )
        self.test_dataloader = DataLoader(
            ImageDataset(config["dataset_name"], transforms_, mode='test'),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8
        )
        
    def train(self):
        gen_x_buffer = ReplayBuffer()
        gen_y_buffer = ReplayBuffer()
        
        for epoch in range(config["epochs"]):
            loss_gan_sum = 0
            loss_cycle_sum = 0
            loss_identity_sum = 0
            loss_total_sum = 0
            
            for X, y in tqdm(self.train_dataloader, desc=f"{epoch + 1} / {config['epochs']}"):
                X = X.to(config["device"])
                y = y.to(config["device"])
                true_labels = torch.ones(X.size(0), *self.discriminator_x.output_shape,
                                        device=config["device"],requires_grad=False)
                false_labels = torch.zeros(X.size(0), *self.discriminator_x.output_shape,
                                        device=config["device"],requires_grad=False)
                
                gen_x, gen_y, loss_gan, loss_cycle, loss_identity, loss_total = self.optimize_generator(X, y, true_labels)
                
                self.optimize_discriminator(X, y,
                                            gen_x_buffer.push_and_pop(gen_x),
                                            gen_y_buffer.push_and_pop(gen_y),
                                            true_labels, false_labels)
                
                loss_gan_sum += loss_gan.item()
                loss_cycle_sum += loss_cycle.item()
                loss_identity_sum += loss_identity.item()
                loss_total_sum += loss_total.item()
                
            swanlab.log({
                "cyclegan_train_loss": loss_total_sum / len(self.train_dataloader),
                "cyclegan_train_loss_gan": loss_gan_sum / len(self.train_dataloader),
                "cyclegan_train_loss_cycle": loss_cycle_sum / len(self.train_dataloader),
                "cyclegan_train_loss_identity": loss_identity_sum / len(self.train_dataloader),
                "cyclegan_train_lr": self.generator_lr_scheduler.get_last_lr()[0]
            })  
            
            if (epoch + 1) % 10 == 0:
                save_model(self, ".", f"cyclegan_epoch_{epoch + 1}.pth")
                self.test()

    def optimize_generator(self, x: torch.Tensor, y: torch.Tensor, true_label: torch.Tensor):
        self.generator_xy.train()
        self.generator_yx.train()
        
        loss_identity = (self.identity_loss(self.generator_xy(y), y) + 
                         self.identity_loss(self.generator_yx(x), x))
        
        gen_y = self.generator_xy(x)
        gen_x = self.generator_yx(y)
        
        loss_gan = (self.gan_loss(self.discriminator_y(gen_y), true_label) +
                    self.gan_loss(self.discriminator_x(gen_x), true_label))
        
        loss_cycle = (self.cycle_loss(self.generator_yx(gen_y), x) +
                      self.cycle_loss(self.generator_xy(gen_x), y))
        
        loss_total = loss_gan + config["lambda_cycle"] * loss_cycle + config["lambda_identity"] * loss_identity
        
        self.generator_optimizer.zero_grad()
        loss_total.backward()
        self.generator_optimizer.step()
        
        return gen_x, gen_y, loss_gan, loss_cycle, loss_identity, loss_total
    
    def optimize_discriminator(self, x: torch.Tensor, y: torch.Tensor,
                               gen_x: torch.Tensor, gen_y: torch.Tensor,
                               true_labels: torch.Tensor, false_labels: torch.Tensor):
        loss_discriminator = (self.gan_loss(self.discriminator_x(x), true_labels) +
                              self.gan_loss(self.discriminator_x(gen_x.detach()), false_labels) +
                              self.gan_loss(self.discriminator_y(y), true_labels) +
                              self.gan_loss(self.discriminator_y(gen_y.detach()), false_labels))
        
        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()
            
    def test(self):
        """测试模型并记录生成图像"""
        self.generator_xy.eval()
        self.generator_yx.eval()
        
        try:
            data = next(iter(self.test_dataloader)) # 获取一个 batch
        except StopIteration:
            return
        
        # 获取真实数据
        real_A, real_B = data[0].to(config['device']), data[1].to(config['device'])
        
        with torch.inference_mode():
            fake_B = self.generator_xy(real_A)          # G(A)
            fake_A = self.generator_yx(real_B)          # F(B)
            reconstructed_A = self.generator_yx(fake_B) # F(G(A))
            reconstructed_B = self.generator_xy(fake_A) # G(F(B))
            
            def tensor_to_grid(tensor):
                return torchvision.utils.make_grid(
                    tensor.detach(),
                    nrow=3,
                    normalize=True,
                    scale_each=True
                )

            def create_comparison_grid(real, fake, recon):
                comparison = torch.stack([
                    real.cpu(),
                    fake.cpu(),
                    recon.cpu()
                ], dim=1).flatten(0, 1)
                return tensor_to_grid(comparison)

            grid_A = create_comparison_grid(real_A[:3], fake_B[:3], reconstructed_A[:3])
            grid_B = create_comparison_grid(real_B[:3], fake_A[:3], reconstructed_B[:3])

            final_grid = torch.cat([grid_A, grid_B], dim=1)
            final_image = transforms.ToPILImage()(final_grid)

        swanlab.log({"cyclegan_test_generated_images": swanlab.Image(final_image)})
        
        self.generator_xy.train()
        self.generator_yx.train()
        
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
          
config = {
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "dataset_name": './dataset/monet2photo',
    "epochs": 200,
    "decay_epochs": 100,
    "batch_size": 1,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "lambda_cycle": 10,
    "lambda_identity": 5,
    "n_residual_blocks": 9,
    "input_shape": (3, 256, 256)
}
    
if __name__ == '__main__':
    model = CycleGAN()
    
    run = swanlab.init(
        project="StyleTransfer",
        experiment_name="CycleGAN_demo",
        description="CycleGAN on monet2photo",
        config=config
    )
    
    model.train()

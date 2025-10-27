import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class UNetBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return dec1


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, grid_size: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.Conv2d(32, 9, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, grid_size: int = 24):
        super().__init__()
        self.conv = ConvBlock(in_channels, 32)
        self.fc = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SOTANetwork(nn.Module):
    def __init__(self, obs_channels: int = 15, memory_channels: int = 18, grid_size: int = 24, base_channels: int = 64):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels)
        self.policy_head = PolicyHead(base_channels, grid_size)
        self.value_head = ValueHead(base_channels, grid_size)
    
    def forward(self, obs: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None or self.memory_channels == 0:
            memory = torch.zeros(
                obs.shape[0],
                self.memory_channels,
                obs.shape[2],
                obs.shape[3],
                device=obs.device,
                dtype=obs.dtype
            )
        x = torch.cat([obs, memory], dim=1) if self.memory_channels > 0 else obs
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


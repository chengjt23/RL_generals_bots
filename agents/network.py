import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import RNNMemoryEncoder


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual  # Skip connection
        return F.relu(out)


class ConvBlock(nn.Module):
    """Convolutional block with optional channel change"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class UNetBackbone(nn.Module):
    """U-Net backbone with residual blocks and skip connections"""
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        
        # Encoder with residual blocks (3 levels for 24x24 grid)
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
        
        # Decoder with residual blocks
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
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
    """Policy head outputting H×W×9 action distribution"""
    def __init__(self, in_channels: int, grid_size: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 64),
            ResidualBlock(64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 9, 1)  # 9 actions: pass + 4 directions × 2 (all/half)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ValueHead(nn.Module):
    """Value head outputting scalar state value"""
    def __init__(self, in_channels: int, grid_size: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 64),
            ResidualBlock(64),
            ConvBlock(64, 32)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SOTANetwork(nn.Module):
    """
    SOTA Network with RNN-based memory encoder.
    
    Architecture:
    1. Observations pass through RNN memory encoder
    2. RNN outputs memory representation
    3. Observation + memory concatenated and fed to U-Net backbone
    4. Policy and value heads produce outputs
    """
    
    def __init__(
        self,
        obs_channels: int = 15,
        memory_channels: int = 6,
        grid_size: int = 24,
        base_channels: int = 64,
        rnn_hidden_channels: int = 32,
        rnn_encoded_channels: int = 10,
        rnn_num_layers: int = 2,
        use_rnn_memory: bool = True
    ):
        """
        Args:
            obs_channels: Number of observation channels
            memory_channels: Number of memory channels output by RNN (4-8 recommended)
            grid_size: Size of the game grid
            base_channels: Base channels for U-Net
            rnn_hidden_channels: Hidden channels in RNN layers
            rnn_encoded_channels: Encoded observation channels for RNN (8-12 recommended)
            rnn_num_layers: Number of RNN layers
            use_rnn_memory: If True, use RNN encoder; if False, use zero memory (for backward compatibility)
        """
        super().__init__()
        self.obs_channels = obs_channels
        self.memory_channels = memory_channels
        self.grid_size = grid_size
        self.use_rnn_memory = use_rnn_memory
        
        # RNN memory encoder with improved architecture
        if use_rnn_memory:
            self.memory_encoder = RNNMemoryEncoder(
                obs_channels=obs_channels,
                encoded_channels=rnn_encoded_channels,
                hidden_channels=rnn_hidden_channels,
                memory_channels=memory_channels,
                num_layers=rnn_num_layers
            )
        else:
            self.memory_encoder = None
        
        # U-Net backbone processes only observations (late fusion)
        self.backbone = UNetBackbone(obs_channels, base_channels)
        
        # Memory projection for late fusion
        self.memory_proj = nn.Sequential(
            nn.Conv2d(memory_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # Policy and value heads
        self.policy_head = PolicyHead(base_channels, grid_size)
        self.value_head = ValueHead(base_channels, grid_size)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        visibility_mask: torch.Tensor | None = None,
        return_hidden: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with RNN memory encoding and late fusion.
        
        Args:
            obs: Observation tensor (batch, obs_channels, height, width)
            hidden_state: RNN hidden state from previous timestep.
                         List of (h, c) tuples for each LSTM layer.
                         If None, will be initialized with zeros.
            visibility_mask: Binary mask (batch, 1, height, width) where 1=visible, 0=fog.
                           Used to preserve memory on hidden tiles.
            return_hidden: If True, returns updated hidden state as third output
        
        Returns:
            policy_logits: Policy logits (batch, 9, height, width)
            value: State value (batch, 1)
            hidden_state_next (optional): Updated hidden state if return_hidden=True
        """
        batch_size = obs.shape[0]
        height, width = obs.shape[2], obs.shape[3]
        
        # Process observations through U-Net backbone
        obs_features = self.backbone(obs)
        
        if self.use_rnn_memory and self.memory_encoder is not None:
            # Generate memory representation using RNN with visibility gating
            memory, hidden_state_next = self.memory_encoder(
                obs, hidden_state, visibility_mask
            )
            
            # Late fusion: project memory and combine with observation features
            mem_features = self.memory_proj(memory)
            features = obs_features + mem_features
        else:
            # Fallback: use only observation features
            features = obs_features
            hidden_state_next = None
        
        # Generate policy and value outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        if return_hidden:
            return policy_logits, value, hidden_state_next
        else:
            return policy_logits, value
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Initialize RNN hidden state."""
        if self.use_rnn_memory and self.memory_encoder is not None:
            return self.memory_encoder.init_hidden(
                batch_size, self.grid_size, self.grid_size, device, dtype
            )
        return None

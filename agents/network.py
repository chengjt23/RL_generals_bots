import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, channels)
    
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
        self.norm = nn.GroupNorm(8, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class UNetBackbone(nn.Module):
    """U-Net backbone with residual blocks and skip connections"""
    def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
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
        
        # ConvLSTM at bottleneck
        # Input channels: base_channels * 8 (from bottleneck)
        # Hidden channels: base_channels * 4 (reduced for efficiency)
        self.conv_lstm = ConvLSTMCell(
            input_dim=base_channels * 8,
            hidden_dim=base_channels * 4,
            kernel_size=(3, 3),
            bias=True
        )
        
        # Decoder with residual blocks
        # Input channels: hidden_dim (base_channels * 4) + bottleneck (base_channels * 8)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 12, base_channels * 4, 2, stride=2)
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
    
    def forward(self, x: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape for Encoder: (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)
        
        # Encoder with skip connections
        enc1 = self.enc1(x_reshaped)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3)) # (B*T, 512, 3, 3)
        
        # ConvLSTM Processing
        # Reshape to (B, T, C, H, W) for sequential processing
        _, C_b, H_b, W_b = bottleneck.shape
        bottleneck_seq = bottleneck.view(B, T, C_b, H_b, W_b)
        
        if hidden_state is None:
            hidden_state = self.conv_lstm.init_hidden(B, (H_b, W_b))
            
        h, c = hidden_state
        outputs = []
        
        for t in range(T):
            # No dropout before LSTM to preserve temporal consistency
            input_t = bottleneck_seq[:, t]
            h, c = self.conv_lstm(input_t, (h, c))
            outputs.append(h)
            
        # Stack outputs: (B, T, C, H, W)
        lstm_out = torch.stack(outputs, dim=1)
        new_hidden_state = (h, c)
        
        # Reshape for Decoder: (B*T, C, H, W)
        lstm_out_reshaped = lstm_out.view(B * T, -1, H_b, W_b)
        
        # Concatenate LSTM output with original bottleneck features (skip connection)
        decoder_input = torch.cat([lstm_out_reshaped, bottleneck], dim=1)
        decoder_input = self.dropout(decoder_input)
        
        # Decoder with skip connections
        dec3 = self.upconv3(decoder_input)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        # Reshape back to (B, T, C, H, W)
        # Output shape: (B, T, base_channels, H, W)
        final_out = dec1.view(B, T, -1, H, W)
        
        return final_out, new_hidden_state


class PolicyHead(nn.Module):
    """Policy head outputting H×W×9 action distribution"""
    def __init__(self, in_channels: int, grid_size: int = 24):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32),
            ResidualBlock(32),
            ConvBlock(32, 16),
            nn.Conv2d(16, 9, 1)  # 9 actions: pass + 4 directions × 2 (all/half)
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
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

32
class SOTANetwork(nn.Module):
    def __init__(self, obs_channels: int = 15, memory_channels: int = 18, grid_size: int = 24, base_channels: int = 64, dropout: float = 0.0):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels, dropout=dropout)
        self.policy_head = PolicyHead(base_channels, grid_size)
        self.value_head = ValueHead(base_channels, grid_size)
    
    def forward(self, obs: torch.Tensor, memory: torch.Tensor | None = None, hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # obs shape: (B, T, C, H, W)
        # memory shape: (B, T, C, H, W)
        
        if memory is None or self.memory_channels == 0:
            memory = torch.zeros(
                obs.shape[0],
                obs.shape[1], # Time dimension
                self.memory_channels,
                obs.shape[3],
                obs.shape[4],
                device=obs.device,
                dtype=obs.dtype
            )
            
        x = torch.cat([obs, memory], dim=2) if self.memory_channels > 0 else obs
        
        # Backbone now returns features and new hidden state
        features, new_hidden_state = self.backbone(x, hidden_state)
        
        # Reshape features for heads: (B*T, C, H, W)
        B, T, C, H, W = features.shape
        features_reshaped = features.view(B * T, C, H, W)
        
        policy_logits = self.policy_head(features_reshaped)
        value = self.value_head(features_reshaped)
        
        # Reshape outputs back to (B, T, ...)
        policy_logits = policy_logits.view(B, T, -1, H, W)
        value = value.view(B, T, 1)
        
        return policy_logits, value, new_hidden_state

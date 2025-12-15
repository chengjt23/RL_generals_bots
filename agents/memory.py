"""
RNN-based memory encoder for the Generals.io agent.

This module replaces the hand-crafted memory augmentation with a learned
recurrent memory system using ConvLSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    """Compress raw observations to low-bandwidth semantic representation.
    
    This bottleneck prevents ConvLSTM from simply memorizing current frame.
    """
    
    def __init__(self, obs_channels: int = 15, encoded_channels: int = 10):
        super().__init__()
        # Aggressive channel reduction without skip connections or residuals
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, encoded_channels, 3, padding=1),
            nn.BatchNorm2d(encoded_channels),
            nn.ReLU()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to compressed representation."""
        return self.encoder(obs)


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing spatial observations with write gate."""
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
        
        # Explicit write gate for selective memory updates
        self.write_gate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        visibility_mask: torch.Tensor | None = None
    ):
        """
        Args:
            x: Input tensor (batch, input_channels, height, width)
            hidden_state: Tuple of (h, c) where each is (batch, hidden_channels, height, width)
            visibility_mask: Binary mask (batch, 1, height, width) where 1=visible, 0=hidden
        
        Returns:
            Tuple of (h_next, c_next)
        """
        batch_size, _, height, width = x.size()
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        # Split into 4 gates
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Compute candidate updates
        c_update = f * c + i * g
        h_update = o * torch.tanh(c_update)
        
        # Apply write gate for selective memory updates
        write = torch.sigmoid(self.write_gate(combined))
        h_gated = write * h_update + (1 - write) * h
        c_gated = write * c_update + (1 - write) * c
        
        # Apply visibility masking: update visible, preserve hidden
        if visibility_mask is not None:
            # Ensure mask broadcasts correctly
            mask = visibility_mask.expand_as(h_gated)
            h_next = mask * h_gated + (1 - mask) * h
            c_next = mask * c_gated + (1 - mask) * c
        else:
            h_next = h_gated
            c_next = c_gated
        
        return h_next, c_next


class RNNMemoryEncoder(nn.Module):
    """
    RNN-based memory encoder using ConvLSTM with improvements:
    1. Observation encoder bottleneck (prevents frame memorization)
    2. Visibility-based gating (preserves hidden tile memory)
    3. Explicit write gates (selective updates)
    4. Reduced memory dimensionality (symbolic representation)
    """
    
    def __init__(
        self,
        obs_channels: int = 15,
        encoded_channels: int = 10,
        hidden_channels: int = 32,
        memory_channels: int = 6,
        num_layers: int = 2,
        kernel_size: int = 3
    ):
        """
        Args:
            obs_channels: Number of input observation channels
            encoded_channels: Compressed observation channels (8-12 recommended)
            hidden_channels: Number of hidden channels in ConvLSTM layers
            memory_channels: Number of output memory channels (4-8 recommended)
            num_layers: Number of ConvLSTM layers
            kernel_size: Kernel size for ConvLSTM convolutions
        """
        super().__init__()
        self.obs_channels = obs_channels
        self.encoded_channels = encoded_channels
        self.hidden_channels = hidden_channels
        self.memory_channels = memory_channels
        self.num_layers = num_layers
        
        # Observation encoder: compress to low-bandwidth semantic features
        self.obs_encoder = ObservationEncoder(obs_channels, encoded_channels)
        
        # Build ConvLSTM layers
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = encoded_channels if i == 0 else hidden_channels
            self.lstm_cells.append(
                ConvLSTMCell(input_dim, hidden_channels, kernel_size)
            )
        
        # Project hidden state to memory representation
        self.memory_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, memory_channels, 1)
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        visibility_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process observation through encoder and ConvLSTM to generate memory.
        
        Args:
            obs: Observation tensor (batch, obs_channels, height, width)
            hidden_state: List of (h, c) tuples for each LSTM layer.
                         Each tuple contains tensors of shape (batch, hidden_channels, height, width)
            visibility_mask: Binary mask (batch, 1, height, width) where 1=visible, 0=hidden.
                           If provided, memory is preserved on hidden tiles.
        
        Returns:
            memory: Memory representation (batch, memory_channels, height, width)
            hidden_state_next: Updated hidden states for next timestep
        """
        batch_size, _, height, width = obs.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, height, width, obs.device, obs.dtype)
        
        # Encode observation to compressed semantic representation
        z = self.obs_encoder(obs)
        
        # Process through ConvLSTM layers with visibility gating
        x = z
        hidden_state_next = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = lstm_cell(x, hidden_state[i], visibility_mask)
            hidden_state_next.append((h, c))
            x = h
        
        # Project to memory representation
        memory = self.memory_proj(x)
        
        return memory, hidden_state_next
    
    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states with zeros."""
        hidden_state = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            hidden_state.append((h, c))
        return hidden_state


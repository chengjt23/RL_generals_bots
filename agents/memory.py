"""
RNN-based memory encoder for the Generals.io agent.

This module replaces the hand-crafted memory augmentation with a learned
recurrent memory system using ConvLSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing spatial observations."""
    
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
    
    def forward(self, x: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Args:
            x: Input tensor (batch, input_channels, height, width)
            hidden_state: Tuple of (h, c) where each is (batch, hidden_channels, height, width)
        
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
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class RNNMemoryEncoder(nn.Module):
    """
    RNN-based memory encoder using ConvLSTM.
    
    Takes raw observations and encodes them into a memory representation
    through a recurrent process, maintaining spatial structure.
    """
    
    def __init__(
        self,
        obs_channels: int = 15,
        hidden_channels: int = 32,
        memory_channels: int = 16,
        num_layers: int = 2,
        kernel_size: int = 3
    ):
        """
        Args:
            obs_channels: Number of input observation channels
            hidden_channels: Number of hidden channels in ConvLSTM layers
            memory_channels: Number of output memory channels
            num_layers: Number of ConvLSTM layers
            kernel_size: Kernel size for ConvLSTM convolutions
        """
        super().__init__()
        self.obs_channels = obs_channels
        self.hidden_channels = hidden_channels
        self.memory_channels = memory_channels
        self.num_layers = num_layers
        
        # Build ConvLSTM layers
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = obs_channels if i == 0 else hidden_channels
            self.lstm_cells.append(
                ConvLSTMCell(input_dim, hidden_channels, kernel_size)
            )
        
        # Project hidden state to memory representation
        self.memory_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, memory_channels, 1)
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process observation through ConvLSTM to generate memory representation.
        
        Args:
            obs: Observation tensor (batch, obs_channels, height, width)
            hidden_state: List of (h, c) tuples for each LSTM layer.
                         Each tuple contains tensors of shape (batch, hidden_channels, height, width)
        
        Returns:
            memory: Memory representation (batch, memory_channels, height, width)
            hidden_state_next: Updated hidden states for next timestep
        """
        batch_size, _, height, width = obs.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, height, width, obs.device, obs.dtype)
        
        # Process through ConvLSTM layers
        x = obs
        hidden_state_next = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = lstm_cell(x, hidden_state[i])
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


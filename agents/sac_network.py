import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import UNetBackbone, ConvBlock


class SACActor(nn.Module):
    def __init__(self, obs_channels=15, memory_channels=18, grid_size=24, base_channels=64):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels)
        self.policy_head = nn.Sequential(
            ConvBlock(base_channels, 32),
            nn.Conv2d(32, 9, 1)
        )
    
    def forward(self, obs, memory=None):
        if memory is None:
            memory = torch.zeros(obs.shape[0], self.memory_channels, obs.shape[2], obs.shape[3], device=obs.device, dtype=obs.dtype)
        x = torch.cat([obs, memory], dim=1) if self.memory_channels > 0 else obs
        features = self.backbone(x)
        logits = self.policy_head(features)
        return logits
    
    def get_action_and_log_prob(self, obs, memory, valid_mask=None, deterministic=False):
        logits = self.forward(obs, memory)
        batch_size = logits.size(0)
        h, w = logits.shape[2], logits.shape[3]
        logits_flat = logits.view(batch_size, 9, h * w)
        
        if valid_mask is not None:
            valid_mask_flat = valid_mask.view(batch_size, 9, h * w)
            logits_flat = logits_flat.masked_fill(~valid_mask_flat, float('-inf'))
        
        dist = torch.distributions.Categorical(logits=logits_flat.transpose(1, 2).contiguous().view(-1, 9))
        
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        action = action.view(batch_size, h * w)
        log_prob = log_prob.view(batch_size, h * w)
        entropy = entropy.view(batch_size, h * w)
        
        return action, log_prob, entropy


class SACCritic(nn.Module):
    def __init__(self, obs_channels=15, memory_channels=18, grid_size=24, base_channels=64):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels)
        self.q_head = nn.Sequential(
            ConvBlock(base_channels, 32),
            nn.Conv2d(32, 9, 1)
        )
    
    def forward(self, obs, memory=None):
        if memory is None:
            memory = torch.zeros(obs.shape[0], self.memory_channels, obs.shape[2], obs.shape[3], device=obs.device, dtype=obs.dtype)
        x = torch.cat([obs, memory], dim=1) if self.memory_channels > 0 else obs
        features = self.backbone(x)
        q_values = self.q_head(features)
        return q_values


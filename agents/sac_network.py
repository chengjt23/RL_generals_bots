import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import UNetBackbone, PolicyHead, ConvBlock, ResidualBlock


class SACActor(nn.Module):
    def __init__(self, obs_channels=15, memory_channels=18, grid_size=24, base_channels=64):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels)
        self.policy_head = PolicyHead(base_channels, grid_size)
    
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
        
        # Enhanced Q head structure (similar to Value Head)
        # ConvBlock(64→64) → ResidualBlock(64) → ConvBlock(64→32) → Conv2d(32→9)
        self.q_head = nn.Sequential(
            ConvBlock(base_channels, 64),
            ResidualBlock(64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 9, 1)  # Output 9 action Q-values
        )
    
    def forward(self, obs, memory=None):
        if memory is None:
            memory = torch.zeros(obs.shape[0], self.memory_channels, obs.shape[2], obs.shape[3], device=obs.device, dtype=obs.dtype)
        x = torch.cat([obs, memory], dim=1) if self.memory_channels > 0 else obs
        features = self.backbone(x)
        q_values = self.q_head(features)
        return q_values


def initialize_critic_from_bc_value(critic, bc_checkpoint_path, device):
    """Initialize critic's Q head from BC model's Value head
    
    Args:
        critic: SACCritic instance to initialize
        bc_checkpoint_path: Path to BC model checkpoint
        device: Device to load checkpoint on
    
    Returns:
        critic: Initialized critic
    """
    ckpt = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
    bc_state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 1. Initialize backbone
    backbone_dict = {k.replace('backbone.', ''): v 
                     for k, v in bc_state_dict.items() 
                     if k.startswith('backbone.')}
    missing, unexpected = critic.backbone.load_state_dict(backbone_dict, strict=False)
    print(f"  Critic backbone: loaded {len(backbone_dict)} parameters")
    
    # 2. Initialize Q Head's first 3 layers from Value Head
    # Value Head structure: conv.0 (ConvBlock), conv.1 (ResidualBlock), conv.2 (ConvBlock)
    # Q Head structure: 0 (ConvBlock), 1 (ResidualBlock), 2 (ConvBlock), 3 (Conv2d output)
    value_conv_mapping = {
        'conv.0': '0',  # ConvBlock(64→64)
        'conv.1': '1',  # ResidualBlock(64)
        'conv.2': '2',  # ConvBlock(64→32)
    }
    
    loaded_layers = 0
    for value_key, q_key in value_conv_mapping.items():
        value_prefix = f'value_head.{value_key}.'
        q_prefix = f'{q_key}.'
        
        layer_dict = {k.replace(value_prefix, q_prefix): v 
                      for k, v in bc_state_dict.items() 
                      if k.startswith(value_prefix)}
        
        if layer_dict:
            # Load into corresponding Q head layer
            q_layer = critic.q_head[int(q_key)]
            remapped_dict = {k.replace(q_prefix, ''): v for k, v in layer_dict.items()}
            q_layer.load_state_dict(remapped_dict, strict=False)
            loaded_layers += len(remapped_dict)
    
    print(f"  Q head: initialized first 3 layers with {loaded_layers} parameters from Value head")
    print(f"  Q head output layer (9 channels): randomly initialized")
    
    return critic


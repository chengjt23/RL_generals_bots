import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from collections import deque
from tqdm import tqdm
import copy

from agents.network import UNetBackbone, PolicyHead, SOTANetwork
from agents.env import make_env
from agents.memory import MemoryAugmentation
from agents.reward_shaping import DenseRewardFn
from generals.agents import RandomAgent
from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Logging to swanlab is disabled.")

class QNetwork(nn.Module):
    """
    Q-Network for Discrete SAC.
    Outputs Q(s, a) for all actions a.
    Architecture is identical to the Policy Network (SOTANetwork's policy head).
    """
    def __init__(self, obs_channels: int, memory_channels: int, grid_size: int, base_channels: int):
        super().__init__()
        self.memory_channels = memory_channels
        total_channels = obs_channels + memory_channels
        self.backbone = UNetBackbone(total_channels, base_channels)
        self.q_head = PolicyHead(base_channels, grid_size)

    def forward(self, obs: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        if memory is None and self.memory_channels > 0:
            # Should not happen in training if memory is properly handled
            raise ValueError("Memory must be provided if memory_channels > 0")
            
        x = torch.cat([obs, memory], dim=1) if self.memory_channels > 0 else obs
        features = self.backbone(x)
        q_values = self.q_head(features) # (B, 9, H, W)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, memory_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.memory = torch.zeros((capacity, *memory_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device) # Flattened action index
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.next_memory = torch.zeros((capacity, *memory_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def add(self, obs, memory, action, reward, next_obs, next_memory, done):
        self.obs[self.ptr] = obs
        self.memory[self.ptr] = memory
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_memory[self.ptr] = next_memory
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[ind],
            self.memory[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_obs[ind],
            self.next_memory[ind],
            self.dones[ind]
        )

class SACTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'])
        self.setup_seed()
        self.setup_dirs()
        self.setup_env()
        self.setup_agent()
        self.setup_swanlab()
        
    def setup_seed(self):
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_dirs(self):
        exp_name = self.config['logging']['experiment_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config['logging']['save_dir']) / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def setup_env(self):
        # Use RandomAgent as opponent for now
        # Use DenseRewardFn for better training signal
        reward_fn = DenseRewardFn(
            army_weight=0.001,
            land_weight=0.01,
            city_weight=0.5
        )
        
        self.env = make_env(
            opponent_class=RandomAgent,
            render_mode=None,
            reward_fn=reward_fn
        )()
        self.grid_size = self.config['model']['grid_size']
        
        # Memory augmentation
        self.memory_aug = MemoryAugmentation((self.grid_size, self.grid_size))

    def setup_agent(self):
        model_config = self.config['model']
        obs_channels = model_config['obs_channels']
        memory_channels = model_config['memory_channels']
        base_channels = model_config['base_channels']
        
        # Actor (Policy)
        self.actor = SOTANetwork(
            obs_channels=obs_channels,
            memory_channels=memory_channels,
            grid_size=self.grid_size,
            base_channels=base_channels
        ).to(self.device)
        
        # Critics (Q1, Q2)
        self.q1 = QNetwork(obs_channels, memory_channels, self.grid_size, base_channels).to(self.device)
        self.q2 = QNetwork(obs_channels, memory_channels, self.grid_size, base_channels).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # Optimizers
        lr = self.config['training']['learning_rate']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        
        # Entropy
        self.target_entropy = -np.log(1.0 / (9 * self.grid_size * self.grid_size)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Replay Buffer
        buffer_size = 100000 # Configurable
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            (obs_channels, self.grid_size, self.grid_size),
            (memory_channels, self.grid_size, self.grid_size),
            self.device
        )
        
        print(f"Actor params: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic params: {sum(p.numel() for p in self.q1.parameters())}")

    def setup_swanlab(self):
        if SWANLAB_AVAILABLE and self.config['logging'].get('use_wandb', False):
            swanlab.init(
                project=self.config['logging'].get('wandb_project', 'generals-sac'),
                workspace=self.config['logging'].get('wandb_entity', None),
                config=self.config,
                experiment_name=self.exp_dir.name,
                logdir=str(self.exp_dir),
            )

    def obs_to_tensor(self, obs_dict):
        # Convert dict observation to tensor (15 channels)
        # Assuming keys based on SOTAAgent
        keys = [
            'armies', 'generals', 'cities', 'mountains', 'neutral_cells',
            'owned_cells', 'opponent_cells', 'fog_cells', 'structures_in_fog'
        ]
        layers = [obs_dict[k] for k in keys]
        # Add timestep (scalar expanded to grid)
        # timestep is usually normalized or just raw. SOTAAgent uses tensor[13].
        # We need to check how many channels.
        # SOTAAgent uses 15 channels.
        # The 9 keys above give 9 channels.
        # We need 6 more.
        # owned_land_count, owned_army_count, opponent_land_count, opponent_army_count, timestep, priority
        # These are scalars. We expand them to (H, W).
        
        scalars = [
            obs_dict.get('owned_land_count', 0),
            obs_dict.get('owned_army_count', 0),
            obs_dict.get('opponent_land_count', 0),
            obs_dict.get('opponent_army_count', 0),
            obs_dict.get('timestep', 0),
            obs_dict.get('priority', 0)
        ]
        
        scalar_layers = [np.full((self.grid_size, self.grid_size), s, dtype=np.float32) for s in scalars]
        
        all_layers = layers + scalar_layers
        tensor = np.stack(all_layers, axis=0).astype(np.float32)
        return torch.from_numpy(tensor).to(self.device)

    def get_action(self, obs_tensor, memory_tensor, deterministic=False):
        with torch.no_grad():
            # SOTANetwork returns (logits, value). We only need logits for the policy.
            # The value head is untrained/unused in SAC (we use Q-networks instead).
            logits, _ = self.actor(obs_tensor.unsqueeze(0), memory_tensor.unsqueeze(0))
            logits = logits.squeeze(0) # (9, H, W)
            
            # Mask invalid actions?
            # For simplicity, we let the agent learn.
            # But to run in env, we need valid actions.
            # We can sample from logits, then map to action.
            
            # Flatten logits
            flat_logits = logits.view(-1)
            
            if deterministic:
                action_idx = torch.argmax(flat_logits).item()
            else:
                probs = F.softmax(flat_logits, dim=0)
                action_idx = torch.multinomial(probs, 1).item()
                
            return action_idx

    def idx_to_action(self, idx):
        # Map flat index to (pass, row, col, dir, split)
        # Shape (9, H, W)
        # 0: pass (at 0,0,0) ? No, pass is usually a separate action or encoded.
        # SOTAAgent: pass_logit = policy_logits_np[0, 0, 0]
        # action_logits = policy_logits_np[1:9]
        
        channel = idx // (self.grid_size * self.grid_size)
        rem = idx % (self.grid_size * self.grid_size)
        row = rem // self.grid_size
        col = rem % self.grid_size
        
        if channel == 0:
            return Action(to_pass=True)
        else:
            # channel 1-8 -> (dir, split)
            # 0-3: dir, split=0
            # 4-7: dir, split=1
            c = channel - 1
            direction = c // 2 # 0-3
            split = c % 2 # 0-1
            # Wait, SOTAAgent: action_logits = policy_logits_np[1:9].reshape(4, 2, h, w)
            # So 8 channels.
            # My logic: c is 0..7.
            # If reshape(4, 2), then dim 0 is dir, dim 1 is split.
            direction = c // 2
            split = c % 2
            return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))

    def update(self, batch_size):
        obs, memory, actions, rewards, next_obs, next_memory, dones = self.replay_buffer.sample(batch_size)
        
        alpha = self.log_alpha.exp()
        
        # --- Critic Update ---
        with torch.no_grad():
            # Sample next actions
            next_logits, _ = self.actor(next_obs, next_memory) # (B, 9, H, W)
            next_logits = next_logits.view(batch_size, -1)
            next_probs = F.softmax(next_logits, dim=1)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            q1_next = self.q1_target(next_obs, next_memory).view(batch_size, -1)
            q2_next = self.q2_target(next_obs, next_memory).view(batch_size, -1)
            q_next = torch.min(q1_next, q2_next)
            
            # V(s') = E[Q(s', a') - alpha * log pi(a'|s')]
            target_v = (next_probs * (q_next - alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + (1 - dones) * 0.99 * target_v # Gamma 0.99
            
        # Current Q
        q1 = self.q1(obs, memory).view(batch_size, -1)
        q2 = self.q2(obs, memory).view(batch_size, -1)
        
        # Gather Q values for taken actions
        q1_curr = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_curr = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        q1_loss = F.mse_loss(q1_curr, target_q)
        q2_loss = F.mse_loss(q2_curr, target_q)
        q_loss = q1_loss + q2_loss
        
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        # --- Actor Update ---
        # SOTANetwork returns (logits, value). Ignore value.
        logits, _ = self.actor(obs, memory)
        logits = logits.view(batch_size, -1)
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        
        # Re-evaluate Q with current critics (or targets? usually current)
        q1_pi = self.q1(obs, memory).view(batch_size, -1)
        q2_pi = self.q2(obs, memory).view(batch_size, -1)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # J = E[alpha * log pi(a|s) - Q(s, a)]
        actor_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Alpha Update ---
        alpha_loss = -(self.log_alpha * (probs.detach() * (log_probs + self.target_entropy).detach()).sum(dim=1)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # --- Soft Update Targets ---
        tau = 0.005
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        return q_loss.item(), actor_loss.item(), alpha_loss.item(), alpha.item()

    def train(self):
        print("Starting training...")
        obs_dict, _ = self.env.reset()
        self.memory_aug.reset()
        
        obs_tensor = self.obs_to_tensor(obs_dict)
        memory_tensor = torch.from_numpy(self.memory_aug.get_memory_features()).float().to(self.device)
        
        total_steps = self.config['training']['steps_per_epoch'] * self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']
        start_steps = self.config['training']['warmup_steps']
        
        episode_reward = 0
        episode_steps = 0
        episodes = 0
        
        pbar = tqdm(range(total_steps))
        for step in pbar:
            # Select action
            if step < start_steps:
                action_idx = np.random.randint(0, 9 * self.grid_size * self.grid_size)
            else:
                action_idx = self.get_action(obs_tensor, memory_tensor)
            
            action = self.idx_to_action(action_idx)
            
            # Step env
            next_obs_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update memory
            action_arr = np.array([
                int(action.to_pass),
                action.row if action.row is not None else 0,
                action.col if action.col is not None else 0,
                action.direction if action.direction is not None else 0,
                int(action.to_split) if action.to_split is not None else 0
            ], dtype=np.int8)
            
            self.memory_aug.update(next_obs_dict, action_arr)
            
            next_memory_tensor = torch.from_numpy(self.memory_aug.get_memory_features()).float().to(self.device)
            next_obs_tensor = self.obs_to_tensor(next_obs_dict)
            
            # Add to buffer
            self.replay_buffer.add(
                obs_tensor, memory_tensor, action_idx, reward, next_obs_tensor, next_memory_tensor, float(done)
            )
            
            obs_tensor = next_obs_tensor
            memory_tensor = next_memory_tensor
            obs_dict = next_obs_dict
            episode_reward += reward
            episode_steps += 1
            
            if done:
                episodes += 1
                if SWANLAB_AVAILABLE and self.config['logging'].get('use_wandb', False):
                    swanlab.log({"episode_reward": episode_reward, "episode_len": episode_steps})
                
                obs_dict, _ = self.env.reset()
                self.memory_aug.reset()
                obs_tensor = self.obs_to_tensor(obs_dict)
                memory_tensor = torch.from_numpy(self.memory_aug.get_memory_features()).float().to(self.device)
                episode_reward = 0
                episode_steps = 0
            
            # Update parameters
            if step >= start_steps:
                q_loss, actor_loss, alpha_loss, alpha_val = self.update(batch_size)
                
                if step % 100 == 0:
                    pbar.set_postfix({
                        'rew': f"{episode_reward:.2f}",
                        'q_loss': f"{q_loss:.4f}",
                        'pi_loss': f"{actor_loss:.4f}",
                        'alpha': f"{alpha_val:.4f}"
                    })
                    if SWANLAB_AVAILABLE and self.config['logging'].get('use_wandb', False):
                        swanlab.log({
                            "q_loss": q_loss,
                            "actor_loss": actor_loss,
                            "alpha_loss": alpha_loss,
                            "alpha": alpha_val,
                            "step": step
                        })
            
            # Save checkpoint
            if step > 0 and step % self.config['training']['eval_every'] == 0:
                self.save_checkpoint(step)

    def save_checkpoint(self, step):
        path = self.ckpt_dir / f"sac_step_{step}.pt"
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'step': step
        }, path)
        print(f"Saved checkpoint: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sac.yaml')
    args = parser.parse_args()
    
    trainer = SACTrainer(args.config)
    trainer.train()

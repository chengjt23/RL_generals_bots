import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from generals.agents import Agent
from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

from .sac_network import SACActor, SACCritic
from .memory import MemoryAugmentation


class SACAgent(Agent):
    def __init__(
        self,
        id: str = "SAC",
        grid_size: int = 24,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        bc_model_path: str | None = None,
        model_path: str | None = None,
        memory_channels: int = 18,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # Offline RL parameters
        cql_alpha: float = 0.0,           # CQL penalty weight (0 = disabled)
        bc_weight: float = 0.0,           # BC regularization weight (0 = disabled)
        gradient_clip: float = 1.0,       # Gradient clipping
    ):
        super().__init__(id)
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha
        
        # Offline RL settings
        self.cql_alpha = cql_alpha
        self.bc_weight = bc_weight
        self.gradient_clip = gradient_clip
        
        self.actor = SACActor(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_1 = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_2 = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_1_target = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_2_target = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device))

        if bc_model_path is not None:
            self.load_bc_weights(bc_model_path)
        
        action_space_size = 9 * grid_size * grid_size
        self.target_entropy = float(-np.log(1.0 / action_space_size) * 0.98)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        if model_path is not None:
            self.load(model_path)
        else:
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.memory = MemoryAugmentation((grid_size, grid_size), history_length=7)
        self.last_action = None
        self.opponent_last_action = Action(to_pass=True)
    
    def load_bc_weights(self, bc_model_path):
        if bc_model_path is None:
            print("  No BC model path provided, using random initialization")
            return
        
        ckpt = torch.load(bc_model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            bc_state_dict = ckpt['model_state_dict']
        else:
            bc_state_dict = ckpt
        
        actor_state_dict = {k: v for k, v in bc_state_dict.items() if k.startswith('backbone.') or k.startswith('policy_head.')}
        missing, unexpected = self.actor.load_state_dict(actor_state_dict, strict=False)
        
        actor_num_params = sum(p.numel() for p in self.actor.parameters())
        actor_loaded_params = sum(v.numel() for v in actor_state_dict.values())
        
        print(f"  Actor: Loaded {len(actor_state_dict)} parameter tensors ({actor_loaded_params:,} values)")
        print(f"         Total actor parameters: {actor_num_params:,}")
        if missing:
            print(f"         Missing keys: {missing}")
        if unexpected:
            print(f"         Unexpected keys: {unexpected}")
        
        critic_state_dict = {k: v for k, v in bc_state_dict.items() if k.startswith('backbone.')}
        self.critic_1.load_state_dict(critic_state_dict, strict=False)
        self.critic_2.load_state_dict(critic_state_dict, strict=False)
        
        critic_num_params = sum(p.numel() for p in self.critic_1.parameters())
        critic_loaded_params = sum(v.numel() for v in critic_state_dict.values())
        
        print(f"  Critics: Loaded {len(critic_state_dict)} parameter tensors ({critic_loaded_params:,} values)")
        print(f"           Total critic parameters (each): {critic_num_params:,}")
    
    def load(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'actor_state_dict' in ckpt:
            self.actor.load_state_dict(ckpt['actor_state_dict'])
            print(f"  Loaded actor state dict")
        
        if 'critic_1_state_dict' in ckpt:
            self.critic_1.load_state_dict(ckpt['critic_1_state_dict'])
            print(f"  Loaded critic_1 state dict")
        
        if 'critic_2_state_dict' in ckpt:
            self.critic_2.load_state_dict(ckpt['critic_2_state_dict'])
            print(f"  Loaded critic_2 state dict")
        
        if 'critic_1_target_state_dict' in ckpt:
            self.critic_1_target.load_state_dict(ckpt['critic_1_target_state_dict'])
            print(f"  Loaded critic_1_target state dict")
        else:
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            print(f"  Initialized critic_1_target from critic_1")
        
        if 'critic_2_target_state_dict' in ckpt:
            self.critic_2_target.load_state_dict(ckpt['critic_2_target_state_dict'])
            print(f"  Loaded critic_2_target state dict")
        else:
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())
            print(f"  Initialized critic_2_target from critic_2")
        
        if 'actor_optimizer' in ckpt:
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            print(f"  Loaded actor optimizer state")
        
        if 'critic_optimizer' in ckpt:
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            print(f"  Loaded critic optimizer state")
        
        if 'alpha_optimizer' in ckpt:
            self.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
            print(f"  Loaded alpha optimizer state")
        
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                if isinstance(ckpt['log_alpha'], torch.Tensor):
                    self.log_alpha.copy_(ckpt['log_alpha'].to(self.device))
                else:
                    self.log_alpha.fill_(ckpt['log_alpha'])
            print(f"  Loaded log_alpha: {self.log_alpha.item():.4f}")
        
        epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"  Checkpoint info: epoch={epoch}, global_step={global_step}")
        
        return epoch, global_step

    def reset(self):
        self.memory.reset()
        self.last_action = None
        self.opponent_last_action = Action(to_pass=True)
    
    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        if self.last_action is not None:
            self.memory.update(self._obs_to_dict(observation), self.last_action, self.opponent_last_action)
        
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            action = self._sample_action(obs_tensor, memory_tensor, observation, deterministic)
        
        self.last_action = action
        return action
    
    def _obs_to_dict(self, obs: Observation) -> dict:
        return {
            "armies": obs.armies,
            "generals": obs.generals,
            "cities": obs.cities,
            "mountains": obs.mountains,
            "neutral_cells": obs.neutral_cells,
            "owned_cells": obs.owned_cells,
            "opponent_cells": obs.opponent_cells,
            "fog_cells": obs.fog_cells,
            "structures_in_fog": obs.structures_in_fog,
        }
    
    def _prepare_observation(self, obs: Observation) -> torch.Tensor:
        obs.pad_observation(pad_to=self.grid_size)
        obs_tensor = torch.from_numpy(obs.as_tensor()).float().unsqueeze(0).to(self.device)
        return obs_tensor
    
    def _prepare_memory(self) -> torch.Tensor:
        memory_features = self.memory.get_memory_features()
        
        current_h, current_w = memory_features.shape[1], memory_features.shape[2]
        if current_h < self.grid_size or current_w < self.grid_size:
            pad_h = max(0, self.grid_size - current_h)
            pad_w = max(0, self.grid_size - current_w)
            memory_features = np.pad(memory_features, 
                                    ((0, 0), (0, pad_h), (0, pad_w)), 
                                    mode='constant', 
                                    constant_values=0)
        
        memory_tensor = torch.from_numpy(memory_features).float().unsqueeze(0).to(self.device)
        return memory_tensor
    
    def _sample_action(self, obs_tensor, memory_tensor, observation, deterministic):
        valid_mask = compute_valid_move_mask(observation)
        h, w = valid_mask.shape[:2]
        
        policy_logits = self.actor(obs_tensor, memory_tensor).squeeze(0)
        policy_logits_np = policy_logits.cpu().numpy()
        
        pass_logit = policy_logits_np[0, 0, 0]
        action_logits = policy_logits_np[1:9].reshape(4, 2, h, w)
        
        masked_logits = []
        valid_actions = []
        
        for direction in range(4):
            for split in range(2):
                logits_slice = action_logits[direction, split]
                mask_slice = valid_mask[:, :, direction]
                valid_positions = np.argwhere(mask_slice > 0)
                for pos in valid_positions:
                    row, col = pos
                    masked_logits.append(logits_slice[row, col])
                    valid_actions.append((row, col, direction, split))
        
        if len(masked_logits) == 0:
            return Action(to_pass=True)
        
        all_logits = np.array([pass_logit] + masked_logits)
        
        if deterministic:
            choice = np.argmax(all_logits)
        else:
            probs = torch.softmax(torch.from_numpy(all_logits), dim=0).numpy()
            choice = np.random.choice(len(all_logits), p=probs)
        
        if choice == 0:
            return Action(to_pass=True)
        
        row, col, direction, split = valid_actions[choice - 1]
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))
    
    def update(self, batch):
        obs = batch['observations']
        memory = batch['memories']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        next_memory = batch['next_memories']
        dones = batch['dones']
        
        # ============ Critic Update with CQL ============
        with torch.no_grad():
            next_policy_logits = self.actor(next_obs, next_memory)
            q1_next = self.critic_1_target(next_obs, next_memory)
            q2_next = self.critic_2_target(next_obs, next_memory)
            min_q_next = torch.min(q1_next, q2_next)
            
            batch_size = obs.size(0)
            h, w = min_q_next.shape[2], min_q_next.shape[3]
            
            next_policy_probs = F.softmax(next_policy_logits.view(batch_size, -1), dim=-1)
            next_log_probs = F.log_softmax(next_policy_logits.view(batch_size, -1), dim=-1)
            min_q_next_flat = min_q_next.view(batch_size, -1)
            
            alpha = self.log_alpha.exp()
            next_q_value = (next_policy_probs * (min_q_next_flat - alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * float(self.gamma) * next_q_value
        
        q1 = self.critic_1(obs, memory)
        q2 = self.critic_2(obs, memory)
        
        # Compute action indices
        is_pass = actions[:, 0].long()
        rows = torch.clamp(actions[:, 1], 0, h - 1).long()
        cols = torch.clamp(actions[:, 2], 0, w - 1).long()
        directions = actions[:, 3].long()
        splits = actions[:, 4].long()
        
        rows = torch.where(is_pass == 1, torch.zeros_like(rows), rows)
        cols = torch.where(is_pass == 1, torch.zeros_like(cols), cols)
        
        action_channel = torch.where(
            is_pass == 1,
            torch.zeros_like(is_pass),
            1 + directions * 2 + splits
        ).long()
        
        batch_indices = torch.arange(batch_size, device=obs.device, dtype=torch.long)
        q1_pred = q1[batch_indices, action_channel, rows, cols]
        q2_pred = q2[batch_indices, action_channel, rows, cols]
        
        # Standard Bellman loss
        bellman_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        
        # CQL penalty (if enabled)
        cql_loss = torch.tensor(0.0, device=obs.device)
        if self.cql_alpha > 0:
            q1_flat = q1.view(batch_size, -1)
            q2_flat = q2.view(batch_size, -1)
            q1_logsumexp = torch.logsumexp(q1_flat, dim=-1)
            q2_logsumexp = torch.logsumexp(q2_flat, dim=-1)
            cql_loss = (q1_logsumexp.mean() - q1_pred.mean() + 
                        q2_logsumexp.mean() - q2_pred.mean())
        
        critic_loss = bellman_loss + self.cql_alpha * cql_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            max_norm=self.gradient_clip
        )
        self.critic_optimizer.step()
        
        # ============ Actor Update with BC Regularization ============
        policy_logits = self.actor(obs, memory)
        q1_new = self.critic_1(obs, memory)
        q2_new = self.critic_2(obs, memory)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_probs = F.softmax(policy_logits.view(batch_size, -1), dim=-1)
        log_probs = F.log_softmax(policy_logits.view(batch_size, -1), dim=-1)
        min_q_new_flat = min_q_new.view(batch_size, -1)
        
        # Standard SAC actor loss
        alpha = self.log_alpha.exp()
        sac_actor_loss = (policy_probs * (alpha * log_probs - min_q_new_flat)).sum(dim=-1).mean()
        
        # BC regularization (if enabled)
        bc_loss = torch.tensor(0.0, device=obs.device)
        if self.bc_weight > 0:
            # Compute flat action indices
            action_indices_flat = action_channel * h * w + rows * w + cols
            expert_log_probs = log_probs[batch_indices, action_indices_flat]
            bc_loss = -expert_log_probs.mean()
        
        actor_loss = sac_actor_loss + self.bc_weight * bc_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.gradient_clip
        )
        self.actor_optimizer.step()
        
        # ============ Alpha Update ============
        if self.auto_tune_alpha:
            entropy = -(policy_probs * log_probs).sum(dim=-1).mean()
            # Fix: Remove negative sign to correct gradient direction
            # When entropy < target_entropy: grad < 0 → log_alpha increases → alpha increases → encourages higher entropy ✅
            # When entropy > target_entropy: grad > 0 → log_alpha decreases → alpha decreases → encourages lower entropy ✅
            alpha_loss = self.log_alpha * (entropy - torch.tensor(self.target_entropy, dtype=torch.float32, device=self.device)).detach()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ============ Target Network Update ============
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'bellman_loss': bellman_loss.item(),
            'cql_loss': cql_loss.item(),
            'actor_loss': actor_loss.item(),
            'sac_actor_loss': sac_actor_loss.item(),
            'bc_loss': bc_loss.item(),
            'alpha': alpha.item(),
            'entropy': entropy.item() if self.auto_tune_alpha else 0.0
        }
    
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


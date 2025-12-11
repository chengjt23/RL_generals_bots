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
        memory_channels: int = 18,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
    ):
        super().__init__(id)
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha
        
        self.actor = SACActor(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_1 = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_2 = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_1_target = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        self.critic_2_target = SACCritic(obs_channels=15, memory_channels=memory_channels, grid_size=grid_size, base_channels=64).to(self.device)
        
        if bc_model_path is not None:
            self.load_bc_weights(bc_model_path)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), device=self.device))
        self.target_entropy = -np.log(1.0 / 9) * 0.98
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.memory = MemoryAugmentation((grid_size, grid_size), history_length=7)
        self.last_action = None
        self.opponent_last_action = Action(to_pass=True)
    
    def load_bc_weights(self, bc_model_path):
        ckpt = torch.load(bc_model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            bc_state_dict = ckpt['model_state_dict']
        else:
            bc_state_dict = ckpt
        
        actor_state_dict = {k: v for k, v in bc_state_dict.items() if k.startswith('backbone.') or k.startswith('policy_head.')}
        self.actor.load_state_dict(actor_state_dict, strict=False)
        
        critic_state_dict = {k: v for k, v in bc_state_dict.items() if k.startswith('backbone.')}
        self.critic_1.load_state_dict(critic_state_dict, strict=False)
        self.critic_2.load_state_dict(critic_state_dict, strict=False)
    
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
        
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action_and_log_prob(next_obs, next_memory)
            q1_next = self.critic_1_target(next_obs, next_memory)
            q2_next = self.critic_2_target(next_obs, next_memory)
            min_q_next = torch.min(q1_next, q2_next)
            
            batch_size = obs.size(0)
            h, w = min_q_next.shape[2], min_q_next.shape[3]
            min_q_next_flat = min_q_next.view(batch_size, 9, h * w)
            next_actions_expanded = next_actions.unsqueeze(1)
            next_q_values = torch.gather(min_q_next_flat, 1, next_actions_expanded).squeeze(1)
            
            alpha = self.log_alpha.exp()
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * (next_q_values - alpha * next_log_probs)
        
        q1 = self.critic_1(obs, memory)
        q2 = self.critic_2(obs, memory)
        q1_flat = q1.view(batch_size, 9, h * w)
        q2_flat = q2.view(batch_size, 9, h * w)
        
        actions_idx = actions[:, 1] * w + actions[:, 2]
        actions_expanded = actions_idx.unsqueeze(1)
        q1_pred = torch.gather(q1_flat, 1, actions_expanded).squeeze(1)
        q2_pred = torch.gather(q2_flat, 1, actions_expanded).squeeze(1)
        
        critic_loss = F.mse_loss(q1_pred, target_q.mean(dim=1)) + F.mse_loss(q2_pred, target_q.mean(dim=1))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        curr_actions, log_probs, entropy = self.actor.get_action_and_log_prob(obs, memory)
        q1_new = self.critic_1(obs, memory)
        q2_new = self.critic_2(obs, memory)
        min_q_new = torch.min(q1_new, q2_new)
        min_q_new_flat = min_q_new.view(batch_size, 9, h * w)
        curr_actions_expanded = curr_actions.unsqueeze(1)
        q_values = torch.gather(min_q_new_flat, 1, curr_actions_expanded).squeeze(1)
        
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_probs - q_values).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item(),
            'entropy': entropy.mean().item()
        }
    
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


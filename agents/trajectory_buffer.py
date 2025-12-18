import numpy as np
import torch


class TrajectoryBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, memory_shape, grid_size, device='cuda'):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs_shape = obs_shape
        self.memory_shape = memory_shape
        self.grid_size = grid_size
        
        self.observations = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.valid_masks = np.zeros((n_steps, n_envs, grid_size, grid_size, 4), dtype=np.bool_)
        self.memories = np.zeros((n_steps, n_envs, *memory_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, 5), dtype=np.int32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
    
    def store_transition(self, step, env_idx, obs, memory, action, log_prob, value, reward, done, valid_mask=None):
        self.observations[step, env_idx] = obs
        if valid_mask is not None:
            self.valid_masks[step, env_idx] = valid_mask
        self.memories[step, env_idx] = memory
        self.actions[step, env_idx] = action
        self.log_probs[step, env_idx] = log_prob
        self.values[step, env_idx] = value
        self.rewards[step, env_idx] = reward
        self.dones[step, env_idx] = done
    
    def finish_trajectory(self, last_values, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE advantages and returns.
        
        GAE formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        
        Note: If done_t == 1, then next_values should be 0 (episode ended).
        """
        last_gae_lam = np.zeros(self.n_envs)
        for t in reversed(range(self.n_steps)):
            # If episode ended at step t, next_values should be 0
            # Otherwise, use the value of next state
            if t == self.n_steps - 1:
                # Last step: use last_values if episode didn't end
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values * next_non_terminal  # Set to 0 if done
            else:
                # Check if episode ended at current step
                next_non_terminal = 1.0 - self.dones[t]
                # If episode ended at t, next_values = 0
                # Otherwise, use value at t+1
                next_values = self.values[t + 1] * next_non_terminal
            
            delta = self.rewards[t] + gamma * next_values - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.returns = self.advantages + self.values
        
        if np.any(np.isnan(self.advantages)) or np.any(np.isinf(self.advantages)):
            raise ValueError("NaN or Inf detected in advantages")
        if np.any(np.isnan(self.returns)) or np.any(np.isinf(self.returns)):
            raise ValueError("NaN or Inf detected in returns")
    
    def prepare_for_training(self, normalize_advantages=True):
        total_samples = self.n_steps * self.n_envs
        
        self._obs_flat = self.observations.reshape(total_samples, *self.obs_shape)
        self._mem_flat = self.memories.reshape(total_samples, *self.memory_shape)
        self._act_flat = self.actions.reshape(total_samples, 5)
        self._logp_flat = self.log_probs.reshape(total_samples)
        self._val_flat = self.values.reshape(total_samples)
        self._adv_flat = self.advantages.reshape(total_samples).copy()
        self._ret_flat = self.returns.reshape(total_samples)
        
        if normalize_advantages:
            self._adv_flat = (self._adv_flat - self._adv_flat.mean()) / (self._adv_flat.std() + 1e-8)
        
        self._obs_tensor = torch.from_numpy(self._obs_flat).to(self.device)
        self._mem_tensor = torch.from_numpy(self._mem_flat).to(self.device)
        self._act_tensor = torch.from_numpy(self._act_flat).to(self.device)
        self._logp_tensor = torch.from_numpy(self._logp_flat).to(self.device)
        self._val_tensor = torch.from_numpy(self._val_flat).to(self.device)
        self._adv_tensor = torch.from_numpy(self._adv_flat).to(self.device)
        self._ret_tensor = torch.from_numpy(self._ret_flat).to(self.device)
    
    def get_batches(self, batch_size, normalize_advantages=True):
        total_samples = self.n_steps * self.n_envs
        
        if not hasattr(self, '_obs_tensor'):
            self.prepare_for_training(normalize_advantages)
        
        valid_masks_flat = self.valid_masks.reshape(total_samples, self.grid_size, self.grid_size, 4)
        valid_masks_tensor = torch.from_numpy(valid_masks_flat).to(self.device)
        
        indices = torch.randperm(total_samples, device=self.device)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            batch_valid_masks = valid_masks_tensor[batch_indices]
            
            yield {
                'observations': self._obs_tensor[batch_indices],
                'memories': self._mem_tensor[batch_indices],
                'actions': self._act_tensor[batch_indices],
                'log_probs': self._logp_tensor[batch_indices],
                'values': self._val_tensor[batch_indices],
                'advantages': self._adv_tensor[batch_indices],
                'returns': self._ret_tensor[batch_indices],
                'valid_masks': batch_valid_masks,
            }
    
    def clear(self):
        if hasattr(self, '_obs_tensor'):
            del self._obs_tensor, self._mem_tensor, self._act_tensor
            del self._logp_tensor, self._val_tensor, self._adv_tensor, self._ret_tensor
            del self._obs_flat, self._mem_flat, self._act_flat
            del self._logp_flat, self._val_flat, self._adv_flat, self._ret_flat


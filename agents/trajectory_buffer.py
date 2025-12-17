import numpy as np
import torch


class TrajectoryBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, memory_shape, device='cuda'):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs_shape = obs_shape
        self.memory_shape = memory_shape
        
        self.observations = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.observation_objects = [[None] * n_envs for _ in range(n_steps)]  # Store Observation objects for accurate log_prob calculation
        self.memories = np.zeros((n_steps, n_envs, *memory_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, 5), dtype=np.int32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
    
    def store_transition(self, step, env_idx, obs, memory, action, log_prob, value, reward, done, observation_obj=None):
        self.observations[step, env_idx] = obs
        self.observation_objects[step][env_idx] = observation_obj  # Store Observation object
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
    
    def get_batches(self, batch_size, normalize_advantages=True):
        total_samples = self.n_steps * self.n_envs
        
        obs_flat = self.observations.reshape(total_samples, *self.obs_shape)
        mem_flat = self.memories.reshape(total_samples, *self.memory_shape)
        act_flat = self.actions.reshape(total_samples, 5)
        logp_flat = self.log_probs.reshape(total_samples)
        val_flat = self.values.reshape(total_samples)
        adv_flat = self.advantages.reshape(total_samples)
        ret_flat = self.returns.reshape(total_samples)
        
        # Flatten observation objects
        obs_objs_flat = []
        for step in range(self.n_steps):
            for env_idx in range(self.n_envs):
                obs_objs_flat.append(self.observation_objects[step][env_idx])
        
        if normalize_advantages:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # Get observation objects for this batch
            batch_obs_objs = [obs_objs_flat[idx] for idx in batch_indices]
            
            batch = {
                'observations': torch.from_numpy(obs_flat[batch_indices]).to(self.device),
                'memories': torch.from_numpy(mem_flat[batch_indices]).to(self.device),
                'actions': torch.from_numpy(act_flat[batch_indices]).to(self.device),
                'log_probs': torch.from_numpy(logp_flat[batch_indices]).to(self.device),
                'values': torch.from_numpy(val_flat[batch_indices]).to(self.device),
                'advantages': torch.from_numpy(adv_flat[batch_indices]).to(self.device),
                'returns': torch.from_numpy(ret_flat[batch_indices]).to(self.device),
                'observation_objects': batch_obs_objs,  # Include Observation objects
            }
            
            yield batch
    
    def clear(self):
        pass


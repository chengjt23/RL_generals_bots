import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, memory_shape, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.memory_shape = memory_shape
        
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.memories = np.zeros((capacity, *memory_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 5), dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_memories = np.zeros((capacity, *memory_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def store(self, obs, memory, action, reward, next_obs, next_memory, done):
        self.observations[self.position] = obs
        self.memories[self.position] = memory
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.next_memories[self.position] = next_memory
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = dict(
            observations=torch.from_numpy(self.observations[indices]).to(self.device),
            memories=torch.from_numpy(self.memories[indices]).to(self.device),
            actions=torch.from_numpy(self.actions[indices]).to(self.device),
            rewards=torch.from_numpy(self.rewards[indices]).to(self.device),
            next_observations=torch.from_numpy(self.next_observations[indices]).to(self.device),
            next_memories=torch.from_numpy(self.next_memories[indices]).to(self.device),
            dones=torch.from_numpy(self.dones[indices]).to(self.device)
        )
        return batch
    
    def __len__(self):
        return self.size


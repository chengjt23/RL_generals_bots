import numpy as np
import torch
from pathlib import Path
from utils.elo import update_elo_ratings, softmax_weights


class OpponentPoolPPO:
    def __init__(self, max_size=20, initial_elo=1500, k_factor=32, temperature=0.5):
        self.max_size = max_size
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.temperature = temperature
        
        self.opponents = []
        self.opponent_elos = []
        self.opponent_metadata = []
    
    def add_opponent(self, agent, checkpoint_path, iteration, description="", sota_config=None):
        from agents.ppo_agent import PPOAgent
        
        if sota_config is None:
            raise ValueError("sota_config must be provided to add_opponent")
        
        frozen_agent = PPOAgent(
            sota_config=sota_config,
            id=f"Opponent_{iteration}",
            grid_size=agent.grid_size,
            device=agent.device,
            memory_channels=sota_config['memory_channels']
        )
        
        frozen_agent.network.load_state_dict(agent.network.state_dict())
        frozen_agent.network.eval()
        for param in frozen_agent.network.parameters():
            param.requires_grad = False
        
        self.opponents.append(frozen_agent)
        self.opponent_elos.append(self.initial_elo)
        self.opponent_metadata.append({
            'checkpoint_path': checkpoint_path,
            'iteration': iteration,
            'description': description,
            'games_played': 0,
            'wins': 0
        })
        
        if len(self.opponents) > self.max_size:
            weakest_idx = np.argmin(self.opponent_elos)
            self.opponents.pop(weakest_idx)
            self.opponent_elos.pop(weakest_idx)
            self.opponent_metadata.pop(weakest_idx)
        
        print(f"[Opponent Pool] Added opponent from iteration {iteration}. Pool size: {len(self.opponents)}")
    
    def sample_opponent(self):
        if len(self.opponents) == 0:
            return None, None
        
        weights = softmax_weights(self.opponent_elos, self.temperature)
        idx = np.random.choice(len(self.opponents), p=weights)
        
        return self.opponents[idx], idx
    
    def update_elo(self, agent_elo, opponent_idx, agent_won):
        if opponent_idx is None or opponent_idx >= len(self.opponent_elos):
            return agent_elo
        
        result_agent = 1.0 if agent_won else 0.0
        
        new_agent_elo, new_opponent_elo = update_elo_ratings(
            agent_elo, 
            self.opponent_elos[opponent_idx],
            result_agent,
            self.k_factor
        )
        
        self.opponent_elos[opponent_idx] = new_opponent_elo
        self.opponent_metadata[opponent_idx]['games_played'] += 1
        if agent_won:
            pass
        else:
            self.opponent_metadata[opponent_idx]['wins'] += 1
        
        return new_agent_elo
    
    def get_statistics(self):
        if len(self.opponents) == 0:
            return {}
        
        return {
            'pool_size': len(self.opponents),
            'avg_elo': np.mean(self.opponent_elos),
            'max_elo': np.max(self.opponent_elos),
            'min_elo': np.min(self.opponent_elos),
            'elo_std': np.std(self.opponent_elos)
        }
    
    def __len__(self):
        return len(self.opponents)


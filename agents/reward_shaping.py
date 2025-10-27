import numpy as np
from generals.core.observation import Observation
from generals.core.action import Action
from generals.core.rewards import RewardFn


class PotentialBasedRewardFn(RewardFn):
    def __init__(
        self,
        land_weight: float = 0.3,
        army_weight: float = 0.3,
        castle_weight: float = 0.4,
        max_ratio: float = 100.0,
        gamma: float = 0.99,
    ):
        self.land_weight = land_weight
        self.army_weight = army_weight
        self.castle_weight = castle_weight
        self.max_ratio = max_ratio
        self.gamma = gamma
    
    def _compute_potential(self, obs: Observation) -> float:
        agent_land = obs.owned_land_count
        enemy_land = obs.opponent_land_count
        agent_army = obs.owned_army_count
        enemy_army = obs.opponent_army_count
        
        agent_castles = (obs.cities & obs.owned_cells).sum()
        enemy_castles = (obs.cities & obs.opponent_cells).sum()
        
        epsilon = 1e-6
        
        land_ratio = (agent_land + epsilon) / (enemy_land + epsilon)
        army_ratio = (agent_army + epsilon) / (enemy_army + epsilon)
        castle_ratio = (agent_castles + 1) / (enemy_castles + 1)
        
        phi_land = np.log(land_ratio) / np.log(self.max_ratio)
        phi_army = np.log(army_ratio) / np.log(self.max_ratio)
        phi_castle = np.log(castle_ratio) / np.log(self.max_ratio)
        
        phi_land = np.clip(phi_land, -1.0, 1.0)
        phi_army = np.clip(phi_army, -1.0, 1.0)
        phi_castle = np.clip(phi_castle, -1.0, 1.0)
        
        potential = (
            self.land_weight * phi_land +
            self.army_weight * phi_army +
            self.castle_weight * phi_castle
        )
        
        return float(potential)
    
    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        agent_generals = (obs.generals & obs.owned_cells).sum()
        prior_agent_generals = (prior_obs.generals & prior_obs.owned_cells).sum()
        
        if agent_generals > prior_agent_generals:
            original_reward = 1.0
        elif agent_generals < prior_agent_generals:
            original_reward = -1.0
        else:
            original_reward = 0.0
        
        potential_current = self._compute_potential(obs)
        potential_prior = self._compute_potential(prior_obs)
        
        shaped_reward = original_reward + self.gamma * potential_current - potential_prior
        
        return shaped_reward


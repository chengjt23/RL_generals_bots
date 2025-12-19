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
        self._log_max_ratio = np.log(max_ratio)
    
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
        
        phi_land = np.clip(np.log(land_ratio) / self._log_max_ratio, -1.0, 1.0)
        phi_army = np.clip(np.log(army_ratio) / self._log_max_ratio, -1.0, 1.0)
        phi_castle = np.clip(np.log(castle_ratio) / self._log_max_ratio, -1.0, 1.0)
        
        return self.land_weight * phi_land + self.army_weight * phi_army + self.castle_weight * phi_castle
    
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
        
        return original_reward + self.gamma * potential_current - potential_prior
    
    def compute_batch(self, prior_obs_list, next_obs_list, prior_actions=None, return_details=False):
        n = len(prior_obs_list)
        rewards = np.empty(n, dtype=np.float32)
        reward_details = [] if return_details else None
        
        for i in range(n):
            prior_obs = prior_obs_list[i]
            obs = next_obs_list[i]
            
            agent_generals = (obs.generals & obs.owned_cells).sum()
            prior_generals = (prior_obs.generals & prior_obs.owned_cells).sum()
            
            if agent_generals > prior_generals:
                original_reward = 1.0
            elif agent_generals < prior_generals:
                original_reward = -1.0
            else:
                original_reward = 0.0
            
            potential_current = self._compute_potential(obs)
            potential_prior = self._compute_potential(prior_obs)
            potential_diff = self.gamma * potential_current - potential_prior
            
            r = original_reward + potential_diff
            
            pass_penalty = 0.0
            if prior_actions is not None:
                act = prior_actions[i]
                is_pass = act.is_pass()
                if is_pass:
                    pass_penalty = -0.05
                    r += pass_penalty
            
            rewards[i] = r
            
            if return_details:
                agent_land = obs.owned_land_count
                enemy_land = obs.opponent_land_count
                agent_army = obs.owned_army_count
                enemy_army = obs.opponent_army_count
                agent_castles = (obs.cities & obs.owned_cells).sum()
                enemy_castles = (obs.cities & obs.opponent_cells).sum()
                
                prior_agent_land = prior_obs.owned_land_count
                prior_enemy_land = prior_obs.opponent_land_count
                prior_agent_army = prior_obs.owned_army_count
                prior_enemy_army = prior_obs.opponent_army_count
                prior_agent_castles = (prior_obs.cities & prior_obs.owned_cells).sum()
                prior_enemy_castles = (prior_obs.cities & prior_obs.opponent_cells).sum()
                
                reward_details.append({
                    'original_reward': original_reward,
                    'potential_current': potential_current,
                    'potential_prior': potential_prior,
                    'potential_diff': potential_diff,
                    'pass_penalty': pass_penalty,
                    'final_reward': r,
                    'agent_generals': agent_generals,
                    'prior_generals': prior_generals,
                    'agent_land': agent_land,
                    'enemy_land': enemy_land,
                    'agent_army': agent_army,
                    'enemy_army': enemy_army,
                    'agent_castles': agent_castles,
                    'enemy_castles': enemy_castles,
                    'prior_agent_land': prior_agent_land,
                    'prior_enemy_land': prior_enemy_land,
                    'prior_agent_army': prior_agent_army,
                    'prior_enemy_army': prior_enemy_army,
                    'prior_agent_castles': prior_agent_castles,
                    'prior_enemy_castles': prior_enemy_castles,
                })
        
        if return_details:
            return rewards, reward_details
        return rewards


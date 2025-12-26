from generals.core.rewards import RewardFn
from generals.core.observation import Observation
from generals.core.action import Action
import numpy as np

class DenseRewardFn(RewardFn):
    """
    A dense reward function for Generals.io to facilitate RL training.
    Rewards changes in army, land, and cities.
    """
    def __init__(self, 
                 army_weight: float = 0.001, 
                 land_weight: float = 0.01, 
                 city_weight: float = 0.5, 
                 general_weight: float = 5.0,
                 win_weight: float = 1.0):
        self.army_weight = army_weight
        self.land_weight = land_weight
        self.city_weight = city_weight
        self.general_weight = general_weight
        self.win_weight = win_weight

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        reward = 0.0
        
        # 1. Army Change
        # Note: owned_army_count includes army on land and in cities
        army_diff = obs.owned_army_count - prior_obs.owned_army_count
        reward += army_diff * self.army_weight
        
        # 2. Land Change
        land_diff = obs.owned_land_count - prior_obs.owned_land_count
        reward += land_diff * self.land_weight
        
        # 3. City/General Capture (Approximated by large jumps in land/army or specific checks if possible)
        # We can check visible cities ownership if we want, but simple aggregate stats are often enough for shaping.
        
        # 4. Game Over (Win/Loss) - usually handled by env returning 1/-1, but we can shape it.
        # The env default reward is 1 for win, -1 for loss.
        # This reward function ADDS to the default? Or REPLACES it?
        # In PettingZooGenerals, if reward_fn is provided, it is used INSTEAD of default?
        # Usually reward_fn is called at each step.
        
        return reward

class PotentialBasedRewardFn(RewardFn):
    """
    Potential-based reward shaping: R = F(s, s') = Phi(s') - Phi(s)
    Preserves optimal policy.
    """
    def __init__(self, army_weight=0.001, land_weight=0.01):
        self.army_weight = army_weight
        self.land_weight = land_weight
        
    def _potential(self, obs: Observation) -> float:
        return (obs.owned_army_count * self.army_weight + 
                obs.owned_land_count * self.land_weight)

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        return self._potential(obs) - self._potential(prior_obs)

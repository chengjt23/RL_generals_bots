import gymnasium as gym
import numpy as np
from generals.envs import PettingZooGenerals
from generals.agents import Agent

class SingleAgentGenerals(gym.Env):
    """
    A Gymnasium wrapper for the PettingZooGenerals environment to make it suitable for single-agent RL training (e.g., PPO).
    It handles the opponent internally.
    """
    def __init__(self, opponent: Agent, name: str = "Agent", render_mode=None, grid_factory=None, reward_fn=None):
        super().__init__()
        self.opponent = opponent
        self.name = name
        self.opponent_name = opponent.id if opponent.id != name else "Opponent"
        
        self.agents = [self.name, self.opponent_name]
        self.env = PettingZooGenerals(agents=self.agents, render_mode=render_mode, grid_factory=grid_factory, reward_fn=reward_fn)
        
        # Expose observation and action spaces
        self.observation_space = self.env.observation_space(self.name)
        self.action_space = self.env.action_space(self.name)
        
        self.latest_obs = None

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.latest_obs = obs_dict
        
        if hasattr(self.opponent, "reset"):
            self.opponent.reset()
            
        return obs_dict[self.name], info_dict[self.name]

    def step(self, action):
        # Get opponent action
        opponent_obs = self.latest_obs[self.opponent_name]
        opponent_action = self.opponent.act(opponent_obs)
        
        actions = {
            self.name: action,
            self.opponent_name: opponent_action
        }
        
        obs_dict, rewards_dict, terminated, truncated, info_dict = self.env.step(actions)
        self.latest_obs = obs_dict
        
        # Handle termination/truncation
        # PettingZooGenerals usually returns booleans for terminated/truncated in 1v1 games
        # but we handle dicts just in case
        d = terminated[self.name] if isinstance(terminated, dict) else terminated
        t = truncated[self.name] if isinstance(truncated, dict) else truncated
        
        return (
            obs_dict[self.name],
            rewards_dict[self.name],
            d,
            t,
            info_dict[self.name]
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_env(opponent_class, opponent_kwargs=None, grid_factory=None, render_mode=None, reward_fn=None):
    """
    Factory function to create the environment.
    Useful for vector environments.
    """
    if opponent_kwargs is None:
        opponent_kwargs = {}
        
    def _init():
        opponent = opponent_class(**opponent_kwargs)
        env = SingleAgentGenerals(opponent=opponent, grid_factory=grid_factory, render_mode=render_mode, reward_fn=reward_fn)
        return env
    return _init

def run_episode(env: gym.Env, agent: Agent):
    """
    Runs a single episode and returns the total reward.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    if hasattr(agent, "reset"):
        agent.reset()

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    return total_reward

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import copy

from agents.ppo_agent import PPOAgent
from agents.trajectory_buffer import TrajectoryBuffer
from agents.opponent_pool_ppo import OpponentPoolPPO
from agents.parallel_envs import ParallelEnvs
from agents.reward_shaping import PotentialBasedRewardFn
from generals.agents import RandomAgent
from generals.core.observation import Observation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def action_to_array(action):
    try:
        if action.row is not None:
            return np.array([0, action.row, action.col, action.direction, int(action.split)], dtype=np.int32)
    except (AttributeError, TypeError):
        pass
    return np.array([1, 0, 0, 0, 0], dtype=np.int32)


class PPOTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.setup_seed()
        self.setup_dirs()
        self.setup_agent()
        self.setup_envs()
        self.setup_buffer()
        self.setup_opponent_pool()
        self.setup_reward_fn()
        self.setup_optimizer()
        self.setup_wandb()
        
        self.current_elo = self.config['opponent_pool']['initial_elo']
    
    def setup_seed(self):
        seed = self.config['training']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def setup_dirs(self):
        exp_name = self.config['experiment']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config['experiment']['save_dir']) / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def setup_agent(self):
        sota_config = {
            'obs_channels': self.config['model']['obs_channels'],
            'memory_channels': self.config['model']['memory_channels'],
            'grid_size': self.config['model']['grid_size'],
            'base_channels': self.config['model']['base_channels'],
        }
        
        self.agent = PPOAgent(
            sota_config=sota_config,
            id="PPO",
            grid_size=self.config['model']['grid_size'],
            device=self.device,
            model_path=self.config['experiment'].get('bc_pretrain_path'),
            memory_channels=self.config['model']['memory_channels'],
        )
        
        self.agent.network.train()
    
    def setup_envs(self):
        n_envs = self.config['training']['n_parallel_envs']
        self.envs = ParallelEnvs(n_envs)
        self.agent_memories = [copy.deepcopy(self.agent.memory) for _ in range(n_envs)]
    
    def setup_buffer(self):
        grid_size = self.config['model']['grid_size']
        obs_channels = self.config['model']['obs_channels']
        memory_channels = self.config['model']['memory_channels']
        n_steps = self.config['training']['n_steps_per_update']
        n_envs = self.config['training']['n_parallel_envs']
        
        self.buffer = TrajectoryBuffer(
            n_steps=n_steps,
            n_envs=n_envs,
            obs_shape=(obs_channels, grid_size, grid_size),
            memory_shape=(memory_channels, grid_size, grid_size),
            device=self.device
        )
    
    def setup_opponent_pool(self):
        pool_config = self.config['opponent_pool']
        self.opponent_pool = OpponentPoolPPO(
            max_size=pool_config['max_size'],
            initial_elo=pool_config['initial_elo'],
            k_factor=pool_config['k_factor'],
            temperature=pool_config['temperature']
        )
    
    def setup_reward_fn(self):
        reward_cfg = self.config['reward']
        self.reward_fn = PotentialBasedRewardFn(
            land_weight=reward_cfg['land_weight'],
            army_weight=reward_cfg['army_weight'],
            castle_weight=reward_cfg['castle_weight'],
            max_ratio=reward_cfg['max_ratio'],
            gamma=reward_cfg['gamma']
        )
    
    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.agent.network.parameters(),
            lr=self.config['training']['learning_rate']
        )
    
    def setup_wandb(self):
        if WANDB_AVAILABLE and self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                entity=self.config['logging']['wandb_entity'],
                name=self.exp_dir.name,
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def collect_trajectories(self):
        n_steps = self.config['training']['n_steps_per_update']
        n_envs = self.config['training']['n_parallel_envs']
        max_episode_steps = self.config['training']['max_episode_steps']
        
        obs_dicts, infos = self.envs.reset()
        
        current_opponents = []
        opponent_indices = []
        opponent_memories = []
        for _ in range(n_envs):
            opponent, idx = self.opponent_pool.sample_opponent()
            if opponent is None:
                opponent = RandomAgent()
                idx = None
            if hasattr(opponent, 'reset'):
                opponent.reset()
            current_opponents.append(opponent)
            opponent_indices.append(idx)
            if hasattr(opponent, 'memory'):
                opponent_memories.append(copy.deepcopy(opponent.memory))
            else:
                opponent_memories.append(None)
        
        for mem in self.agent_memories:
            mem.reset()
        
        episode_steps = [0] * n_envs
        episode_rewards = [0.0] * n_envs
        episode_results = []
        prior_observations = [obs_dict["Agent"] for obs_dict in obs_dicts]
        
        for step in range(n_steps):
            agent_actions = []
            agent_log_probs = []
            agent_values = []
            obs_tensors = []
            memory_tensors = []
            
            for env_idx in range(n_envs):
                obs = obs_dicts[env_idx]["Agent"]
                obs.pad_observation(pad_to=self.agent.grid_size)
                
                self.agent.memory = self.agent_memories[env_idx]
                
                action, log_prob, value = self.agent.act_with_value(obs)
                
                self.agent_memories[env_idx] = copy.deepcopy(self.agent.memory)
                
                agent_actions.append(action)
                agent_log_probs.append(log_prob)
                agent_values.append(value)
                
                obs_tensor = torch.from_numpy(obs.as_tensor()).float().numpy()
                memory_tensor = self.agent_memories[env_idx].get_memory_features()
                
                obs_tensors.append(obs_tensor)
                memory_tensors.append(memory_tensor)
            
            opponent_actions = []
            for env_idx, opponent in enumerate(current_opponents):
                if opponent_memories[env_idx] is not None:
                    opponent.memory = opponent_memories[env_idx]
                    opp_action = opponent.act(obs_dicts[env_idx]["Opponent"])
                    opponent_memories[env_idx] = copy.deepcopy(opponent.memory)
                else:
                    opp_action = opponent.act(obs_dicts[env_idx]["Opponent"])
                opponent_actions.append(opp_action)
            
            next_obs_dicts, rewards_dicts, dones, next_infos = self.envs.step(agent_actions, opponent_actions)
            
            for env_idx in range(n_envs):
                reward = self.reward_fn(
                    prior_observations[env_idx],
                    agent_actions[env_idx],
                    next_obs_dicts[env_idx]["Agent"]
                )
                
                action_array = action_to_array(agent_actions[env_idx])
                
                self.buffer.store_transition(
                    obs_tensors[env_idx],
                    memory_tensors[env_idx],
                    action_array,
                    agent_log_probs[env_idx],
                    agent_values[env_idx],
                    reward,
                    float(dones[env_idx])
                )
                
                episode_rewards[env_idx] += reward
                episode_steps[env_idx] += 1
                
                if dones[env_idx] or episode_steps[env_idx] >= max_episode_steps:
                    agent_reward = rewards_dicts[env_idx]["Agent"]
                    opp_reward = rewards_dicts[env_idx]["Opponent"]
                    agent_won = agent_reward > opp_reward
                    
                    if opponent_indices[env_idx] is not None:
                        episode_results.append((opponent_indices[env_idx], agent_won))
                    
                    reset_obs, reset_info = self.envs.reset_single(env_idx)
                    obs_dicts = list(obs_dicts)
                    obs_dicts[env_idx] = reset_obs
                    obs_dicts = tuple(obs_dicts)
                    
                    opponent, idx = self.opponent_pool.sample_opponent()
                    if opponent is None:
                        opponent = RandomAgent()
                        idx = None
                    if hasattr(opponent, 'reset'):
                        opponent.reset()
                    current_opponents[env_idx] = opponent
                    opponent_indices[env_idx] = idx
                    if hasattr(opponent, 'memory'):
                        opponent_memories[env_idx] = copy.deepcopy(opponent.memory)
                    else:
                        opponent_memories[env_idx] = None
                    
                    self.agent_memories[env_idx].reset()
                    
                    episode_steps[env_idx] = 0
                    episode_rewards[env_idx] = 0.0
            
            obs_dicts = next_obs_dicts
            prior_observations = [obs_dict["Agent"] for obs_dict in obs_dicts]
        
        last_values = []
        for env_idx in range(n_envs):
            obs = obs_dicts[env_idx]["Agent"]
            self.agent.memory = self.agent_memories[env_idx]
            value = self.agent.get_value(obs)
            last_values.append(value)
        
        last_values = np.array(last_values)
        
        self.buffer.finish_trajectory(
            last_values,
            gamma=self.config['training']['gamma'],
            gae_lambda=self.config['training']['gae_lambda']
        )
        
        return episode_rewards, episode_results
    
    def ppo_update(self):
        clip_epsilon = self.config['training']['clip_epsilon']
        value_clip_epsilon = self.config['training'].get('value_clip_epsilon')
        value_loss_coef = self.config['training']['value_loss_coef']
        entropy_coef = self.config['training']['entropy_coef']
        max_grad_norm = self.config['training']['max_grad_norm']
        ppo_epochs = self.config['training']['ppo_epochs']
        batch_size = self.config['training']['batch_size']
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(ppo_epochs):
            for batch in self.buffer.get_batches(batch_size):
                obs = batch['observations']
                mem = batch['memories']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                old_values = batch['values']
                advantages = batch['advantages']
                returns = batch['returns']
                
                new_log_probs, new_values, entropies = self.agent.evaluate_actions(obs, mem, actions)
                
                if torch.any(torch.isnan(new_log_probs)) or torch.any(torch.isinf(new_log_probs)):
                    raise ValueError("NaN or Inf detected in new_log_probs")
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                if value_clip_epsilon is not None:
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -value_clip_epsilon,
                        value_clip_epsilon
                    )
                    value_loss1 = (new_values - returns).pow(2)
                    value_loss2 = (value_pred_clipped - returns).pow(2)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = (new_values - returns).pow(2).mean()
                
                entropy = entropies.mean()
                
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.network.parameters(), max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def train(self):
        total_iterations = self.config['training']['total_iterations']
        eval_frequency = self.config['training']['eval_frequency']
        save_frequency = self.config['training']['save_frequency']
        log_frequency = self.config['training']['log_frequency']
        pool_update_freq = self.config['opponent_pool']['update_freq']
        warmup_iterations = self.config['opponent_pool']['warmup_iterations']
        
        print(f"\nTraining PPO Agent with Self-Play")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Total iterations: {total_iterations}")
        print(f"Steps per update: {self.config['training']['n_steps_per_update']}")
        print(f"Parallel environments: {self.config['training']['n_parallel_envs']}")
        print(f"Opponent pool warmup: {warmup_iterations} iterations")
        print(f"Opponent pool update frequency: every {pool_update_freq} iterations\n")
        
        for iteration in tqdm(range(total_iterations), desc="Training"):
            episode_rewards, episode_results = self.collect_trajectories()
            
            for opp_idx, agent_won in episode_results:
                self.current_elo = self.opponent_pool.update_elo(
                    self.current_elo, opp_idx, agent_won
                )
            
            metrics = self.ppo_update()
            
            self.buffer.clear()
            
            if iteration > warmup_iterations and iteration % pool_update_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"pool_agent_{iteration}.pt"
                torch.save(self.agent.network.state_dict(), checkpoint_path)
                self.opponent_pool.add_opponent(
                    self.agent,
                    str(checkpoint_path),
                    iteration,
                    f"checkpoint_{iteration}"
                )
            
            if iteration % log_frequency == 0:
                avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
                pool_stats = self.opponent_pool.get_statistics()
                
                log_dict = {
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'current_elo': self.current_elo,
                    **metrics,
                    **pool_stats
                }
                
                if self.use_wandb:
                    wandb.log(log_dict, step=iteration)
            
            if iteration % save_frequency == 0:
                self.save_checkpoint(iteration)
        
        self.save_checkpoint('final')
        self.envs.close()
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Final ELO: {self.current_elo:.1f}")
        pool_stats = self.opponent_pool.get_statistics()
        if pool_stats:
            print(f"Opponent Pool Size: {pool_stats['pool_size']}")
            print(f"Opponent Pool Avg ELO: {pool_stats['avg_elo']:.1f}")
        print("="*60)
    
    def save_checkpoint(self, iteration):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_elo': self.current_elo,
        }
        
        save_path = self.checkpoint_dir / f'checkpoint_{iteration}.pt'
        torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_ppo.yaml')
    args = parser.parse_args()
    
    trainer = PPOTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()


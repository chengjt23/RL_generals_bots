import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from agents.sac_agent import SACAgent
from agents.replay_buffer import ReplayBuffer
from agents.reward_shaping import PotentialBasedRewardFn
from generals import gym as generals_gym

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SACTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.setup_seed()
        self.setup_dirs()
        self.setup_env()
        self.setup_agent()
        self.setup_replay_buffer()
        self.setup_reward_fn()
        self.setup_wandb()
    
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
    
    def setup_env(self):
        self.env = generals_gym.make("Generals-v0", agents=["SAC", "RandomAgent"])
    
    def setup_agent(self):
        self.agent = SACAgent(
            id="SAC",
            grid_size=self.config['model']['grid_size'],
            device=self.device,
            bc_model_path=self.config['experiment']['bc_pretrain_path'],
            memory_channels=self.config['model']['memory_channels'],
            gamma=self.config['training']['gamma'],
            tau=self.config['training']['tau'],
            alpha=self.config['training']['alpha'],
            auto_tune_alpha=self.config['training']['auto_tune_alpha'],
            actor_lr=self.config['training']['actor_lr'],
            critic_lr=self.config['training']['critic_lr'],
            alpha_lr=self.config['training']['alpha_lr'],
        )
    
    def setup_replay_buffer(self):
        grid_size = self.config['model']['grid_size']
        obs_channels = self.config['model']['obs_channels']
        memory_channels = self.config['model']['memory_channels']
        
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['buffer_size'],
            obs_shape=(obs_channels, grid_size, grid_size),
            memory_shape=(memory_channels, grid_size, grid_size),
            device=self.device
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
    
    def train(self):
        total_timesteps = self.config['training']['total_timesteps']
        batch_size = self.config['training']['batch_size']
        learning_starts = self.config['training']['learning_starts']
        update_frequency = self.config['training']['update_frequency']
        gradient_steps = self.config['training']['gradient_steps']
        max_episode_steps = self.config['environment']['max_episode_steps']
        
        episode_rewards = []
        episode_lengths = []
        global_step = 0
        episode_num = 0
        
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        while global_step < total_timesteps:
            obs = self.env.reset()
            self.agent.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            prior_obs = obs[0]
            
            while not done and episode_length < max_episode_steps:
                action = self.agent.act(obs[0], deterministic=False)
                next_obs, _, done, info = self.env.step([action, None])
                
                reward = self.reward_fn(prior_obs, action, next_obs[0])
                
                obs_tensor = self.agent._prepare_observation(obs[0]).squeeze(0).cpu().numpy()
                memory_tensor = self.agent._prepare_memory().squeeze(0).cpu().numpy()
                next_obs_tensor = self.agent._prepare_observation(next_obs[0]).squeeze(0).cpu().numpy()
                self.agent.memory.update(self.agent._obs_to_dict(next_obs[0]), action, self.agent.opponent_last_action)
                next_memory_tensor = self.agent._prepare_memory().squeeze(0).cpu().numpy()
                
                action_array = np.array([action.row if not action.to_pass else -1, 
                                        action.col if not action.to_pass else -1,
                                        action.direction if not action.to_pass else -1,
                                        int(action.to_split) if not action.to_pass else 0], dtype=np.int32)
                
                self.replay_buffer.store(obs_tensor, memory_tensor, action_array, reward, next_obs_tensor, next_memory_tensor, float(done))
                
                obs = next_obs
                prior_obs = next_obs[0]
                episode_reward += reward
                episode_length += 1
                global_step += 1
                pbar.update(1)
                
                if global_step >= learning_starts and global_step % update_frequency == 0:
                    for _ in range(gradient_steps):
                        batch = self.replay_buffer.sample(batch_size)
                        metrics = self.agent.update(batch)
                        
                        if global_step % self.config['training']['log_frequency'] == 0:
                            if self.use_wandb:
                                wandb.log(metrics, step=global_step)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_num += 1
            
            if episode_num % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}', 'avg_length': f'{avg_length:.0f}'})
                
                if self.use_wandb:
                    wandb.log({
                        'episode_reward': episode_reward,
                        'avg_reward_10': avg_reward,
                        'episode_length': episode_length,
                        'episode': episode_num
                    }, step=global_step)
            
            if global_step % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(global_step)
        
        pbar.close()
        self.save_checkpoint('final')
        self.env.close()
    
    def save_checkpoint(self, step):
        checkpoint = {
            'step': step,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_1_state_dict': self.agent.critic_1.state_dict(),
            'critic_2_state_dict': self.agent.critic_2.state_dict(),
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'log_alpha': self.agent.log_alpha.item(),
        }
        
        save_path = self.checkpoint_dir / f'checkpoint_{step}.pt'
        torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_sac.yaml')
    args = parser.parse_args()
    
    trainer = SACTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()


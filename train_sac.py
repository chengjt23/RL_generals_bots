import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import copy

from agents.sac_agent import SACAgent
from agents.replay_buffer import ReplayBuffer
from agents.reward_shaping import PotentialBasedRewardFn
from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent

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


class OpponentPool:
    def __init__(self, max_size=10):
        self.pool = []
        self.max_size = max_size
        self.pool_stats = []
    
    def add_opponent(self, agent, step, description=""):
        frozen_agent = copy.deepcopy(agent)
        frozen_agent.actor.eval()
        for param in frozen_agent.actor.parameters():
            param.requires_grad = False
        
        self.pool.append(frozen_agent)
        self.pool_stats.append({
            'step': step,
            'description': description,
            'usage_count': 0
        })
        
        if len(self.pool) > self.max_size:
            self.pool.pop(0)
            self.pool_stats.pop(0)
        
        print(f"[Opponent Pool] Added agent from step {step}. Pool size: {len(self.pool)}")
    
    def sample_opponent(self):
        if not self.pool:
            return None
        
        idx = np.random.randint(0, len(self.pool))
        self.pool_stats[idx]['usage_count'] += 1
        return self.pool[idx]
    
    def get_recent_opponent(self):
        if not self.pool:
            return None
        return self.pool[-1]
    
    def __len__(self):
        return len(self.pool)


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
        self.env = PettingZooGenerals(agents=["SAC", "Opponent"], render_mode=None)
        self.random_opponent = RandomAgent()
        self.opponent_pool = OpponentPool(max_size=10)
        
        self.self_play_start = self.config['training'].get('self_play_start', 0)
        self.pool_update_freq = self.config['training'].get('pool_update_frequency', 10000)
        self.random_prob_after_selfplay = self.config['training'].get('random_prob_after_selfplay', 0.0)
        self.initial_pool_steps = self.config['training'].get('initial_pool_steps', 10000)
    
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
        
        print(f"\nTraining Plan:")
        if self.self_play_start > 0:
            print(f"  0-{self.self_play_start}: vs RandomAgent")
            print(f"  {self.self_play_start}+: vs Self-play (Opponent Pool)")
        else:
            print(f"  Pure Self-play from start")
            print(f"  0-{self.initial_pool_steps}: Build initial pool")
        print(f"  Pool update frequency: every {self.pool_update_freq} steps")
        if self.random_prob_after_selfplay > 0:
            print(f"  Random opponent probability: {self.random_prob_after_selfplay*100}%")
        else:
            print(f"  Pure self-play (no random opponent)")
        print()
        
        while global_step < total_timesteps:
            current_opponent, opponent_type = self._get_opponent(global_step)
            
            obs_dict, info = self.env.reset()
            self.agent.reset()
            if hasattr(current_opponent, 'reset'):
                current_opponent.reset()
            
            episode_reward = 0
            episode_length = 0
            terminated = truncated = False
            
            prior_obs = obs_dict["SAC"]
            
            while not (terminated or truncated) and episode_length < max_episode_steps:
                sac_action = self.agent.act(obs_dict["SAC"], deterministic=False)
                
                if opponent_type == "Self":
                    opponent_action = current_opponent.act(obs_dict["Opponent"], deterministic=True)
                else:
                    opponent_action = current_opponent.act(obs_dict["Opponent"])
                
                actions_dict = {"SAC": sac_action, "Opponent": opponent_action}
                next_obs_dict, rewards_dict, terminated, truncated, info = self.env.step(actions_dict)
                
                reward = self.reward_fn(prior_obs, sac_action, next_obs_dict["SAC"])
                
                obs_tensor = self.agent._prepare_observation(obs_dict["SAC"]).squeeze(0).cpu().numpy()
                memory_tensor = self.agent._prepare_memory().squeeze(0).cpu().numpy()
                next_obs_tensor = self.agent._prepare_observation(next_obs_dict["SAC"]).squeeze(0).cpu().numpy()
                self.agent.memory.update(self.agent._obs_to_dict(next_obs_dict["SAC"]), sac_action, opponent_action)
                next_memory_tensor = self.agent._prepare_memory().squeeze(0).cpu().numpy()
                
                action_array = action_to_array(sac_action)
                
                done = terminated or truncated
                self.replay_buffer.store(obs_tensor, memory_tensor, action_array, reward, next_obs_tensor, next_memory_tensor, float(done))
                
                obs_dict = next_obs_dict
                prior_obs = next_obs_dict["SAC"]
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
                
                if global_step > 0 and global_step % self.pool_update_freq == 0:
                    self.opponent_pool.add_opponent(
                        self.agent,
                        global_step,
                        f"checkpoint_{global_step}"
                    )
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_num += 1
            
            if episode_num % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.2f}', 
                    'avg_length': f'{avg_length:.0f}',
                    'pool_size': len(self.opponent_pool),
                    'opp': opponent_type
                })
                
                if self.use_wandb:
                    wandb.log({
                        'episode_reward': episode_reward,
                        'avg_reward_10': avg_reward,
                        'episode_length': episode_length,
                        'episode': episode_num,
                        'opponent_pool_size': len(self.opponent_pool),
                        'opponent_type': 1 if opponent_type == "Self" else 0
                    }, step=global_step)
            
            if global_step % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(global_step)
        
        pbar.close()
        self.save_checkpoint('final')
        
        if len(self.opponent_pool) > 0:
            print("\n" + "="*60)
            print("Opponent Pool Statistics:")
            for i, stats in enumerate(self.opponent_pool.pool_stats):
                print(f"  Agent {i+1}: Step {stats['step']}, Used {stats['usage_count']} times")
            print("="*60)
        
        self.env.close()
    
    def _get_opponent(self, step):
        if step < self.self_play_start:
            return self.random_opponent, "Random"
        
        if len(self.opponent_pool) == 0:
            if step < self.initial_pool_steps:
                return self.agent, "Self-Current"
            else:
                return self.random_opponent, "Random-Fallback"
        
        if self.random_prob_after_selfplay > 0 and np.random.random() < self.random_prob_after_selfplay:
            return self.random_opponent, "Random"
        else:
            return self.opponent_pool.sample_opponent(), "Self"
    
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


import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from agents.trajectory_buffer import TrajectoryBuffer
from agents.opponent_pool_ppo import OpponentPoolPPO
from agents.parallel_envs import ParallelEnvs
from agents.reward_shaping import PotentialBasedRewardFn
from generals.agents import RandomAgent
from generals.core.observation import Observation
from generals.core.action import Action, compute_valid_move_mask

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


_PASS_ACTION_ARRAY = np.array([1, 0, 0, 0, 0], dtype=np.int32)

def action_to_array(action, out=None):
    if out is None:
        out = np.empty(5, dtype=np.int32)
    try:
        if action.row is not None:
            out[0] = 0
            out[1] = action.row
            out[2] = action.col
            out[3] = action.direction
            out[4] = int(action.split)
            return out
    except (AttributeError, TypeError):
        pass
    out[:] = _PASS_ACTION_ARRAY
    return out


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
        self.agent_memories = [self.agent.memory.clone() for _ in range(n_envs)]
        
        grid_size = self.config['model']['grid_size']
        obs_channels = self.config['model']['obs_channels']
        memory_channels = self.config['model']['memory_channels']
        
        self._obs_buffer = torch.zeros((n_envs, obs_channels, grid_size, grid_size), dtype=torch.float32, pin_memory=True)
        self._mem_buffer = torch.zeros((n_envs, memory_channels, grid_size, grid_size), dtype=torch.float32, pin_memory=True)
        
        self._opp_obs_buffer = torch.zeros((n_envs, obs_channels, grid_size, grid_size), dtype=torch.float32, pin_memory=True)
        self._opp_mem_buffer = torch.zeros((n_envs, memory_channels, grid_size, grid_size), dtype=torch.float32, pin_memory=True)
    
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
            grid_size=grid_size,
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
    
    def _build_opponent_groups(self, current_opponents, opponent_memories):
        network_groups = {}
        random_env_indices = []
        
        for env_idx, opponent in enumerate(current_opponents):
            if hasattr(opponent, 'network') and opponent_memories[env_idx] is not None:
                network_id = id(opponent.network)
                if network_id not in network_groups:
                    network_groups[network_id] = {
                        'network': opponent.network,
                        'env_indices': [],
                        'opponents': [],
                        'memories': []
                    }
                network_groups[network_id]['env_indices'].append(env_idx)
                network_groups[network_id]['opponents'].append(opponent)
                network_groups[network_id]['memories'].append(opponent_memories[env_idx])
            else:
                random_env_indices.append(env_idx)
        
        return network_groups, random_env_indices
    
    def collect_trajectories(self):
        self.agent.network.eval() # 强制进入评估模式
        n_steps = self.config['training']['n_steps_per_update']
        n_envs = self.config['training']['n_parallel_envs']
        max_episode_steps = self.config['training']['max_episode_steps']
        grid_size = self.agent.grid_size
        
        obs_dicts, infos = self.envs.reset()
        obs_dicts = list(obs_dicts)
        
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
                opponent_memories.append(opponent.memory.clone())
            else:
                opponent_memories.append(None)
        
        for mem in self.agent_memories:
            mem.reset()
        
        episode_steps = np.zeros(n_envs, dtype=np.int32)
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        episode_results = []
        
        agent_last_actions = [None] * n_envs
        agent_prev_obs_snapshots = [None] * n_envs
        agent_prev_actions = [Action(to_pass=True)] * n_envs
        
        network_groups, random_env_indices = self._build_opponent_groups(current_opponents, opponent_memories)
        
        _pass_action = Action(to_pass=True)
        action_array_buf = np.empty(5, dtype=np.int32)
        
        for step in range(n_steps):
            observations = [obs_dicts[env_idx]["Agent"] for env_idx in range(n_envs)]
            prior_observations = observations
            
            opponent_last_actions = [
                self.agent._infer_opponent_action(agent_prev_obs_snapshots[i], observations[i])
                if agent_prev_obs_snapshots[i] is not None else _pass_action
                for i in range(n_envs)
            ]
            
            agent_actions, agent_log_probs, agent_values, obs_tensors, memory_tensors = self.agent.act_with_value_batch_fast(
                observations, 
                self.agent_memories,
                self._obs_buffer,
                self._mem_buffer,
                self.device,
                prev_obs_snapshots=agent_prev_obs_snapshots,
                last_actions=agent_last_actions,
                opponent_last_actions=opponent_last_actions
            )
            
            opponent_actions = [None] * n_envs
            
            for group in network_groups.values():
                env_indices = group['env_indices']
                if not env_indices:
                    continue
                
                opp_obs_list = [obs_dicts[env_idx]["Opponent"] for env_idx in env_indices]
                for idx, (i, env_idx) in enumerate(zip(range(len(env_indices)), env_indices)):
                    obs = opp_obs_list[i]
                    obs.pad_observation(pad_to=grid_size)
                    obs_np = obs.as_tensor().astype(np.float32)
                    mem_np = group['memories'][i].get_memory_features()
                    self._opp_obs_buffer[idx].copy_(torch.from_numpy(obs_np))
                    self._opp_mem_buffer[idx].copy_(torch.from_numpy(mem_np))
                
                group_size = len(env_indices)
                obs_batch = self._opp_obs_buffer[:group_size].to(self.device, non_blocking=True)
                mem_batch = self._opp_mem_buffer[:group_size].to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    policy_logits_batch, _ = group['network'](obs_batch, mem_batch)
                
                for i, env_idx in enumerate(env_indices):
                    opponent = group['opponents'][i]
                    obs = opp_obs_list[i]
                    mem = group['memories'][i]
                    
                    if hasattr(opponent, 'opponent_last_action'):
                        opponent.opponent_last_action = agent_prev_actions[env_idx]
                    
                    opp_action = opponent._sample_action(policy_logits_batch[i], obs)
                    
                    if opponent.last_action is not None:
                        mem.update(
                            opponent._obs_to_dict(obs),
                            np.asarray(opponent.last_action, dtype=np.int8),
                            np.asarray(opponent.opponent_last_action, dtype=np.int8),
                        )
                    
                    opponent.last_action = opp_action
                    opponent_actions[env_idx] = opp_action
            
            for env_idx in random_env_indices:
                opponent_actions[env_idx] = current_opponents[env_idx].act(obs_dicts[env_idx]["Opponent"])

            next_obs_dicts, rewards_dicts, dones, next_infos = self.envs.step(agent_actions, opponent_actions)
            next_obs_dicts = list(next_obs_dicts)

            next_agent_obs = [next_obs_dicts[i]["Agent"] for i in range(n_envs)]
            rewards = self.reward_fn.compute_batch(prior_observations, next_agent_obs)

            reset_envs = []
            
            for env_idx in range(n_envs):
                action_to_array(agent_actions[env_idx], action_array_buf)
                
                obs = observations[env_idx]
                obs.pad_observation(pad_to=grid_size)
                valid_mask = compute_valid_move_mask(obs)
                
                self.buffer.store_transition(
                    step, env_idx,
                    obs_tensors[env_idx],
                    memory_tensors[env_idx],
                    action_array_buf.copy(),
                    agent_log_probs[env_idx],
                    agent_values[env_idx],
                    rewards[env_idx],
                    float(dones[env_idx]),
                    valid_mask=valid_mask
                )
                
                episode_rewards[env_idx] += rewards[env_idx]
                episode_steps[env_idx] += 1
                
                if dones[env_idx] or episode_steps[env_idx] >= max_episode_steps:
                    if opponent_indices[env_idx] is not None:
                        agent_won = rewards_dicts[env_idx]["Agent"] > rewards_dicts[env_idx]["Opponent"]
                        episode_results.append((opponent_indices[env_idx], agent_won))
                    
                    opponent, idx = self.opponent_pool.sample_opponent()
                    if opponent is None:
                        opponent = RandomAgent()
                        idx = None
                    if hasattr(opponent, 'reset'):
                        opponent.reset()
                    current_opponents[env_idx] = opponent
                    opponent_indices[env_idx] = idx
                    opponent_memories[env_idx] = opponent.memory.clone() if hasattr(opponent, 'memory') else None
                    
                    self.agent_memories[env_idx].reset()
                    agent_last_actions[env_idx] = None
                    agent_prev_obs_snapshots[env_idx] = None
                    agent_prev_actions[env_idx] = Action(to_pass=True)
                    episode_steps[env_idx] = 0
                    episode_rewards[env_idx] = 0.0
                    reset_envs.append(env_idx)
                else:
                    agent_last_actions[env_idx] = agent_actions[env_idx]
                    agent_prev_obs_snapshots[env_idx] = self.agent._snapshot_observation(next_obs_dicts[env_idx]["Agent"])
                    agent_prev_actions[env_idx] = agent_actions[env_idx]
            
            if reset_envs:
                network_groups, random_env_indices = self._build_opponent_groups(current_opponents, opponent_memories)
            
            obs_dicts = next_obs_dicts
        
        last_observations = [obs_dicts[i]["Agent"] for i in range(n_envs)]
        for obs in last_observations:
            obs.pad_observation(pad_to=grid_size)
        
        obs_batch = torch.stack([torch.from_numpy(obs.as_tensor()).float() for obs in last_observations]).to(self.device)
        memory_batch = torch.stack([torch.from_numpy(self.agent_memories[i].get_memory_features()).float() for i in range(n_envs)]).to(self.device)
        
        with torch.no_grad():
            _, values_batch = self.agent.network(obs_batch, memory_batch)
        last_values = values_batch.squeeze(-1).cpu().numpy()
        
        last_values = last_values * (1.0 - self.buffer.dones[n_steps - 1])
        
        self.buffer.finish_trajectory(
            last_values,
            gamma=self.config['training']['gamma'],
            gae_lambda=self.config['training']['gae_lambda']
        )
        
        return episode_rewards.tolist(), episode_results
    
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
        
        self.buffer.prepare_for_training()
        self.agent.network.eval()
        
        for epoch in range(ppo_epochs):
            for batch in self.buffer.get_batches(batch_size):
                obs = batch['observations']
                mem = batch['memories']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                old_values = batch['values']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # raw_adv_mean = advantages.abs().mean().item()
                # raw_adv_max = advantages.abs().max().item()
                # print(f"[DIAGNOSIS] Advantage - Mean: {raw_adv_mean:.4f}, Max: {raw_adv_max:.4f}")
                
                advantages = torch.clamp(advantages, -2.0, 2.0)
                
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                valid_masks = batch.get('valid_masks', None)
                vm_tensor = None
                if valid_masks is not None:
                    vm_tensor = valid_masks.permute(0, 3, 1, 2)
                    vm_tensor = vm_tensor.repeat_interleave(2, dim=1)
                
                new_log_probs, new_values, entropies = self.agent.evaluate_actions(obs, mem, actions, valid_masks=vm_tensor)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # with torch.no_grad():
                #     r_max = ratio.max().item()
                #     r_min = ratio.min().item()
                #     if r_max > 1.5 or r_min < 0.5:
                #         print(f"[NUMERICAL CHECK] Ratio Shock! Max: {r_max:.4f}, Min: {r_min:.4f}")
                
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
                # loss = value_loss_coef * value_loss - entropy_coef * entropy
                # print(f"policy_loss: {policy_loss}")
                # print(f"value_loss: {value_loss_coef * value_loss}")
                # print(f"entropy_loss: {entropy_coef * entropy}")
                # loss = loss - loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.network.parameters(), max_grad_norm)
                
                # if n_updates == 0:
                #     print("\n" + "="*50)
                #     print("FIRST UPDATE GRADIENT AUDIT")
                #     print("="*50)
                    
                #     backbone_grad = 0.0
                #     policy_grad = 0.0
                #     value_grad = 0.0
                    
                #     for name, param in self.agent.network.named_parameters():
                #         if param.grad is not None:
                #             g_norm = param.grad.norm().item()
                #             p_norm = param.data.norm().item()
                            
                #             update_ratio = g_norm / (p_norm + 1e-8)
                            
                #             if 'backbone' in name:
                #                 backbone_grad += g_norm
                #             elif 'policy_head' in name:
                #                 policy_grad += g_norm
                #             elif 'value_head' in name:
                #                 value_grad += g_norm

                #             if update_ratio > 0.05:
                #                 print(f"[SHOCK ALERT] Layer: {name:30} | Grad: {g_norm:.6f} | Ratio: {update_ratio:.4f}")

                #     print("-" * 50)
                #     print(f"Total Backbone Grad Norm: {backbone_grad:.6f}")
                #     print(f"Total Policy Head Grad Norm: {policy_grad:.6f}")
                #     print(f"Total Value Head Grad Norm: {value_grad:.6f}")
                    
                #     has_nan = False
                #     for p in self.agent.network.parameters():
                #         if p.grad is not None and torch.any(p.grad.isnan()):
                #             has_nan = True
                #             break
                #     print(f"Gradient Contains NaN: {has_nan}")
                #     print("="*50 + "\n")
                
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
        
        if warmup_iterations == 0:
            checkpoint_path = self.checkpoint_dir / "pool_agent_initial.pt"
            torch.save(self.agent.network.state_dict(), checkpoint_path)
            sota_config = {
                'obs_channels': self.config['model']['obs_channels'],
                'memory_channels': self.config['model']['memory_channels'],
                'grid_size': self.config['model']['grid_size'],
                'base_channels': self.config['model']['base_channels'],
            }
            self.opponent_pool.add_opponent(
                self.agent,
                str(checkpoint_path),
                0,
                "initial_checkpoint",
                sota_config=sota_config
            )
            print(f"[Opponent Pool] Added initial agent to pool. Pool size: {len(self.opponent_pool)}\n")
        
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
                sota_config = {
                    'obs_channels': self.config['model']['obs_channels'],
                    'memory_channels': self.config['model']['memory_channels'],
                    'grid_size': self.config['model']['grid_size'],
                    'base_channels': self.config['model']['base_channels'],
                }
                self.opponent_pool.add_opponent(
                    self.agent,
                    str(checkpoint_path),
                    iteration,
                    f"checkpoint_{iteration}",
                    sota_config=sota_config
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


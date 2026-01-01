import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from torch.distributions import Categorical
from tqdm import tqdm

try:
    import swanlab
except ImportError:  # Swanlab is optional; we fall back to stdout logging
    swanlab = None

from generals.core.action import Action
from generals.core.rewards import FrequentAssetRewardFn, LandRewardFn, WinLoseRewardFn, is_action_valid, compute_num_cities_owned, compute_num_generals_owned
from generals.core.observation import Observation

from env.gymnasium_generals import GymnasiumGenerals
from model.memory import MemoryAugmentation
from model.network import SOTANetwork


@dataclass
class PPOHyperParams:
    rollout_length: int
    total_env_steps: int
    num_minibatches: int
    ppo_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    learning_rate: float
    anneal_lr: bool = False
    adaptive_lr: bool = False
    target_kl: float | None = None
    value_warmup_steps: int = 0
    
    # Collapse Detection
    collapse_win_rate_drop: float = 0.06
    collapse_tie_rate_threshold: float = 0.40
    min_win_rate_for_detection: float = 0.10
    min_episodes_for_detection: int = 50
    recovery_cooldown: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Derive the memory channel count from the augmentation to keep model/input in sync with replay data.
MEMORY_CHANNELS = MemoryAugmentation((1, 1)).get_memory_features().shape[0]


def flatten_action_index(row: int, col: int, direction: int, split: int, grid: int) -> int:
    return (row * grid + col) * 8 + direction * 2 + split


def decode_action_index(index: int, grid: int) -> Tuple[bool, int, int, int, int]:
    if index == 0:
        return True, 0, 0, 0, 0
    move_idx = index - 1
    cell = move_idx // 8
    rem = move_idx % 8
    direction = rem // 2
    split = rem % 2
    row = cell // grid
    col = cell % grid
    return False, row, col, direction, split


def build_action_distribution(
    policy_logits: torch.Tensor,
    valid_mask: np.ndarray,
    grid: int,
) -> Categorical:
    """Mask invalid moves and build a categorical distribution over flattened actions."""
    # policy_logits: (B, 9, H, W)
    device = policy_logits.device
    batch_size = policy_logits.shape[0]
    pass_logits = policy_logits[:, 0, 0, 0]  # (B,)

    move_logits = policy_logits[:, 1:9]  # (B, 8, H, W)
    move_logits = move_logits.permute(0, 2, 3, 1).reshape(batch_size, grid * grid, 8)

    # valid_mask: (B, H, W, 4) or (H, W, 4)
    valid_dirs = torch.from_numpy(valid_mask).to(device=device, dtype=torch.bool)
    
    if valid_dirs.dim() == 3: # (H, W, 4) -> unbatched
        valid_dirs = valid_dirs.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    valid_dirs = valid_dirs.reshape(batch_size, grid * grid, 4)
    valid_moves = valid_dirs.repeat_interleave(2, dim=2)  # (B, cells, 8)

    masked_logits = move_logits.masked_fill(~valid_moves, float("-inf"))
    flat_logits = torch.cat([pass_logits.unsqueeze(1), masked_logits.view(batch_size, -1)], dim=1)

    return Categorical(logits=flat_logits)


def obs_tensor_to_dict(obs: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "armies": obs[0],
        "generals": obs[1],
        "cities": obs[2],
        "mountains": obs[3],
        "neutral_cells": obs[4],
        "owned_cells": obs[5],
        "opponent_cells": obs[6],
        "fog_cells": obs[7],
        "structures_in_fog": obs[8],
        "owned_land_count": obs[9],
        "owned_army_count": obs[10],
        "opponent_land_count": obs[11],
        "opponent_army_count": obs[12],
        "timestep": obs[13],
        "priority": obs[14],
    }


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.obs: List[torch.Tensor] = []
        self.memory: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.memory.append(memory)
        self.masks.append(mask)
        self.actions.append(action)
        self.logprobs.append(logprob.view(1))
        self.values.append(value.view(1))
        # Keep rewards/dones on the same device as value to avoid cross-device cat
        device = value.device
        self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
        self.dones.append(torch.tensor([done], dtype=torch.float32, device=device))

    def compute_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.cat(self.rewards)
        dones = torch.cat(self.dones)
        last_value = last_value.view(1)
        values = torch.cat(self.values + [last_value])  # last value appended for bootstrap

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def get_batch(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.stack(self.obs),
            "memory": torch.stack(self.memory),
            "masks": torch.stack(self.masks),
            "actions": torch.stack(self.actions).squeeze(-1),
            "logprobs": torch.stack(self.logprobs).squeeze(-1),
            "values": torch.cat(self.values),
        }

class FixedFrequentAssetRewardFn(FrequentAssetRewardFn):
    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_army_size = obs.owned_army_count - prior_obs.owned_army_count
        change_in_land_owned = obs.owned_land_count - prior_obs.owned_land_count
        change_in_num_cities_owned = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        
        # Exploration reward: Reward for revealing fog
        # fog_cells is 1 for fog, 0 for visible. Decrease in sum means more visible.
        change_in_fog = np.sum(prior_obs.fog_cells) - np.sum(obs.fog_cells)
        
        # FIX: Check if action is pass
        if prior_action.is_pass():
            is_valid = True
            # Discourage passing after the starting phase (e.g., 20 steps)
            # We want to force the agent to be active.
            if obs.timestep > 20:
                valid_action_reward = -5 # Strong penalty for passing later
            else:
                valid_action_reward = -0.05 # Small penalty for time passing
        else:
            is_valid = is_action_valid(prior_action, prior_obs)
            valid_action_reward = -0.05 if is_valid else -5 # Small penalty for time passing

        reward = (
            valid_action_reward
            + 0.05 * change_in_army_size
            + 2.0 * change_in_land_owned
            + 0.5 * change_in_fog
            + 10 * change_in_num_cities_owned
            + 10_000 * change_in_num_generals_owned
        )

        return reward


from collections import deque

# ... existing imports ...

class PPORunner:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.seed = cfg.get("seed", 42)
        set_seed(self.seed)

        self.device = torch.device(cfg.get("device", "cpu"))
        env_cfg = cfg.get("env", {})
        model_cfg = cfg.get("model", {})
        ppo_cfg = cfg.get("ppo", {})
        log_cfg = cfg.get("logging", {})

        self.agents: List[str] = env_cfg.get("agents", ["player_0", "player_1"])
        self.grid_size: int = env_cfg.get("pad_observations_to", 24)
        self.truncation: int | None = env_cfg.get("truncation")

        reward_fn = self._make_reward_fn(env_cfg.get("reward_fn", "frequent_asset"))

        self.num_workers = ppo_cfg.get("num_workers", 1)
        
        def make_env():
            return GymnasiumGenerals(
                agents=self.agents,
                pad_observations_to=self.grid_size,
                truncation=self.truncation,
                reward_fn=reward_fn,
                render_mode=env_cfg.get("render_mode"),
            )

        if self.num_workers > 1:
            self.env = gym.vector.AsyncVectorEnv([make_env for _ in range(self.num_workers)])
        else:
            self.env = gym.vector.SyncVectorEnv([make_env])

        self.model = SOTANetwork(
            obs_channels=model_cfg.get("obs_channels", 15),
            memory_channels=model_cfg.get("memory_channels", MEMORY_CHANNELS),
            grid_size=model_cfg.get("grid_size", self.grid_size),
            base_channels=model_cfg.get("base_channels", 64),
        ).to(self.device)

        bc_path = model_cfg.get("bc_checkpoint")
        if bc_path:
            self._load_checkpoint_weights(bc_path)

        self.hyper = PPOHyperParams(
            rollout_length=ppo_cfg.get("rollout_length", 256),
            total_env_steps=ppo_cfg.get("total_env_steps", 200000),
            num_minibatches=ppo_cfg.get("num_minibatches", 4),
            ppo_epochs=ppo_cfg.get("ppo_epochs", 4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            value_coef=ppo_cfg.get("value_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            anneal_lr=ppo_cfg.get("anneal_lr", False),
            adaptive_lr=ppo_cfg.get("adaptive_lr", False),
            target_kl=ppo_cfg.get("target_kl", None),
            value_warmup_steps=ppo_cfg.get("value_warmup_steps", 0),
            collapse_win_rate_drop=ppo_cfg.get("collapse_win_rate_drop", 0.06),
            collapse_tie_rate_threshold=ppo_cfg.get("collapse_tie_rate_threshold", 0.40),
            min_win_rate_for_detection=ppo_cfg.get("min_win_rate_for_detection", 0.10),
            min_episodes_for_detection=ppo_cfg.get("min_episodes_for_detection", 50),
            recovery_cooldown=ppo_cfg.get("recovery_cooldown", 50),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper.learning_rate)
        self.buffers = [RolloutBuffer() for _ in range(self.num_workers)]

        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints_ppo"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = log_cfg.get("checkpoint_every", 10000)

        self.use_swanlab = bool(log_cfg.get("use_swanlab", False)) and swanlab is not None
        if self.use_swanlab:
            swanlab.init(
                project=log_cfg.get("project", "generals-ppo"),
                experiment_name=log_cfg.get("experiment_name", "ppo-bc"),
                config=cfg,
            )
        elif log_cfg.get("use_swanlab", False):
            print("Warning: swanlab requested but not installed; proceeding without it.")

        self.log_interval = log_cfg.get("log_interval", 10)
        self.reward_scale = ppo_cfg.get("reward_scale", 0.001) # Scale rewards to avoid exploding value gradients

        self.memories: List[Dict[str, MemoryAugmentation]] = []
        for _ in range(self.num_workers):
            self.memories.append({
                agent: MemoryAugmentation((self.grid_size, self.grid_size)) for agent in self.agents
            })
            
        # Episode tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.episode_ties = deque(maxlen=100)
        self.episode_losses = deque(maxlen=100)
        self.episode_reward_per_step = deque(maxlen=100)
        self.current_episode_return = np.zeros(self.num_workers, dtype=np.float32)
        self.current_episode_length = np.zeros(self.num_workers, dtype=np.int32)
        
        self.recovery_cooldown = 0 # Steps to wait before allowing LR increase after recovery

    def _make_reward_fn(self, name: str):
        name = (name or "").lower()
        if name in {"frequent", "frequent_asset", "asset"}:
            return FixedFrequentAssetRewardFn()
        if name in {"land"}:
            return LandRewardFn()
        if name in {"winlose", "win_lose", "wl", "default"}:
            return WinLoseRewardFn()
        print(f"Unknown reward_fn '{name}', falling back to FixedFrequentAssetRewardFn")
        return FixedFrequentAssetRewardFn()

    def _load_checkpoint_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model_state = self.model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}

        missing = [k for k in model_state.keys() if k not in filtered]
        skipped = [k for k in state.keys() if k not in filtered]

        self.model.load_state_dict(filtered, strict=False)
        if len(filtered) == len(model_state):
            print(f"Loaded BC weights from {path}: all {len(filtered)} tensors loaded.")
        else:
            print(f"Loaded BC weights from {path} with partial match: {len(filtered)}/{len(model_state)} tensors loaded.")
        
        # Re-initialize value head to avoid destroying backbone with huge gradients due to scale mismatch
        # The BC value head is likely trained on a different reward scale (e.g. Win/Loss 0-1)
        # while our RL reward is much larger (e.g. >100).
        print("Re-initializing value head to prevent backbone collapse due to reward scale mismatch.")
        self.model.value_head = type(self.model.value_head)(
            self.model.value_head.conv[0].conv.in_channels, 
            self.grid_size
        ).to(self.device)
        
        if missing:
            print(f"Warning: {len(missing)} parameters missing in checkpoint (kept model init): {missing[:5]}...")
        if skipped:
            print(f"Warning: {len(skipped)} parameters in checkpoint did not match model shape: {skipped[:5]}...")

    def _prepare_tensors(self, obs_np: np.ndarray, memory_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        mem_t = torch.from_numpy(memory_np).float().unsqueeze(0).to(self.device)
        return obs_t, mem_t

    def _action_for_agent(
        self,
        obs_np: np.ndarray,
        memory_np: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[Action, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t, mem_t = self._prepare_tensors(obs_np, memory_np)
        policy_logits, value = self.model(obs_t, mem_t)
        dist = build_action_distribution(policy_logits, valid_mask, self.grid_size)
        action_idx = dist.sample()
        logprob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        to_pass, row, col, direction, split = decode_action_index(action_idx.item(), self.grid_size)
        act = Action(to_pass=to_pass, row=row, col=col, direction=direction, to_split=bool(split))
        action_array = np.array([int(to_pass), row, col, direction, split], dtype=np.int8)
        return act, action_array, action_idx, logprob, value.squeeze(-1)

    def _reset_memories(self) -> None:
        for env_memories in self.memories:
            for mem in env_memories.values():
                mem.reset()

    def collect_rollout(self, start_obs: np.ndarray, start_infos: Tuple[Dict[str, Any]], global_step: int) -> Tuple[np.ndarray, Tuple[Dict[str, Any]], int]:
        self.model.eval()
        obs = start_obs
        infos = start_infos
        
        for buf in self.buffers:
            buf.clear()

        steps_per_worker = self.hyper.rollout_length // self.num_workers
        
        for _ in range(steps_per_worker):
            worker_actions = [[] for _ in range(self.num_workers)]
            
            # Temporary storage for player_0 data
            p0_obs = None
            p0_mem = None
            p0_mask = None
            p0_action = None
            p0_logprob = None
            p0_value = None
            
            for agent_idx, agent_name in enumerate(self.agents):
                # Gather data
                agent_obs_np = obs[:, agent_idx] # (B, C, H, W)
                
                agent_memories_np = []
                for w in range(self.num_workers):
                    agent_memories_np.append(self.memories[w][agent_name].get_memory_features())
                agent_memories_np = np.stack(agent_memories_np) # (B, C_mem, H, W)
                
                # Fix: Access batched infos correctly
                # infos is {agent: {key: batched_array}}
                agent_masks_np = infos[agent_name]["masks"] # (B, H, W, 4)
                
                # Inference
                obs_t = torch.from_numpy(agent_obs_np).float().to(self.device)
                mem_t = torch.from_numpy(agent_memories_np).float().to(self.device)
                
                with torch.no_grad():
                    policy_logits, value = self.model(obs_t, mem_t)
                    dist = build_action_distribution(policy_logits, agent_masks_np, self.grid_size)
                    action_idx = dist.sample()
                    logprob = dist.log_prob(action_idx)
                
                # Store actions for step
                for w in range(self.num_workers):
                    idx_val = action_idx[w].item()
                    to_pass, row, col, direction, split = decode_action_index(idx_val, self.grid_size)
                    act_arr = np.array([int(to_pass), row, col, direction, split], dtype=np.int8)
                    worker_actions[w].append(act_arr)
                
                # Store data for player_0
                if agent_name == self.agents[0]:
                    p0_obs = obs_t
                    p0_mem = mem_t
                    p0_mask = torch.from_numpy(agent_masks_np).to(self.device)
                    p0_action = action_idx
                    p0_logprob = logprob
                    p0_value = value.squeeze(-1)

            # Step environment
            step_actions = np.stack([np.stack(wa) for wa in worker_actions]) # (B, num_agents, 5)
            next_obs, _, terminateds, truncateds, next_infos = self.env.step(step_actions)
            
            # Process results and store in buffers
            for w in range(self.num_workers):
                # Handle done state and extract correct info
                is_done = terminateds[w] or truncateds[w]
                
                # Default values from current step (valid if not done, or if done but no final_info captured yet)
                raw_reward = float(next_infos[self.agents[0]]["reward"][w])
                done_val = bool(next_infos[self.agents[0]]["done"][w])
                is_winner = bool(next_infos[self.agents[0]]["winner"][w])

                if is_done and "final_info" in next_infos:
                    # Extract from final_info for the done step
                    info_w = next_infos["final_info"][w]
                    if info_w is not None:
                        raw_reward = float(info_w[self.agents[0]]["reward"])
                        done_val = bool(info_w[self.agents[0]]["done"])
                        is_winner = bool(info_w[self.agents[0]]["winner"])

                # Explicit Win/Loss Reward
                # This dominates other rewards to ensure winning is the priority
                if done_val:
                    if is_winner:
                        raw_reward += 20000
                    else:
                        raw_reward -= 20000
                elif truncateds[w]:
                    # Penalty for truncation (tie) to force agent to try to win
                    # Not as bad as losing (-20000), but bad enough to discourage stalling
                    raw_reward -= 5000

                reward = raw_reward * self.reward_scale
                done_flag = bool(is_done or done_val)
                
                self.current_episode_return[w] += raw_reward
                self.current_episode_length[w] += 1
                
                self.buffers[w].add(
                    obs=p0_obs[w],
                    memory=p0_mem[w],
                    mask=p0_mask[w],
                    action=p0_action[w],
                    logprob=p0_logprob[w],
                    value=p0_value[w],
                    reward=reward,
                    done=done_flag
                )
                
                # Handle reset and memory update
                if done_flag:
                    self.episode_returns.append(self.current_episode_return[w])
                    self.episode_lengths.append(self.current_episode_length[w])
                    
                    if is_winner:
                        self.episode_wins.append(1)
                        self.episode_ties.append(0)
                        self.episode_losses.append(0)
                    elif truncateds[w]:
                        self.episode_wins.append(0)
                        self.episode_ties.append(1)
                        self.episode_losses.append(0)
                    else:
                        self.episode_wins.append(0)
                        self.episode_ties.append(0)
                        self.episode_losses.append(1)
                    
                    if self.current_episode_length[w] > 0:
                        self.episode_reward_per_step.append(self.current_episode_return[w] / self.current_episode_length[w])
                    
                    self.current_episode_return[w] = 0
                    self.current_episode_length[w] = 0
                    
                    for mem in self.memories[w].values():
                        mem.reset()
                else:
                    for agent_idx, agent_name in enumerate(self.agents):
                        # next_obs is already the new observation (batched)
                        obs_dict = obs_tensor_to_dict(next_obs[w, agent_idx])
                        self.memories[w][agent_name].update(obs_dict, worker_actions[w][agent_idx])
            
            obs = next_obs
            infos = next_infos
            global_step += self.num_workers
            
        return obs, infos, global_step

    def _compute_last_value(self, obs: np.ndarray, infos: Tuple[Dict[str, Any]]) -> torch.Tensor:
        self.model.eval()
        agent_idx = 0
        agent_name = self.agents[0]
        
        obs_agent = obs[:, agent_idx]
        
        mem_agent = []
        for w in range(self.num_workers):
            mem_agent.append(self.memories[w][agent_name].get_memory_features())
        mem_agent = np.stack(mem_agent)
        
        obs_t = torch.from_numpy(obs_agent).float().to(self.device)
        mem_t = torch.from_numpy(mem_agent).float().to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(obs_t, mem_t)
        return value.detach().squeeze(-1)

    def update(self, batch: Dict[str, torch.Tensor], advantages: torch.Tensor, returns: torch.Tensor, warmup: bool = False) -> Dict[str, float]:
        # Keep model in eval mode to ensure consistency with rollout (BatchNorm/Dropout)
        self.model.eval()
        batch_size = batch["obs"].shape[0]
        minibatch_size = batch_size // self.hyper.num_minibatches

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_rate": 0.0, "ratio_mean": 0.0}
        idxs = np.arange(batch_size)
        
        # If warmup, freeze backbone and policy head
        if warmup:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.policy_head.parameters():
                param.requires_grad = False
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            for param in self.model.policy_head.parameters():
                param.requires_grad = True

        for _ in range(self.hyper.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]

                obs_mb = batch["obs"][mb_idx].to(self.device)
                mem_mb = batch["memory"][mb_idx].to(self.device)
                masks_mb = batch["masks"][mb_idx].to(self.device)
                actions_mb = batch["actions"][mb_idx].to(self.device).long() # Ensure actions are long
                old_logprobs_mb = batch["logprobs"][mb_idx].to(self.device)
                returns_mb = returns[mb_idx].to(self.device)
                advantages_mb = advantages[mb_idx].to(self.device)

                policy_logits, values = self.model(obs_mb, mem_mb)
                
                # Rebuild distribution with proper masking
                B = policy_logits.shape[0]
                grid = self.grid_size
                
                # policy_logits: (B, 9, H, W)
                pass_logits = policy_logits[:, 0, 0, 0]  # (B,)
                move_logits = policy_logits[:, 1:9]  # (B, 8, H, W)
                move_logits = move_logits.permute(0, 2, 3, 1).reshape(B, grid * grid, 8)

                # masks_mb: (B, H, W, 4) -> flatten and repeat for split dimension
                # Note: masks_mb is already (B, H, W, 4) from buffer
                valid_dirs = masks_mb.reshape(B, grid * grid, 4)
                valid_moves = valid_dirs.repeat_interleave(2, dim=2)  # (B, cells, 8)

                masked_logits = move_logits.masked_fill(~valid_moves, float("-inf"))
                flat_logits = torch.cat([pass_logits.unsqueeze(1), masked_logits.view(B, -1)], dim=1)
                
                dist = Categorical(logits=flat_logits)
                new_logprobs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_mb)
                pg_loss1 = -advantages_mb * ratio
                pg_loss2 = -advantages_mb * torch.clamp(ratio, 1 - self.hyper.clip_range, 1 + self.hyper.clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss clipping
                values = values.view(-1)
                # old_values_mb = batch["values"][mb_idx].to(self.device) # We need old values for clipping
                # But buffer stores 'values' which are the old values.
                # Wait, batch["values"] IS the old value prediction from rollout.
                old_values_mb = batch["values"][mb_idx].to(self.device)
                
                v_loss_unclipped = (values - returns_mb) ** 2
                v_clipped = old_values_mb + torch.clamp(
                    values - old_values_mb,
                    -self.hyper.clip_range,
                    self.hyper.clip_range,
                )
                v_loss_clipped = (v_clipped - returns_mb) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                if warmup:
                    loss = self.hyper.value_coef * value_loss
                else:
                    loss = policy_loss + self.hyper.value_coef * value_loss - self.hyper.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logprobs_mb - new_logprobs).mean().item()
                    clipped = (ratio < 1 - self.hyper.clip_range) | (ratio > 1 + self.hyper.clip_range)
                    clip_rate = clipped.float().mean().item()
                    ratio_mean = ratio.mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_rate"] += clip_rate
                metrics["ratio_mean"] += ratio_mean

                if self.hyper.target_kl is not None and approx_kl > self.hyper.target_kl * 1.5:
                    break
            else:
                continue
            break

        num_updates = self.hyper.ppo_epochs * self.hyper.num_minibatches
        for k in metrics:
            metrics[k] /= num_updates
        return metrics

    def save_checkpoint(self, step: int, is_best: bool = False) -> None:
        if is_best:
            ckpt_path = self.checkpoint_dir / "best_model.pt"
        else:
            ckpt_path = self.checkpoint_dir / f"ppo_step_{step}.pt"
        torch.save({"step": step, "model_state_dict": self.model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    def load_best_checkpoint(self) -> int:
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        if not ckpt_path.exists():
            print("No best model found to rollback to.")
            return 0
        
        print(f"Rolling back to best model: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        return state["step"]

    def train(self) -> None:
        obs, infos = self.env.reset()
        global_step = 0
        self._reset_memories()
        
        best_win_rate = -1.0
        steps_since_best = 0

        progress = tqdm(total=self.hyper.total_env_steps, desc="PPO Training", unit="step")

        while global_step < self.hyper.total_env_steps:
            if self.hyper.anneal_lr:
                frac = 1.0 - (global_step / self.hyper.total_env_steps)
                lrnow = self.hyper.learning_rate * frac
                self.optimizer.param_groups[0]["lr"] = lrnow

            obs, infos, global_step = self.collect_rollout(obs, infos, global_step)
            
            last_values = self._compute_last_value(obs, infos)
            
            all_advantages = []
            all_returns = []
            
            for w in range(self.num_workers):
                last_val = last_values[w]
                adv, ret = self.buffers[w].compute_advantages(last_val, self.hyper.gamma, self.hyper.gae_lambda)
                all_advantages.append(adv)
                all_returns.append(ret)
            
            flat_advantages = torch.cat(all_advantages)
            flat_returns = torch.cat(all_returns)
            
            # Normalize advantages
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            
            # Merge buffers
            merged_batch = {
                "obs": [], "memory": [], "masks": [], "actions": [], "logprobs": [], "values": []
            }
            
            for w in range(self.num_workers):
                b = self.buffers[w].get_batch()
                for k in merged_batch:
                    merged_batch[k].append(b[k])
            
            for k in merged_batch:
                merged_batch[k] = torch.cat(merged_batch[k])

            avg_reward = torch.cat([torch.cat(b.rewards) for b in self.buffers]).mean().item()
            raw_reward = avg_reward / self.reward_scale
            
            episodic_return_mean = np.mean(self.episode_returns) if len(self.episode_returns) > 0 else 0.0
            episodic_length_mean = np.mean(self.episode_lengths) if len(self.episode_lengths) > 0 else 0.0
            episodic_win_rate = np.mean(self.episode_wins) if len(self.episode_wins) > 0 else 0.0
            episodic_tie_rate = np.mean(self.episode_ties) if len(self.episode_ties) > 0 else 0.0
            episodic_loss_rate = np.mean(self.episode_losses) if len(self.episode_losses) > 0 else 0.0
            episodic_reward_per_step_mean = np.mean(self.episode_reward_per_step) if len(self.episode_reward_per_step) > 0 else 0.0
            
            # Calculate pass rate
            pass_rate = (merged_batch["actions"] == 0).float().mean().item()
            
            # Determine if we are in warmup phase
            is_warmup = global_step < self.hyper.value_warmup_steps
            if is_warmup:
                # During warmup, we only train value head.
                # We don't want adaptive LR to mess with LR based on policy KL (which is 0 or meaningless)
                # So we skip adaptive LR logic or handle it differently.
                pass

            metrics = self.update(merged_batch, flat_advantages, flat_returns, warmup=is_warmup)

            # Adaptive Learning Rate
            if not is_warmup and self.hyper.adaptive_lr and self.hyper.target_kl is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                # If recovering, only allow LR decrease, not increase
                if self.recovery_cooldown > 0:
                    self.recovery_cooldown -= 1
                    if metrics["approx_kl"] > self.hyper.target_kl * 2.0:
                        current_lr = max(1e-6, current_lr / 1.5)
                    # No increase allowed
                else:
                    if metrics["approx_kl"] > self.hyper.target_kl * 2.0:
                        current_lr = max(1e-6, current_lr / 1.5)
                    elif metrics["approx_kl"] < self.hyper.target_kl / 2.0:
                        current_lr = min(1e-3, current_lr * 1.5)
                
                self.optimizer.param_groups[0]["lr"] = current_lr

            if self.use_swanlab:
                log_dict = {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": metrics["entropy"],
                    "train/approx_kl": metrics["approx_kl"],
                    "train/clip_rate": metrics["clip_rate"],
                    "train/ratio_mean": metrics["ratio_mean"],
                    "train/reward_mean": avg_reward,
                    "train/raw_reward_mean": raw_reward,
                    "train/pass_rate": pass_rate,
                    "train/episodic_return_mean": episodic_return_mean,
                    "train/episodic_length_mean": episodic_length_mean,
                    "train/episodic_win_rate": episodic_win_rate,
                    "train/episodic_tie_rate": episodic_tie_rate,
                    "train/episodic_loss_rate": episodic_loss_rate,
                    "train/episodic_reward_per_step_mean": episodic_reward_per_step_mean,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/step": global_step,
                    "train/is_warmup": float(is_warmup),
                }
                swanlab.log(log_dict)

            # Save Best Model
            if len(self.episode_wins) >= self.hyper.min_episodes_for_detection: # Wait for some stats
                if episodic_win_rate > best_win_rate:
                    best_win_rate = episodic_win_rate
                    self.save_checkpoint(global_step, is_best=True)
                    steps_since_best = 0
                else:
                    steps_since_best += self.hyper.rollout_length

            # Collapse Detection & Recovery
            # Condition: Win rate dropped significantly AND we had a decent win rate before
            # OR Tie rate is extremely high (stalling)
            is_collapse = False
            if best_win_rate > self.hyper.min_win_rate_for_detection and len(self.episode_wins) >= self.hyper.min_episodes_for_detection:
                # User requested sensitive detection: > 10% absolute drop is alarming
                if episodic_win_rate < best_win_rate - self.hyper.collapse_win_rate_drop:
                    print(f"Collapse Detected: Win rate dropped from {best_win_rate:.2f} to {episodic_win_rate:.2f}")
                    is_collapse = True
                # User requested sensitive tie detection: > 40% is alarming
                elif episodic_tie_rate > self.hyper.collapse_tie_rate_threshold:
                    print(f"Collapse Detected: Tie rate {episodic_tie_rate:.2f} is too high")
                    is_collapse = True
            
            if is_collapse:
                print("Initiating Recovery...")
                # Load best model
                step_loaded = self.load_best_checkpoint()
                if step_loaded > 0:
                    # Reduce LR
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    new_lr = current_lr * 0.5
                    self.optimizer.param_groups[0]["lr"] = new_lr
                    print(f"Reduced Learning Rate to {new_lr}")
                    
                    # Set cooldown to prevent adaptive LR from increasing it immediately
                    self.recovery_cooldown = self.hyper.recovery_cooldown
                    
                    # Reset buffers/stats
                    self._reset_memories()
                    self.episode_returns.clear()
                    self.episode_lengths.clear()
                    self.episode_wins.clear()
                    self.episode_ties.clear()
                    self.episode_losses.clear()
                    self.episode_reward_per_step.clear()
                    
                    # Reset environment to clear bad states
                    obs, infos = self.env.reset()
                    
                    # Note: We do NOT reset global_step, we continue counting
                else:
                    print("Recovery failed: No best model found.")

            if global_step % self.checkpoint_every == 0:
                self.save_checkpoint(global_step)

            if global_step % (self.log_interval * self.hyper.rollout_length) == 0:
                progress.set_postfix({
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "ep_ret": episodic_return_mean,
                })
            progress.update(self.hyper.rollout_length)

        progress.close()
        if self.use_swanlab:
            swanlab.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ppo.yaml", help="Path to PPO config file")
    args = parser.parse_args()

    runner = PPORunner(args.config)
    runner.train()


if __name__ == "__main__":
    main()

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
            # Discourage passing after the starting phase (e.g., 50 steps)
            if obs.timestep > 50:
                valid_action_reward = -2
            else:
                valid_action_reward = 1
        else:
            is_valid = is_action_valid(prior_action, prior_obs)
            valid_action_reward = 1 if is_valid else -5

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
                workspace="coc",
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
        
        # BC model for evaluation
        self.bc_model = None
        eval_vs_bc_ratio = ppo_cfg.get("eval_vs_bc_ratio", 0.25)
        if bc_path:
            self.bc_model = SOTANetwork(
                obs_channels=model_cfg.get("obs_channels", 15),
                memory_channels=model_cfg.get("memory_channels", MEMORY_CHANNELS),
                grid_size=model_cfg.get("grid_size", self.grid_size),
                base_channels=model_cfg.get("base_channels", 64),
            ).to(self.device)
            self.bc_model.load_state_dict(self.model.state_dict())
            for param in self.bc_model.parameters():
                param.requires_grad = False
            self.bc_model.eval()
            print(f"BC model loaded for evaluation. {int(eval_vs_bc_ratio * 100)}% of workers will play against BC.")
        
        self.episode_wins_vs_bc = deque(maxlen=100)
        self.episode_wins_vs_pool = deque(maxlen=100)
        self.episode_ties_vs_bc = deque(maxlen=100)
        self.eval_vs_bc_ratio = eval_vs_bc_ratio
        
        self.warmup_games = ppo_cfg.get("warmup_games", 50)
        
        self.pool_size = ppo_cfg.get("pool_size", 3)
        self.tournament_interval = ppo_cfg.get("tournament_interval", 100000)
        self.tournament_threshold = ppo_cfg.get("tournament_threshold", 0.55)
        self.tournament_games = ppo_cfg.get("tournament_games", 40)
        self.last_eval_step = 0
        
        self.opponent_pool_paths = []
        self.opp_models = None
        self.worker_tasks = None
        self.worker_current_opp_idx = None
        
        if bc_path:
            init_path = self.checkpoint_dir / "pool_init_bc.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, init_path)
            self.opponent_pool_paths = [str(init_path)] * self.pool_size
            
            self.opp_models = torch.nn.ModuleList([
                SOTANetwork(
                    obs_channels=model_cfg.get("obs_channels", 15),
                    memory_channels=model_cfg.get("memory_channels", MEMORY_CHANNELS),
                    grid_size=model_cfg.get("grid_size", self.grid_size),
                    base_channels=model_cfg.get("base_channels", 64),
                ).to(self.device) for _ in range(self.pool_size)
            ])
            for opp_model in self.opp_models:
                opp_model.eval()
            
            for i, opp_model in enumerate(self.opp_models):
                opp_state = torch.load(self.opponent_pool_paths[i], map_location=self.device, weights_only=False)
                if isinstance(opp_state, dict) and "model_state_dict" in opp_state:
                    opp_state = opp_state["model_state_dict"]
                opp_model.load_state_dict(opp_state, strict=False)
            
            print(f"Opponent pool initialized with {self.pool_size} BC models (loaded in memory).")
            
            num_self_play = max(1, int(self.num_workers * 0.5))
            num_vs_pool = max(1, int(self.num_workers * 0.25))
            num_vs_bc_base = self.num_workers - num_self_play - num_vs_pool
            self.worker_tasks = [0] * num_self_play + [1] * num_vs_pool + [2] * num_vs_bc_base
            self.worker_current_opp_idx = [random.randint(0, self.pool_size - 1) for _ in range(self.num_workers)]
            print(f"Worker tasks: {num_self_play} self-play, {num_vs_pool} vs pool, {num_vs_bc_base} vs BC")
        else:
            self.worker_tasks = [0] * self.num_workers
            self.worker_current_opp_idx = [0] * self.num_workers

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

    def collect_rollout(self, start_obs: np.ndarray, start_infos: Tuple[Dict[str, Any]], global_step: int, eval_mode: bool = False) -> Tuple[np.ndarray, Tuple[Dict[str, Any]], int, List[int]]:
        self.model.eval()
        if self.opp_models is not None:
            for opp_model in self.opp_models:
                opp_model.eval()
        obs = start_obs
        infos = start_infos
        
        for buf in self.buffers:
            buf.clear()

        finished_game_results = []
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
                    
                    if agent_name == self.agents[1] and self.worker_tasks is not None:
                        for w in range(self.num_workers):
                            task = self.worker_tasks[w]
                            if task == 1:
                                if self.opp_models is not None and len(self.opp_models) > 0:
                                    opp_idx = self.worker_current_opp_idx[w]
                                    opp_logits, _ = self.opp_models[opp_idx](obs_t[w:w+1], mem_t[w:w+1])
                                    policy_logits[w] = opp_logits[0]
                            elif task == 2 and self.bc_model is not None:
                                bc_logits, _ = self.bc_model(obs_t[w:w+1], mem_t[w:w+1])
                                policy_logits[w] = bc_logits[0]
                    
                    dist = build_action_distribution(policy_logits, agent_masks_np, self.grid_size)
                    if eval_mode:
                        action_idx = torch.argmax(dist.probs, dim=-1)
                        logprob = dist.log_prob(action_idx)
                    else:
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
                    
                    strict_win_score = 1.0 if is_winner else 0.0
                    
                    if self.worker_tasks is not None and self.worker_tasks[w] == 2:
                        self.episode_wins_vs_bc.append(strict_win_score)
                        self.episode_ties_vs_bc.append(1.0 if truncateds[w] else 0.0)
                    
                    if self.worker_tasks is not None and self.worker_tasks[w] == 1:
                        self.episode_wins_vs_pool.append(strict_win_score)
                    
                    if (self.worker_tasks is not None and self.worker_tasks[w] == 1) or eval_mode:
                        if eval_mode:
                            tourney_score = 1.0 if is_winner else (0.5 if truncateds[w] else 0.0)
                        else:
                            tourney_score = strict_win_score
                        finished_game_results.append(tourney_score)
                    
                    if self.worker_tasks is not None and self.worker_tasks[w] == 1 and self.opp_models is not None:
                        self.worker_current_opp_idx[w] = random.randint(0, len(self.opp_models) - 1)
                    
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
            
        return obs, infos, global_step, finished_game_results

    def run_tournament(self, global_step: int) -> bool:
        """运行锦标赛评估：当前模型 vs 对手池"""
        print("run_tournament")
        if len(self.opponent_pool_paths) == 0:
            return False
        
        print(f"\n[Tournament] Testing candidate at step {global_step}...")
        
        old_tasks = self.worker_tasks.copy()
        self.worker_tasks = [1] * self.num_workers
        
        tourney_results = []
        obs, infos = self.env.reset()
        self._reset_memories()

        pbar = tqdm(total=self.tournament_games, desc="Tournament Games", unit="game")
        while len(tourney_results) < self.tournament_games:
            obs, infos, _, step_results = self.collect_rollout(obs, infos, 0, eval_mode=True)
            tourney_results.extend(step_results)
            pbar.update(len(step_results))
        pbar.close()

        avg_win_rate = np.mean(tourney_results[:self.tournament_games]) if len(tourney_results) > 0 else 0.0
        print(f"[Tournament] Candidate win-rate: {avg_win_rate:.2%}")

        self.worker_tasks = old_tasks
        obs, infos = self.env.reset()
        self._reset_memories()
        
        for buf in self.buffers:
            buf.clear()

        if avg_win_rate >= self.tournament_threshold:
            print(f">>> SUCCESS! Candidate ({avg_win_rate:.2%}) replaces the oldest pool model.")
            new_ckpt_path = self.checkpoint_dir / f"pool_v_{global_step}.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, new_ckpt_path)
            
            self.opponent_pool_paths.pop(0)
            self.opponent_pool_paths.append(str(new_ckpt_path))
            
            for i in range(self.pool_size):
                state = torch.load(self.opponent_pool_paths[i], map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.opp_models[i].load_state_dict(state, strict=False)
            
            print(f"Pool updated. Current pool size: {len(self.opponent_pool_paths)}")
            return True
        else:
            print(f">>> FAILED. Win rate {avg_win_rate:.2%} < threshold {self.tournament_threshold:.2%}")
            return False

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

    def update(self, batch: Dict[str, torch.Tensor], advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        # Keep model in eval mode to ensure consistency with rollout (BatchNorm/Dropout)
        self.model.eval()
        batch_size = batch["obs"].shape[0]
        minibatch_size = batch_size // self.hyper.num_minibatches

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_rate": 0.0, "ratio_mean": 0.0}
        idxs = np.arange(batch_size)

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

        num_updates = self.hyper.ppo_epochs * self.hyper.num_minibatches
        for k in metrics:
            metrics[k] /= num_updates
        return metrics

    def save_checkpoint(self, step: int) -> None:
        ckpt_path = self.checkpoint_dir / f"ppo_step_{step}.pt"
        torch.save({"step": step, "model_state_dict": self.model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    def train(self) -> None:
        obs, infos = self.env.reset()
        global_step = 0
        self._reset_memories()

        if self.bc_model is not None and self.warmup_games > 0:
            print(f"Starting warmup: Collecting {self.warmup_games} games vs BC baseline...")
            
            original_worker_tasks = self.worker_tasks.copy() if self.worker_tasks is not None else None
            
            self.worker_tasks = [2] * self.num_workers
            
            warmup_pbar = tqdm(total=self.warmup_games, desc="Warmup (PPO vs BC)", unit="game")
            last_completed_games = 0
            
            while len(self.episode_wins_vs_bc) < self.warmup_games:
                obs, infos, _, _ = self.collect_rollout(obs, infos, 0)
                
                current_completed = len(self.episode_wins_vs_bc)
                warmup_pbar.update(current_completed - last_completed_games)
                last_completed_games = current_completed
            
            warmup_pbar.close()
            
            initial_wr = np.mean(self.episode_wins_vs_bc) if len(self.episode_wins_vs_bc) > 0 else 0.5
            initial_ep_win_rate = np.mean(self.episode_wins) if len(self.episode_wins) > 0 else 0.5
            
            if self.use_swanlab:
                swanlab.log({
                    "train/win_rate_to_bc": initial_wr,
                    "train/step": 0,
                    "train/episodic_win_rate": initial_ep_win_rate,
                })
            
            print(f"Warmup finished. Initial Win Rate vs BC: {initial_wr:.4f}")
            
            if original_worker_tasks is not None:
                self.worker_tasks = original_worker_tasks
            for buf in self.buffers:
                buf.clear()
            
            obs, infos = self.env.reset()
            self._reset_memories()

        progress = tqdm(total=self.hyper.total_env_steps, desc="PPO Training", unit="step")

        while global_step < self.hyper.total_env_steps:
            if self.hyper.anneal_lr:
                frac = 1.0 - (global_step - 1.0) / self.hyper.total_env_steps
                lrnow = self.hyper.learning_rate * frac
                self.optimizer.param_groups[0]["lr"] = lrnow

            obs, infos, global_step, _ = self.collect_rollout(obs, infos, global_step)
            
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
            episodic_win_rate_vs_bc = np.mean(self.episode_wins_vs_bc) if len(self.episode_wins_vs_bc) > 0 else 0.0
            episodic_win_rate_vs_pool = np.mean(self.episode_wins_vs_pool) if len(self.episode_wins_vs_pool) > 0 else 0.0

            metrics = self.update(merged_batch, flat_advantages, flat_returns)

            if global_step > 0 and (global_step - self.last_eval_step) >= self.tournament_interval and self.opp_models is not None:
                pool_updated = self.run_tournament(global_step)
                self.last_eval_step = global_step
                if self.use_swanlab:
                    swanlab.log({
                        "train/pool_updated": 1 if pool_updated else 0,
                        "train/step": global_step,
                    })

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
                    "train/episodic_return_mean": episodic_return_mean,
                    "train/episodic_length_mean": episodic_length_mean,
                    "train/episodic_win_rate": episodic_win_rate,
                    "train/episodic_tie_rate": episodic_tie_rate,
                    "train/episodic_loss_rate": episodic_loss_rate,
                    "train/episodic_reward_per_step_mean": episodic_reward_per_step_mean,
                    "train/step": global_step,
                }
                if self.bc_model is not None:
                    log_dict["train/win_rate_to_bc"] = episodic_win_rate_vs_bc
                    log_dict["train/tie_rate_to_bc"] = np.mean(self.episode_ties_vs_bc) if self.episode_ties_vs_bc else 0.0
                if self.opp_models is not None:
                    log_dict["train/win_rate_pool"] = episodic_win_rate_vs_pool
                swanlab.log(log_dict)

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

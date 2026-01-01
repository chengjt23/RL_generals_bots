import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical
from tqdm import tqdm

try:
    import swanlab
except ImportError:  # Swanlab is optional; we fall back to stdout logging
    swanlab = None

from generals.core.action import Action
from generals.core.rewards import FrequentAssetRewardFn, LandRewardFn, WinLoseRewardFn

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

    # valid_mask: (H, W, 4) -> flatten and repeat for split dimension
    valid_dirs = torch.from_numpy(valid_mask).to(device=device, dtype=torch.bool).reshape(grid * grid, 4)
    valid_moves = valid_dirs.repeat_interleave(2, dim=1)  # (cells, 8)

    masked_logits = move_logits.masked_fill(~valid_moves.unsqueeze(0), float("-inf"))
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

        self.env = GymnasiumGenerals(
            agents=self.agents,
            pad_observations_to=self.grid_size,
            truncation=self.truncation,
            reward_fn=reward_fn,
            render_mode=env_cfg.get("render_mode"),
        )

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
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper.learning_rate)
        self.buffer = RolloutBuffer()

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

        self.memories: Dict[str, MemoryAugmentation] = {
            agent: MemoryAugmentation((self.grid_size, self.grid_size)) for agent in self.agents
        }

    def _make_reward_fn(self, name: str):
        name = (name or "").lower()
        if name in {"frequent", "frequent_asset", "asset"}:
            return FrequentAssetRewardFn()
        if name in {"land"}:
            return LandRewardFn()
        if name in {"winlose", "win_lose", "wl", "default"}:
            return WinLoseRewardFn()
        print(f"Unknown reward_fn '{name}', falling back to FrequentAssetRewardFn")
        return FrequentAssetRewardFn()

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
        for mem in self.memories.values():
            mem.reset()

    def collect_rollout(self, start_obs: np.ndarray, start_infos: Dict[str, Any], global_step: int) -> Tuple[np.ndarray, Dict[str, Any], int]:
        self.model.eval()
        obs = start_obs
        infos = start_infos
        steps_collected = 0
        self.buffer.clear()

        while steps_collected < self.hyper.rollout_length and global_step < self.hyper.total_env_steps:
            actions: List[Action] = []
            action_arrays: Dict[str, np.ndarray] = {}
            logprobs_step: Dict[str, torch.Tensor] = {}
            values_step: Dict[str, torch.Tensor] = {}
            masks_step: Dict[str, torch.Tensor] = {}
            action_indices_step: Dict[str, int] = {}

            # Compute actions for each agent (self-play using the same policy)
            for idx, agent in enumerate(self.agents):
                obs_agent = obs[idx]
                mem_agent = self.memories[agent].get_memory_features()
                valid_mask = infos[agent]["masks"]
                
                # Clone mask immediately to avoid modification by env.step
                masks_step[agent] = torch.from_numpy(valid_mask).bool().clone()

                with torch.no_grad():
                    act, act_arr, action_idx, logprob, value = self._action_for_agent(obs_agent, mem_agent, valid_mask)
                    action_indices_step[agent] = action_idx.item()

                actions.append(act)
                action_arrays[agent] = act_arr
                logprobs_step[agent] = logprob
                values_step[agent] = value

            next_obs, _, terminated, truncated, next_infos = self.env.step(actions)

            # Reward for learning agent (player_0)
            reward = float(next_infos[self.agents[0]]["reward"])
            done_flag = bool(terminated or truncated or next_infos[self.agents[0]]["done"])

            # Store rollout only for learning agent (player_0)
            # Note: We must store the mask used to generate the action, which comes from 'infos' (start_infos or next_infos from prev step)
            # The 'infos' variable holds the info for the CURRENT observation 'obs'.
            # We clone the mask to ensure it's not modified later if it shares memory with numpy array
            self.buffer.add(
                obs=torch.from_numpy(obs[0]).float(),
                memory=torch.from_numpy(self.memories[self.agents[0]].get_memory_features()).float(),
                mask=masks_step[self.agents[0]],
                action=logprobs_step[self.agents[0]].detach().new_tensor(action_indices_step[self.agents[0]]),
                logprob=logprobs_step[self.agents[0]].detach(),
                value=values_step[self.agents[0]].detach(),
                reward=reward,
                done=done_flag,
            )

            # Update memories using post-step observations
            for idx, agent in enumerate(self.agents):
                obs_dict = obs_tensor_to_dict(next_obs[idx])
                self.memories[agent].update(obs_dict, action_arrays[agent])

            if done_flag:
                obs, infos = self.env.reset()
                self._reset_memories()
            else:
                obs, infos = next_obs, next_infos

            steps_collected += 1
            global_step += 1

        return obs, infos, global_step

    def _compute_last_value(self, obs: np.ndarray, infos: Dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        obs_agent = obs[0]
        mem_agent = self.memories[self.agents[0]].get_memory_features()
        obs_t, mem_t = self._prepare_tensors(obs_agent, mem_agent)
        with torch.no_grad():
            policy_logits, value = self.model(obs_t, mem_t)
        return value.detach()

    def update(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        # Keep model in eval mode to ensure consistency with rollout (BatchNorm/Dropout)
        self.model.eval()
        batch = self.buffer.get_batch()
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

                values = values.view(-1)
                value_loss = F.mse_loss(values, returns_mb)

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

        progress = tqdm(total=self.hyper.total_env_steps, desc="PPO Training", unit="step")

        while global_step < self.hyper.total_env_steps:
            obs, infos, global_step = self.collect_rollout(obs, infos, global_step)
            avg_reward = torch.cat(self.buffer.rewards).mean().item()
            last_value = self._compute_last_value(obs, infos)
            advantages, returns = self.buffer.compute_advantages(last_value, self.hyper.gamma, self.hyper.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            metrics = self.update(advantages, returns)

            if self.use_swanlab:
                swanlab.log({
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": metrics["entropy"],
                    "train/approx_kl": metrics["approx_kl"],
                    "train/clip_rate": metrics["clip_rate"],
                    "train/ratio_mean": metrics["ratio_mean"],
                    "train/reward_mean": avg_reward,
                    "train/step": global_step,
                })

            if global_step % self.checkpoint_every == 0:
                self.save_checkpoint(global_step)

            if global_step % (self.log_interval * self.hyper.rollout_length) == 0:
                progress.set_postfix({
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
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

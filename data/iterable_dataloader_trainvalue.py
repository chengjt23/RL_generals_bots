import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from generals.core.action import Action
from agents.memory import MemoryAugmentation
from agents.reward_shaping import PotentialBasedRewardFn


class GeneralsReplayIterableDatasetWithValue(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        grid_size: int = 24,
        max_replays: int | None = None,
        min_stars: int = 70,
        max_turns: int = 500,
        gamma: float = 0.99,
        n_step: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.min_stars = min_stars
        self.max_turns = max_turns
        self.max_replays = max_replays
        self.gamma = gamma
        self.n_step = n_step
        
        self.reward_fn = PotentialBasedRewardFn(gamma=gamma)
        
        try:
            import pyarrow.parquet as pq
            self.parquet_file = pq.ParquetFile(
                self.data_dir / "data" / "train-00000-of-00001.parquet"
            )
            self.use_pyarrow = True
        except ImportError:
            import pandas as pd
            self.df = pd.read_parquet(self.data_dir / "data" / "train-00000-of-00001.parquet")
            self.use_pyarrow = False
        
        if self.use_pyarrow:
            self.num_row_groups = self.parquet_file.num_row_groups
        else:
            self.num_replays = len(self.df)
            if max_replays is not None:
                self.num_replays = min(self.num_replays, max_replays)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, float, int]]:
        worker_info = get_worker_info()
        
        if worker_info is None:
            w_id = 0
            w_num = 1
        else:
            w_id = worker_info.id
            w_num = worker_info.num_workers
        
        if self.use_pyarrow:
            groups_per_worker = (self.num_row_groups + w_num - 1) // w_num
            start_group = w_id * groups_per_worker
            end_group = min(self.num_row_groups, (w_id + 1) * groups_per_worker)
            
            max_replays_per_worker = None
            if self.max_replays is not None:
                max_replays_per_worker = (self.max_replays + w_num - 1) // w_num
            
            valid_replay_count = 0
            for g in range(start_group, end_group):
                if max_replays_per_worker is not None and valid_replay_count >= max_replays_per_worker:
                    break
                
                row_group = self.parquet_file.read_row_group(g)
                batch = row_group.to_pydict()
                
                num_rows = len(batch['mapWidth'])
                for i in range(num_rows):
                    if max_replays_per_worker is not None and valid_replay_count >= max_replays_per_worker:
                        break
                    
                    replay = {k: batch[k][i] for k in batch}
                    
                    if not self._is_valid_replay(replay):
                        continue
                    
                    valid_replay_count += 1
                    
                    for sample in self._extract_samples_from_replay(replay):
                        obs, memory, action, n_step_return, next_obs, next_memory, done, player_idx = sample
                        obs_tensor = torch.from_numpy(obs).to(torch.float32)
                        memory_tensor = torch.from_numpy(memory).to(torch.float32)
                        action_tensor = torch.from_numpy(action).long()
                        next_obs_tensor = torch.from_numpy(next_obs).to(torch.float32)
                        next_memory_tensor = torch.from_numpy(next_memory).to(torch.float32)
                        yield obs_tensor, memory_tensor, action_tensor, n_step_return, next_obs_tensor, next_memory_tensor, done, player_idx
        else:
            per_worker = self.num_replays // w_num
            start_idx = w_id * per_worker
            end_idx = start_idx + per_worker
            if w_id == w_num - 1:
                end_idx = self.num_replays
            
            for idx in range(start_idx, end_idx):
                replay = self.df.iloc[idx].to_dict()
                
                if not self._is_valid_replay(replay):
                    continue
                
                for sample in self._extract_samples_from_replay(replay):
                    obs, memory, action, n_step_return, next_obs, next_memory, done, player_idx = sample
                    obs_tensor = torch.from_numpy(obs).to(torch.float32)
                    memory_tensor = torch.from_numpy(memory).to(torch.float32)
                    action_tensor = torch.from_numpy(action).long()
                    next_obs_tensor = torch.from_numpy(next_obs).to(torch.float32)
                    next_memory_tensor = torch.from_numpy(next_memory).to(torch.float32)
                    yield obs_tensor, memory_tensor, action_tensor, n_step_return, next_obs_tensor, next_memory_tensor, done, player_idx
    
    def _is_valid_replay(self, replay: Dict) -> bool:
        if len(replay['moves']) > self.max_turns:
            return False
        
        if max(replay['stars']) < self.min_stars:
            return False
        
        return True
    
    def _extract_samples_from_replay(self, replay: Dict) -> Iterator[Tuple]:
        width = replay['mapWidth']
        height = replay['mapHeight']
        
        grid = self._create_initial_grid(replay, height, width)
        
        try:
            game = Game(grid, ["player_0", "player_1"])
        except Exception:
            return
        
        memory_0 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        memory_1 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        
        trajectory_0 = []
        trajectory_1 = []
        
        for move_idx, move in enumerate(replay['moves']):
            if len(move) < 5:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            
            obs_0 = game.agent_observation("player_0")
            obs_1 = game.agent_observation("player_1")
            
            obs_0.pad_observation(pad_to=self.grid_size)
            obs_1.pad_observation(pad_to=self.grid_size)
            obs_0_prev = self._obs_to_dict(obs_0)
            obs_1_prev = self._obs_to_dict(obs_1)
            
            start_row = start_tile // width
            start_col = start_tile % width
            end_row = end_tile // width
            end_col = end_tile % width
            
            direction = self._get_direction(start_row, start_col, end_row, end_col)
            if direction == -1:
                continue
            
            action = np.array([0, start_row, start_col, direction, is_half], dtype=np.int8)
            
            if player_idx == 0:
                obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                trajectory_0.append({
                    'obs': obs_tensor,
                    'memory': memory_features,
                    'action': action.copy(),
                    'obs_object': obs_0,
                    'move_idx': move_idx,
                })
            else:
                obs_tensor = obs_1.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_1.get_memory_features().astype(np.float32, copy=True)
                trajectory_1.append({
                    'obs': obs_tensor,
                    'memory': memory_features,
                    'action': action.copy(),
                    'obs_object': obs_1,
                    'move_idx': move_idx,
                })
            
            actions = {
                "player_0": np.array([1, 0, 0, 0, 0], dtype=np.int8),
                "player_1": np.array([1, 0, 0, 0, 0], dtype=np.int8),
            }
            
            if player_idx == 0:
                actions["player_0"] = action
            else:
                actions["player_1"] = action
            
            try:
                game.step(actions)
                
                obs_0_after = game.agent_observation("player_0")
                obs_1_after = game.agent_observation("player_1")
                obs_0_after.pad_observation(pad_to=self.grid_size)
                obs_1_after.pad_observation(pad_to=self.grid_size)
                
                obs_0_curr = self._obs_to_dict(obs_0_after)
                obs_1_curr = self._obs_to_dict(obs_1_after)
                
                opp_action_for_0 = self._infer_opponent_action(obs_0_prev, obs_0_curr)
                opp_action_for_1 = self._infer_opponent_action(obs_1_prev, obs_1_curr)
                
                memory_0.update(obs_0_curr, actions["player_0"], opp_action_for_0)
                memory_1.update(obs_1_curr, actions["player_1"], opp_action_for_1)
                
                # Compute reward for the previous state
                if player_idx == 0 and len(trajectory_0) > 0:
                    prev_obs_obj = trajectory_0[-1]['obs_object']
                    reward = self.reward_fn(prev_obs_obj, Action(
                        to_pass=action[0] == 1,
                        row=int(action[1]),
                        col=int(action[2]),
                        direction=int(action[3]),
                        to_split=action[4] == 1
                    ), obs_0_after)
                    trajectory_0[-1]['reward'] = reward
                    trajectory_0[-1]['next_obs'] = obs_0_after.as_tensor().astype(np.float32, copy=True)
                    trajectory_0[-1]['next_memory'] = memory_0.get_memory_features().astype(np.float32, copy=True)
                elif player_idx == 1 and len(trajectory_1) > 0:
                    prev_obs_obj = trajectory_1[-1]['obs_object']
                    reward = self.reward_fn(prev_obs_obj, Action(
                        to_pass=action[0] == 1,
                        row=int(action[1]),
                        col=int(action[2]),
                        direction=int(action[3]),
                        to_split=action[4] == 1
                    ), obs_1_after)
                    trajectory_1[-1]['reward'] = reward
                    trajectory_1[-1]['next_obs'] = obs_1_after.as_tensor().astype(np.float32, copy=True)
                    trajectory_1[-1]['next_memory'] = memory_1.get_memory_features().astype(np.float32, copy=True)
                
            except Exception:
                break
        
        # Mark terminal states
        final_obs_0 = game.agent_observation("player_0")
        final_obs_1 = game.agent_observation("player_1")
        final_obs_0.pad_observation(pad_to=self.grid_size)
        final_obs_1.pad_observation(pad_to=self.grid_size)
        
        agent_0_generals = (final_obs_0.generals & final_obs_0.owned_cells).sum()
        agent_1_generals = (final_obs_1.generals & final_obs_1.owned_cells).sum()
        
        terminal_reward_0 = 0.0
        terminal_reward_1 = 0.0
        if agent_0_generals == 0:
            terminal_reward_0 = -1.0
        if agent_1_generals == 0:
            terminal_reward_1 = -1.0
        if agent_0_generals > 0 and agent_1_generals == 0:
            terminal_reward_0 = 1.0
        if agent_1_generals > 0 and agent_0_generals == 0:
            terminal_reward_1 = 1.0
        
        if len(trajectory_0) > 0:
            trajectory_0[-1]['reward'] = trajectory_0[-1].get('reward', 0.0) + terminal_reward_0
            trajectory_0[-1]['next_obs'] = final_obs_0.as_tensor().astype(np.float32, copy=True)
            trajectory_0[-1]['next_memory'] = memory_0.get_memory_features().astype(np.float32, copy=True)
        
        if len(trajectory_1) > 0:
            trajectory_1[-1]['reward'] = trajectory_1[-1].get('reward', 0.0) + terminal_reward_1
            trajectory_1[-1]['next_obs'] = final_obs_1.as_tensor().astype(np.float32, copy=True)
            trajectory_1[-1]['next_memory'] = memory_1.get_memory_features().astype(np.float32, copy=True)
        
        # Compute n-step returns and yield samples
        for samples in self._compute_nstep_samples(trajectory_0, 0):
            yield samples
        
        for samples in self._compute_nstep_samples(trajectory_1, 1):
            yield samples

    def _compute_nstep_samples(self, trajectory: List[Dict], player_idx: int) -> Iterator[Tuple]:
        """Compute n-step TD samples from trajectory"""
        T = len(trajectory)
        
        for t in range(T):
            # Compute n-step return: r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}
            n_step_return = 0.0
            done = 0.0
            
            # Determine actual steps to look ahead
            steps = min(self.n_step, T - t)
            
            for k in range(steps):
                reward = trajectory[t + k].get('reward', 0.0)
                n_step_return += (self.gamma ** k) * reward
            
            # Check if episode terminates within n steps
            if t + steps >= T:
                # Episode ends within n steps
                done = 1.0
                # Use zero state as next state (will be masked by done flag)
                next_obs = np.zeros_like(trajectory[t]['obs'])
                next_memory = np.zeros_like(trajectory[t]['memory'])
            else:
                # Episode continues, use state at t+n_step (before action at t+n_step)
                next_obs = trajectory[t + steps]['obs']
                next_memory = trajectory[t + steps]['memory']
            
            yield (
                trajectory[t]['obs'],
                trajectory[t]['memory'],
                trajectory[t]['action'],
                np.float32(n_step_return),
                next_obs,
                next_memory,
                np.float32(done),
                player_idx
            )

    def _infer_opponent_action(self, prev: dict, curr: dict) -> np.ndarray:
        fog = curr['fog_cells']
        sif = curr['structures_in_fog']
        visible = (fog == 0) & (sif == 0)

        prev_opp = prev['opponent_cells']
        cur_opp = curr['opponent_cells']

        new_opp = visible & (cur_opp.astype(bool)) & (~prev_opp.astype(bool))
        new_positions = np.argwhere(new_opp)
        
        action = np.array([1, 0, 0, 0, 0], dtype=np.int8)

        if new_positions.shape[0] != 1:
            return action

        dest_r, dest_c = (int(new_positions[0][0]), int(new_positions[0][1]))

        for direction, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            src_r, src_c = dest_r - dr, dest_c - dc
            if src_r < 0 or src_c < 0 or src_r >= self.grid_size or src_c >= self.grid_size:
                continue
            if prev_opp[src_r, src_c] == 1:
                return np.array([0, src_r, src_c, direction, 0], dtype=np.int8)

        return action
    
    def _create_initial_grid(self, replay: Dict, height: int, width: int) -> Grid:
        grid_array = np.full((height, width), '.', dtype='U1')
        
        for mountain_idx in replay['mountains']:
            row = mountain_idx // width
            col = mountain_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = '#'
        
        for city_idx in replay['cities']:
            row = city_idx // width
            col = city_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = 'C'
        
        general_indices = replay['generals']
        if len(general_indices) >= 2:
            gen_0 = general_indices[0]
            gen_1 = general_indices[1]
            
            row_0 = gen_0 // width
            col_0 = gen_0 % width
            row_1 = gen_1 // width
            col_1 = gen_1 % width
            
            if 0 <= row_0 < height and 0 <= col_0 < width:
                grid_array[row_0, col_0] = 'A'
            if 0 <= row_1 < height and 0 <= col_1 < width:
                grid_array[row_1, col_1] = 'B'
        
        return Grid(grid_array)
    
    def _get_direction(self, start_row: int, start_col: int, end_row: int, end_col: int) -> int:
        if start_row == end_row and start_col == end_col:
            return -1
        
        if end_row < start_row:
            return 0
        elif end_row > start_row:
            return 1
        elif end_col < start_col:
            return 2
        elif end_col > start_col:
            return 3
        return -1
    
    def _obs_to_dict(self, obs: Observation) -> dict:
        tensor = obs.as_tensor()
        return {
            'armies': tensor[0],
            'generals': tensor[1],
            'cities': tensor[2],
            'mountains': tensor[3],
            'neutral_cells': tensor[4],
            'owned_cells': tensor[5],
            'opponent_cells': tensor[6],
            'fog_cells': tensor[7],
            'structures_in_fog': tensor[8],
        }


def create_iterable_dataloader_with_value(
    data_dir: str,
    batch_size: int,
    grid_size: int = 24,
    num_workers: int = 4,
    max_replays: int | None = None,
    min_stars: int = 70,
    max_turns: int = 500,
    gamma: float = 0.99,
    n_step: int = 50,
) -> DataLoader:
    dataset = GeneralsReplayIterableDatasetWithValue(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars,
        max_turns=max_turns,
        gamma=gamma,
        n_step=n_step,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


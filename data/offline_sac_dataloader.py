import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from typing import Iterator, Tuple
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from agents.memory import MemoryAugmentation
from agents.reward_shaping import PotentialBasedRewardFn


class OfflineSACDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        grid_size: int = 24,
        max_replays: int | None = None,
        min_stars: int = 70,
        max_turns: int = 500,
    ):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.min_stars = min_stars
        self.max_turns = max_turns
        self.max_replays = max_replays
        
        self.reward_fn = PotentialBasedRewardFn()
        
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
    
    def __iter__(self) -> Iterator[Tuple]:
        worker_info = get_worker_info()
        
        if worker_info is None:
            w_id = 0
            w_num = 1
        else:
            w_id = worker_info.id
            w_num = worker_info.num_workers
        
        if self.use_pyarrow:
            for rg_idx in range(w_id, self.num_row_groups, w_num):
                table = self.parquet_file.read_row_group(rg_idx)
                df_chunk = table.to_pandas()
                
                for _, row in df_chunk.iterrows():
                    yield from self._process_replay(row.to_dict())
        else:
            for idx in range(w_id, self.num_replays, w_num):
                replay = self.df.iloc[idx].to_dict()
                yield from self._process_replay(replay)
    
    def _process_replay(self, replay: dict):
        stars = replay.get('stars', [0])
        if hasattr(stars, '__iter__') and not isinstance(stars, str):
            stars = max(stars) if len(stars) > 0 else 0
        if stars < self.min_stars:
            return
        
        width = replay['mapWidth']
        height = replay['mapHeight']
        if hasattr(width, '__iter__'):
            width = int(width[0]) if len(width) > 0 else 0
        else:
            width = int(width)
        if hasattr(height, '__iter__'):
            height = int(height[0]) if len(height) > 0 else 0
        else:
            height = int(height)
        
        if width > self.grid_size or height > self.grid_size:
            return
        
        grid = self._create_initial_grid(replay, height, width)
        game = Game(grid)
        
        memory_0 = MemoryAugmentation((height, width), history_length=7)
        memory_1 = MemoryAugmentation((height, width), history_length=7)
        
        afks = replay.get('afks', [])
        if hasattr(afks, 'tolist'):
            afks = afks.tolist()
        moves = replay.get('moves', [])
        if hasattr(moves, 'tolist'):
            moves = moves.tolist()
        
        if not moves:
            return
        
        turn = 0
        prior_obs_0 = None
        prior_obs_1 = None
        
        for move in moves:
            if turn >= self.max_turns:
                break
            
            if len(move) < 4:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            
            if player_idx in afks:
                continue
            
            turn += 1
            
            if start_tile < 0 or end_tile < 0:
                action_pass = np.array([1, 0, 0, 0, 0], dtype=np.int8)
                actions = {"player_0": action_pass, "player_1": action_pass}
                
                try:
                    game.step(actions)
                except Exception:
                    break
                continue
            
            start_row = start_tile // width
            start_col = start_tile % width
            end_row = end_tile // width
            end_col = end_tile % width
            
            direction = self._get_direction(start_row, start_col, end_row, end_col)
            if direction == -1:
                continue
            
            action = np.array([0, start_row, start_col, direction, is_half], dtype=np.int8)
            action_pass = np.array([1, 0, 0, 0, 0], dtype=np.int8)
            
            obs_0 = game.agent_observation("player_0")
            obs_1 = game.agent_observation("player_1")
            obs_0.pad_observation(pad_to=self.grid_size)
            obs_1.pad_observation(pad_to=self.grid_size)
            
            if player_idx == 0:
                obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                memory_features = self._pad_memory(memory_features, self.grid_size)
                action_array = action.copy()
            else:
                obs_tensor = obs_1.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_1.get_memory_features().astype(np.float32, copy=True)
                memory_features = self._pad_memory(memory_features, self.grid_size)
                action_array = action.copy()
            
            actions = {
                "player_0": action if player_idx == 0 else action_pass,
                "player_1": action if player_idx == 1 else action_pass
            }
            
            try:
                game.step(actions)
                
                next_obs_0 = game.agent_observation("player_0")
                next_obs_1 = game.agent_observation("player_1")
                next_obs_0.pad_observation(pad_to=self.grid_size)
                next_obs_1.pad_observation(pad_to=self.grid_size)
                
                obs_0_dict = self._obs_to_dict(obs_0)
                obs_1_dict = self._obs_to_dict(obs_1)
                next_obs_0_dict = self._obs_to_dict(next_obs_0)
                next_obs_1_dict = self._obs_to_dict(next_obs_1)
                
                memory_0.update(obs_0_dict, actions["player_0"], actions["player_1"])
                memory_1.update(obs_1_dict, actions["player_1"], actions["player_0"])
                
                if player_idx == 0:
                    next_obs_tensor = next_obs_0.as_tensor().astype(np.float32, copy=True)
                    next_memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                    next_memory_features = self._pad_memory(next_memory_features, self.grid_size)
                    
                    if prior_obs_0 is not None:
                        reward = self.reward_fn(prior_obs_0, action, obs_0)
                    else:
                        reward = 0.0
                    
                    prior_obs_0 = next_obs_0
                else:
                    next_obs_tensor = next_obs_1.as_tensor().astype(np.float32, copy=True)
                    next_memory_features = memory_1.get_memory_features().astype(np.float32, copy=True)
                    next_memory_features = self._pad_memory(next_memory_features, self.grid_size)
                    
                    if prior_obs_1 is not None:
                        reward = self.reward_fn(prior_obs_1, action, obs_1)
                    else:
                        reward = 0.0
                    
                    prior_obs_1 = next_obs_1
                
                done = 0.0
                
                yield (obs_tensor, memory_features, action_array, reward, 
                       next_obs_tensor, next_memory_features, done)
                
            except Exception:
                break
    
    def _pad_memory(self, memory: np.ndarray, target_size: int) -> np.ndarray:
        current_h, current_w = memory.shape[1], memory.shape[2]
        if current_h < target_size or current_w < target_size:
            pad_h = max(0, target_size - current_h)
            pad_w = max(0, target_size - current_w)
            memory = np.pad(memory, ((0, 0), (0, pad_h), (0, pad_w)), 
                           mode='constant', constant_values=0)
        return memory
    
    def _obs_to_dict(self, obs: Observation) -> dict:
        return {
            "armies": obs.armies,
            "generals": obs.generals,
            "cities": obs.cities,
            "mountains": obs.mountains,
            "neutral_cells": obs.neutral_cells,
            "owned_cells": obs.owned_cells,
            "opponent_cells": obs.opponent_cells,
            "fog_cells": obs.fog_cells,
            "structures_in_fog": obs.structures_in_fog,
        }
    
    def _create_initial_grid(self, replay: dict, height: int, width: int) -> Grid:
        grid_array = np.full((height, width), '.', dtype='U1')
        
        mountains = replay['mountains']
        if hasattr(mountains, 'tolist'):
            mountains = mountains.tolist()
        for mountain_idx in mountains:
            row = mountain_idx // width
            col = mountain_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = '#'
        
        cities = replay['cities']
        city_armies = replay['cityArmies']
        if hasattr(cities, 'tolist'):
            cities = cities.tolist()
        if hasattr(city_armies, 'tolist'):
            city_armies = city_armies.tolist()
        for city_idx, army in zip(cities, city_armies):
            row = city_idx // width
            col = city_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = 'C'
        
        generals = replay['generals']
        if hasattr(generals, 'tolist'):
            generals = generals.tolist()
        if len(generals) >= 2:
            gen0_row, gen0_col = generals[0] // width, generals[0] % width
            gen1_row, gen1_col = generals[1] // width, generals[1] % width
            
            if 0 <= gen0_row < height and 0 <= gen0_col < width:
                grid_array[gen0_row, gen0_col] = 'A'
            if 0 <= gen1_row < height and 0 <= gen1_col < width:
                grid_array[gen1_row, gen1_col] = 'B'
        
        grid_str = '\n'.join([''.join(row) for row in grid_array])
        return Grid.from_string(grid_str)
    
    def _get_direction(self, start_row: int, start_col: int, end_row: int, end_col: int) -> int:
        if end_row < start_row and end_col == start_col:
            return 0
        elif end_col > start_col and end_row == start_row:
            return 1
        elif end_row > start_row and end_col == start_col:
            return 2
        elif end_col < start_col and end_row == start_row:
            return 3
        return -1


def create_offline_sac_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    grid_size: int = 24,
    max_replays: int | None = None,
    min_stars: int = 70,
) -> DataLoader:
    dataset = OfflineSACDataset(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )


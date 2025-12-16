import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from agents.memory import MemoryAugmentation


class GeneralsReplayIterableDataset(IterableDataset):
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
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
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
                        obs, memory, action, player_idx = sample
                        obs_tensor = torch.from_numpy(obs).to(torch.float32)
                        memory_tensor = torch.from_numpy(memory).to(torch.float32)
                        action_tensor = torch.from_numpy(action).long()
                        yield obs_tensor, memory_tensor, action_tensor, player_idx
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
                    obs, memory, action, player_idx = sample
                    obs_tensor = torch.from_numpy(obs).to(torch.float32)
                    memory_tensor = torch.from_numpy(memory).to(torch.float32)
                    action_tensor = torch.from_numpy(action).long()
                    yield obs_tensor, memory_tensor, action_tensor, player_idx
    
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
        
        # Initialize memory augmentation for both players
        memory_0 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        memory_1 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        
        turn_step = 0
        for move in replay['moves']:
            turn_step += 1
            if len(move) < 5:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            
            obs_0 = game.agent_observation("player_0")
            obs_1 = game.agent_observation("player_1")
            
            start_row = start_tile // width
            start_col = start_tile % width
            end_row = end_tile // width
            end_col = end_tile % width
            
            direction = self._get_direction(start_row, start_col, end_row, end_col)
            if direction == -1:
                continue
            
            action = np.array([0, start_row, start_col, direction, is_half], dtype=np.int8)
            action_pass = np.array([1, 0, 0, 0, 0], dtype=np.int8)
            
            # Get current memory features before taking action
            if player_idx == 0:
                obs_0.pad_observation(pad_to=self.grid_size)
                obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                yield (obs_tensor, memory_features, action.copy(), 0)
            else:
                obs_1.pad_observation(pad_to=self.grid_size)
                obs_tensor = obs_1.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_1.get_memory_features().astype(np.float32, copy=True)
                yield (obs_tensor, memory_features, action.copy(), 1)
            
            actions = {
                "player_0": np.array([1, 0, 0, 0, 0], dtype=np.int8),
                "player_1": np.array([1, 0, 0, 0, 0], dtype=np.int8),
            }
            
            if player_idx == 0:
                actions["player_0"] = action
            else:
                actions["player_1"] = action
            
            # Filter opponent actions based on visibility to prevent "God mode"
            # We use the pre-step observation to check if the move start position was visible
            action_0_for_memory = actions["player_0"].copy()
            action_1_for_memory = actions["player_1"].copy()
            
            # Filter Player 1's action for Player 0's memory
            if action_1_for_memory[0] == 0: # Move
                r, c = action_1_for_memory[1], action_1_for_memory[2]
                # Check if start tile is in fog (channel 0 > 0)
                if obs_0.as_tensor()[0, r, c] > 0:
                    action_1_for_memory = np.array([1, 0, 0, 0, 0], dtype=np.int8) # Mask as Pass

            # Filter Player 0's action for Player 1's memory
            if action_0_for_memory[0] == 0: # Move
                r, c = action_0_for_memory[1], action_0_for_memory[2]
                if obs_1.as_tensor()[0, r, c] > 0:
                    action_0_for_memory = np.array([1, 0, 0, 0, 0], dtype=np.int8)
            
            try:
                game.step(actions)
                
                # Update memory augmentation after step
                obs_0_after = game.agent_observation("player_0")
                obs_1_after = game.agent_observation("player_1")
                obs_0_after.pad_observation(pad_to=self.grid_size)
                obs_1_after.pad_observation(pad_to=self.grid_size)
                
                # Convert observations to dict format for memory update
                obs_0_dict = self._obs_to_dict(obs_0_after)
                obs_1_dict = self._obs_to_dict(obs_1_after)
                
                memory_0.update(obs_0_dict, actions["player_0"], action_1_for_memory, turn_step=turn_step)
                memory_1.update(obs_1_dict, actions["player_1"], action_0_for_memory, turn_step=turn_step)
            except Exception:
                break
    
    def _create_initial_grid(self, replay: Dict, height: int, width: int) -> Grid:
        grid_array = np.full((height, width), '.', dtype='U1')
        
        for mountain_idx in replay['mountains']:
            row = mountain_idx // width
            col = mountain_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = '#'
        
        for city_idx, army in zip(replay['cities'], replay['cityArmies']):
            row = city_idx // width
            col = city_idx % width
            if 0 <= row < height and 0 <= col < width:
                city_value = min(max(army - 40, 0), 10)
                if city_value == 10:
                    grid_array[row, col] = 'x'
                else:
                    grid_array[row, col] = str(city_value)
        
        for player_idx, general_idx in enumerate(replay['generals']):
            row = general_idx // width
            col = general_idx % width
            if 0 <= row < height and 0 <= col < width:
                grid_array[row, col] = chr(ord('A') + player_idx)
        
        grid_str = '\n'.join([''.join(row) for row in grid_array])
        return Grid(grid_str)
    
    def _get_direction(self, start_row: int, start_col: int, end_row: int, end_col: int) -> int:
        if end_row < start_row and end_col == start_col:
            return 0
        elif end_row > start_row and end_col == start_col:
            return 1
        elif end_row == start_row and end_col < start_col:
            return 2
        elif end_row == start_row and end_col > start_col:
            return 3
        return -1
    
    def _obs_to_dict(self, obs: Observation) -> Dict:
        """Convert Observation to dict format needed by MemoryAugmentation"""
        tensor = obs.as_tensor()
        return {
            'fog_cells': tensor[0],
            'structures_in_fog': tensor[1],
            'cities': tensor[2],
            'generals': tensor[3],
            'owned_cells': tensor[4],
            'opponent_cells': tensor[5],
            'armies': obs.armies,
        }


def create_iterable_dataloader(
    data_dir: str,
    batch_size: int = 32,
    grid_size: int = 24,
    num_workers: int = 0,
    max_replays: int | None = None,
    min_stars: int = 70,
    max_turns: int = 500,
) -> DataLoader:
    dataset = GeneralsReplayIterableDataset(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars,
        max_turns=max_turns,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


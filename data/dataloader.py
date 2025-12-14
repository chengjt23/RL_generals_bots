import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from agents.memory import MemoryAugmentation


class GeneralsReplayDataset(Dataset):
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
        
        try:
            import pyarrow.parquet as pq
            self.parquet_file = pq.ParquetFile(
                self.data_dir / "data" / "train-00000-of-00001.parquet"
            )
            self.table = self.parquet_file.read()
            self.num_replays = len(self.table)
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(self.data_dir / "data" / "train-00000-of-00001.parquet")
            self.df = df
            self.num_replays = len(df)
            self.table = None
        
        if max_replays is not None:
            self.num_replays = min(self.num_replays, max_replays)
        
        self.samples = []
        self._process_replays()
    
    def _process_replays(self):
        print(f"\n{'='*60}")
        print(f"Processing replays from dataset")
        print(f"{'='*60}")
        
        pbar = tqdm(range(self.num_replays), desc="Loading replays", unit="replay")
        for idx in pbar:
            if self.table is not None:
                replay = {
                    'mapWidth': self.table['mapWidth'][idx].as_py(),
                    'mapHeight': self.table['mapHeight'][idx].as_py(),
                    'usernames': self.table['usernames'][idx].as_py(),
                    'stars': self.table['stars'][idx].as_py(),
                    'cities': self.table['cities'][idx].as_py(),
                    'cityArmies': self.table['cityArmies'][idx].as_py(),
                    'generals': self.table['generals'][idx].as_py(),
                    'mountains': self.table['mountains'][idx].as_py(),
                    'moves': self.table['moves'][idx].as_py(),
                }
            else:
                replay = self.df.iloc[idx].to_dict()
            
            if not self._is_valid_replay(replay):
                continue
            
            samples = self._extract_samples_from_replay(replay)
            self.samples.extend(samples)
            
            pbar.set_postfix({
                'samples': len(self.samples),
                'avg_moves': len(self.samples) // (idx + 1) if idx > 0 else 0
            })
        
        print(f"\n{'='*60}")
        print(f"✓ Dataset ready: {len(self.samples):,} training samples")
        print(f"{'='*60}\n")
    
    def _is_valid_replay(self, replay: Dict) -> bool:
        if len(replay['moves']) > self.max_turns:
            return False
        
        if max(replay['stars']) < self.min_stars:
            return False
        
        return True
    
    def _extract_samples_from_replay(self, replay: Dict) -> List[Tuple]:
        samples = []
        width = replay['mapWidth']
        height = replay['mapHeight']
        
        grid = self._create_initial_grid(replay, height, width)
        
        try:
            game = Game(grid, ["player_0", "player_1"])
        except Exception:
            return samples
        
        # Initialize memory augmentation for both players
        memory_0 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        memory_1 = MemoryAugmentation((self.grid_size, self.grid_size), history_length=7)
        
        for move in replay['moves']:
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
                obs_tensor = obs_0.as_tensor().astype(np.float16, copy=True)
                memory_features = memory_0.get_memory_features().astype(np.float16, copy=True)
                samples.append((obs_tensor, memory_features, action.copy(), 0))
            else:
                obs_1.pad_observation(pad_to=self.grid_size)
                obs_tensor = obs_1.as_tensor().astype(np.float16, copy=True)
                memory_features = memory_1.get_memory_features().astype(np.float16, copy=True)
                samples.append((obs_tensor, memory_features, action.copy(), 1))
            
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
                
                # Update memory augmentation after step
                obs_0_after = game.agent_observation("player_0")
                obs_1_after = game.agent_observation("player_1")
                obs_0_after.pad_observation(pad_to=self.grid_size)
                obs_1_after.pad_observation(pad_to=self.grid_size)
                
                # Convert observations to dict format for memory update
                obs_0_dict = self._obs_to_dict(obs_0_after)
                obs_1_dict = self._obs_to_dict(obs_1_after)
                
                memory_0.update(obs_0_dict, actions["player_0"], actions["player_1"])
                memory_1.update(obs_1_dict, actions["player_1"], actions["player_0"])
            except Exception:
                break
        
        return samples
    
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
    
    def _obs_to_dict(self, obs: Observation) -> dict:
        """Convert Observation to dict format needed by MemoryAugmentation"""
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
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        obs, memory, action, player_idx = self.samples[idx]

        obs_tensor = torch.from_numpy(obs).to(torch.float32)
        memory_tensor = torch.from_numpy(memory).to(torch.float32)
        action_tensor = torch.from_numpy(action).long()

        return obs_tensor, memory_tensor, action_tensor, player_idx


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    grid_size: int = 24,
    num_workers: int = 0,
    max_replays: int | None = None,
    min_stars: int = 70,
    max_turns: int = 500,
) -> DataLoader:
    dataset = GeneralsReplayDataset(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars,
        max_turns=max_turns,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from generals.core.action import Action


class GeneralsReplayIterableDataset(IterableDataset):
    """
    Iterable dataset for training with RNN memory.
    
    Provides per-step (obs, action) pairs without hand-crafted memory features.
    The RNN memory encoder will learn temporal patterns through hidden states.
    """
    
    def __init__(
        self,
        data_dir: str,
        grid_size: int = 24,
        max_replays: int | None = None,
        min_stars: int = 70,
        max_turns: int = 500,
        sequence_length: int = 32,
        skip_replays: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.min_stars = min_stars
        self.max_turns = max_turns
        self.max_replays = max_replays
        self.sequence_length = sequence_length
        self.skip_replays = skip_replays
        
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
            
            skip_replays_per_worker = (self.skip_replays + w_num - 1) // w_num
            
            valid_replay_count = 0
            skipped_count = 0
            
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
                    
                    if skipped_count < skip_replays_per_worker:
                        skipped_count += 1
                        continue
                    
                    valid_replay_count += 1
                    
                    for sample in self._extract_samples_from_replay(replay):
                        obs, action, player_idx = sample
                        obs_tensor = torch.from_numpy(obs).to(torch.float32)
                        action_tensor = torch.from_numpy(action).long()
                        yield obs_tensor, action_tensor, player_idx
        else:
            total_replays = len(self.df)
            start_replay = self.skip_replays
            end_replay = total_replays
            if self.max_replays is not None:
                end_replay = min(total_replays, start_replay + self.max_replays)
            
            num_to_process = max(0, end_replay - start_replay)
            per_worker = (num_to_process + w_num - 1) // w_num
            
            worker_start = start_replay + w_id * per_worker
            worker_end = min(end_replay, worker_start + per_worker)
            
            for idx in range(worker_start, worker_end):
                replay = self.df.iloc[idx].to_dict()
                
                if not self._is_valid_replay(replay):
                    continue
                
                for sample in self._extract_samples_from_replay(replay):
                    obs, action, player_idx = sample
                    obs_tensor = torch.from_numpy(obs).to(torch.float32)
                    action_tensor = torch.from_numpy(action).long()
                    yield obs_tensor, action_tensor, player_idx
    
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
        
        # Buffers for sequences
        obs_buffer_0, action_buffer_0 = [], []
        obs_buffer_1, action_buffer_1 = [], []
        
        for move in replay['moves']:
            if len(move) < 5:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            
            obs_0 = game.agent_observation("player_0")
            obs_1 = game.agent_observation("player_1")
            
            # Pad observations
            obs_0.pad_observation(pad_to=self.grid_size)
            obs_1.pad_observation(pad_to=self.grid_size)
            
            start_row = start_tile // width
            start_col = start_tile % width
            end_row = end_tile // width
            end_col = end_tile % width
            
            direction = self._get_direction(start_row, start_col, end_row, end_col)
            if direction == -1:
                continue
            
            action = np.array([0, start_row, start_col, direction, is_half], dtype=np.int8)
            
            # Collect samples
            if player_idx == 0:
                obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                obs_buffer_0.append(obs_tensor)
                action_buffer_0.append(action.copy())
            else:
                obs_tensor = obs_1.as_tensor().astype(np.float32, copy=True)
                obs_buffer_1.append(obs_tensor)
                action_buffer_1.append(action.copy())
            
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
            except Exception:
                break
        
        # Yield full episodes
        for obs_buf, act_buf, pid in [
            (obs_buffer_0, action_buffer_0, 0),
            (obs_buffer_1, action_buffer_1, 1)
        ]:
            if len(obs_buf) == 0:
                continue
                
            obs_seq = np.stack(obs_buf)
            act_seq = np.stack(act_buf)
            yield (obs_seq, act_seq, pid)
    
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


def collate_episode_batch(batch):
    """
    Collate function for variable length episodes.
    Pads sequences to the maximum length in the batch.
    """
    # batch is list of (obs_seq, act_seq, pid)
    # obs_seq: (L, C, H, W)
    # act_seq: (L, 5)
    
    # Sort by length (descending) for efficiency
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    obs_seqs, act_seqs, pids = zip(*batch)
    lengths = torch.tensor([len(x) for x in obs_seqs])
    
    # Pad sequences
    # obs_padded: (B, MaxLen, C, H, W)
    # act_padded: (B, MaxLen, 5)
    obs_padded = torch.nn.utils.rnn.pad_sequence(obs_seqs, batch_first=True)
    act_padded = torch.nn.utils.rnn.pad_sequence(act_seqs, batch_first=True)
    
    return obs_padded, act_padded, lengths


def create_iterable_dataloader(
    data_dir: str,
    batch_size: int = 32,
    grid_size: int = 24,
    num_workers: int = 0,
    max_replays: int | None = None,
    min_stars: int = 70,
    max_turns: int = 500,
    sequence_length: int = 32,
    skip_replays: int = 0,
) -> DataLoader:
    dataset = GeneralsReplayIterableDataset(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars,
        max_turns=max_turns,
        sequence_length=sequence_length,
        skip_replays=skip_replays,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_episode_batch,
    )


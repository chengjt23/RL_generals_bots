import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
from tqdm import tqdm
from generals.core.grid import Grid
from generals.core.game import Game
from generals.core.observation import Observation
from generals.core.action import Action
from agents.memory import MemoryAugmentation


class GeneralsReplayIterableDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        grid_size: int = 24,
        max_replays: int | None = None,
        min_stars: int = 70,
        max_turns: int = 500,
        sequence_len: int = 32,
        batch_size: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.min_stars = min_stars
        self.max_turns = max_turns
        self.max_replays = max_replays
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        
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
    
    def _get_replay_iterator(self, worker_id: int, num_workers: int) -> Iterator[Tuple[Dict, int]]:
        """Yields (replay, player_idx) tuples indefinitely for a specific worker."""
        if self.use_pyarrow:
            groups_per_worker = (self.num_row_groups + num_workers - 1) // num_workers
            start_group = worker_id * groups_per_worker
            end_group = min(self.num_row_groups, (worker_id + 1) * groups_per_worker)
            
            max_replays_per_worker = None
            if self.max_replays is not None:
                max_replays_per_worker = (self.max_replays + num_workers - 1) // num_workers
            
            while True:
                group_indices = torch.arange(start_group, end_group)
                perm = torch.randperm(len(group_indices))
                shuffled_groups = group_indices[perm].tolist()
                
                valid_replay_count = 0
                for g in shuffled_groups:
                    if max_replays_per_worker is not None and valid_replay_count >= max_replays_per_worker:
                        break
                    
                    row_group = self.parquet_file.read_row_group(g)
                    batch = row_group.to_pydict()
                    
                    num_rows = len(batch['mapWidth'])
                    # Shuffle within row group
                    row_indices = torch.randperm(num_rows).tolist()
                    
                    for i in row_indices:
                        if max_replays_per_worker is not None and valid_replay_count >= max_replays_per_worker:
                            break
                        
                        replay = {k: batch[k][i] for k in batch}
                        
                        if not self._is_valid_replay(replay):
                            continue
                        
                        valid_replay_count += 1
                        
                        # Yield both perspectives as separate tasks
                        # Randomize order of players to avoid bias
                        if torch.rand(1).item() < 0.5:
                            yield replay, 0
                            yield replay, 1
                        else:
                            yield replay, 1
                            yield replay, 0
        else:
            per_worker = self.num_replays // num_workers
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker
            if worker_id == num_workers - 1:
                end_idx = self.num_replays
            
            while True:
                indices = torch.arange(start_idx, end_idx)
                perm = torch.randperm(len(indices))
                shuffled_indices = indices[perm].tolist()
                
                for idx in shuffled_indices:
                    replay = self.df.iloc[idx].to_dict()
                    if not self._is_valid_replay(replay):
                        continue
                    
                    # Yield both perspectives
                    if torch.rand(1).item() < 0.5:
                        yield replay, 0
                        yield replay, 1
                    else:
                        yield replay, 1
                        yield replay, 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        
        if worker_info is None:
            w_id = 0
            w_num = 1
        else:
            w_id = worker_info.id
            w_num = worker_info.num_workers
            
        # Each worker produces a sub-batch
        assert self.batch_size % w_num == 0, "Batch size must be divisible by num_workers"
        worker_batch_size = self.batch_size // w_num
        
        print(f"Worker {w_id} starting. Batch size: {self.batch_size}, Worker batch size: {worker_batch_size}")
        
        # Create replay source
        replay_source = self._get_replay_iterator(w_id, w_num)
        
        # Initialize streams
        # streams[i] is a generator yielding chunks from a single (replay, player) tuple
        streams = [None] * worker_batch_size
        
        print(f"Worker {w_id}: Initialized. Waiting for data...")
        
        first_batch = True
        
        while True:
            batch_obs = []
            batch_memory = []
            batch_actions = []
            reset_mask = [] 
            
            iterator = range(worker_batch_size)
            if first_batch:
                iterator = tqdm(iterator, desc=f"Worker {w_id} Init", position=w_id, leave=False)
            
            for i in iterator:
                # Ensure we get a sample for this slot
                while True:
                    is_new_game = False
                    
                    if streams[i] is None:
                        try:
                            replay, target_player = next(replay_source)
                            streams[i] = self._extract_samples_from_replay(replay, target_player)
                            is_new_game = True
                        except StopIteration:
                            continue
                    
                    try:
                        sample = next(streams[i])
                        obs, memory, action, _ = sample
                        
                        batch_obs.append(obs)
                        batch_memory.append(memory)
                        batch_actions.append(action)
                        
                        # If we just started a new game (or switched to a new one), reset mask is True
                        # Note: if we had a stream, it finished, and we got a new one in the same slot,
                        # is_new_game will be True, which is correct.
                        reset_mask.append(is_new_game)
                        break # Successfully filled slot i
                        
                    except StopIteration:
                        streams[i] = None
                        # Loop continues to get a new replay for this slot
            
            if len(batch_obs) == worker_batch_size:
                first_batch = False
                # print(f"Worker {w_id}: Yielding batch")
                yield (
                    torch.from_numpy(np.stack(batch_obs)).float(),
                    torch.from_numpy(np.stack(batch_memory)).float(),
                    torch.from_numpy(np.stack(batch_actions)).long(),
                    torch.tensor(reset_mask, dtype=torch.bool),
                    torch.tensor(w_id, dtype=torch.long)
                )
    
    def _is_valid_replay(self, replay: Dict) -> bool:
        if len(replay['moves']) > self.max_turns:
            return False
        
        if max(replay['stars']) < self.min_stars:
            return False
        
        return True
    
    def _extract_samples_from_replay(self, replay: Dict, target_player_idx: int) -> Iterator[Tuple]:
        width = replay['mapWidth']
        height = replay['mapHeight']
        
        grid = self._create_initial_grid(replay, height, width)
        
        try:
            game = Game(grid, ["player_0", "player_1"])
        except Exception:
            return
        
        # Initialize memory augmentation for both players
        memory_0 = MemoryAugmentation((self.grid_size, self.grid_size))
        memory_1 = MemoryAugmentation((self.grid_size, self.grid_size))
        
        # Buffer for sequential yielding (only for target player)
        buffer = {'obs': [], 'memory': [], 'action': []}
        
        for move in replay['moves']:
            if len(move) < 5:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            turn_number = move[4]
            
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
            
            # Only process if the player meets the star requirement AND matches target
            if replay['stars'][player_idx] >= self.min_stars and player_idx == target_player_idx:
                if player_idx == 0:
                    obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                    memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                    
                    buffer['obs'].append(obs_tensor)
                    buffer['memory'].append(memory_features)
                    buffer['action'].append(action.copy())
                else:
                    obs_tensor = obs_1.as_tensor().astype(np.float32, copy=True)
                    memory_features = memory_1.get_memory_features().astype(np.float32, copy=True)
                    
                    buffer['obs'].append(obs_tensor)
                    buffer['memory'].append(memory_features)
                    buffer['action'].append(action.copy())
                
                # If buffer is full, yield the sequence and clear it
                if len(buffer['obs']) >= self.sequence_len:
                    # print(f"DEBUG: Yielding sequence for player {player_idx}")
                    yield (
                        np.stack(buffer['obs']),    # (Seq, C, H, W)
                        np.stack(buffer['memory']), # (Seq, C, H, W)
                        np.stack(buffer['action']), # (Seq, 5)
                        player_idx
                    )
                    # Clear buffer to start collecting the next chunk
                    buffer['obs'] = []
                    buffer['memory'] = []
                    buffer['action'] = []

            actions = {
                "player_0": np.array([1, 0, 0, 0, 0], dtype=np.int8),
                "player_1": np.array([1, 0, 0, 0, 0], dtype=np.int8),
            }
            
            if player_idx == 0:
                actions["player_0"] = action
            else:
                actions["player_1"] = action
            
            game.step(actions)
            
            # Update memory augmentation after step
            obs_0_after = game.agent_observation("player_0")
            obs_1_after = game.agent_observation("player_1")
            obs_0_after.pad_observation(pad_to=self.grid_size)
            obs_1_after.pad_observation(pad_to=self.grid_size)
            
            # Convert observations to dict format for memory update
            obs_0_curr = self._obs_to_dict(obs_0_after)
            obs_1_curr = self._obs_to_dict(obs_1_after)
            
            memory_0.update(obs_0_curr, actions["player_0"])
            memory_1.update(obs_1_curr, actions["player_1"])
    
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
            'owned_land_count': tensor[9],
            'owned_army_count': tensor[10],
            'opponent_land_count': tensor[11],
            'opponent_army_count': tensor[12],
            'timestep': tensor[13],
            'priority': tensor[14],
        }

def create_iterable_dataloader(
    data_dir: str,
    batch_size: int = 32,
    grid_size: int = 24,
    num_workers: int = 0,
    max_replays: int | None = None,
    min_stars: int = 70,
    max_turns: int = 500,
    sequence_len: int = 32,
) -> DataLoader:
    dataset = GeneralsReplayIterableDataset(
        data_dir=data_dir,
        grid_size=grid_size,
        max_replays=max_replays,
        min_stars=min_stars,
        max_turns=max_turns,
        sequence_len=sequence_len,
        batch_size=batch_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=None, # Disable auto-batching, dataset yields batches
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

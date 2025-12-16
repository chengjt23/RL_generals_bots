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
        
        import pyarrow.parquet as pq
        self.parquet_file = pq.ParquetFile(
            self.data_dir / "data" / "train-00000-of-00001.parquet"
        )
        
        self.num_row_groups = self.parquet_file.num_row_groups
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        worker_info = get_worker_info()
        
        print("worker info:", worker_info)
        input()
        if worker_info is None:
            w_id = 0
            w_num = 1
        else:
            w_id = worker_info.id
            w_num = worker_info.num_workers
        
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
        
        for move in replay['moves']:
            if len(move) < 5:
                continue
            
            player_idx = move[0]
            start_tile = move[1]
            end_tile = move[2]
            is_half = move[3]
            
            obs_0 = game.agent_observation("player_0")
            obs_1 = game.agent_observation("player_1")
            
            # Pad and cache previous observations for inference
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
            action_pass = np.array([1, 0, 0, 0, 0], dtype=np.int8)
            
            # Get current memory features before taking action
            if player_idx == 0:
                obs_tensor = obs_0.as_tensor().astype(np.float32, copy=True)
                memory_features = memory_0.get_memory_features().astype(np.float32, copy=True)
                yield (obs_tensor, memory_features, action.copy(), 0)
            else:
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
            
            try:
                game.step(actions)
                
                # Update memory augmentation after step
                obs_0_after = game.agent_observation("player_0")
                obs_1_after = game.agent_observation("player_1")
                obs_0_after.pad_observation(pad_to=self.grid_size)
                obs_1_after.pad_observation(pad_to=self.grid_size)
                
                # Convert observations to dict format for memory update
                obs_0_curr = self._obs_to_dict(obs_0_after)
                obs_1_curr = self._obs_to_dict(obs_1_after)
                
                # Infer opponent actions
                opp_action_for_0 = self._infer_opponent_action(obs_0_prev, obs_0_curr)
                opp_action_for_1 = self._infer_opponent_action(obs_1_prev, obs_1_curr)
                
                memory_0.update(obs_0_curr, actions["player_0"], opp_action_for_0)
                memory_1.update(obs_1_curr, actions["player_1"], opp_action_for_1)
            except Exception:
                break

    def _infer_opponent_action(self, prev: dict, curr: dict) -> np.ndarray:
        """Heuristic opponent action inference matching SOTAAgent."""
        fog = curr['fog_cells']
        sif = curr['structures_in_fog']
        visible = (fog == 0) & (sif == 0)

        prev_opp = prev['opponent_cells']
        cur_opp = curr['opponent_cells']

        # Newly visible opponent-owned cells (candidate destination).
        new_opp = visible & (cur_opp.astype(bool)) & (~prev_opp.astype(bool))
        new_positions = np.argwhere(new_opp)
        
        # Default to PASS
        action = np.array([1, 0, 0, 0, 0], dtype=np.int8)

        if new_positions.shape[0] != 1:
            return action

        dest_r, dest_c = (int(new_positions[0][0]), int(new_positions[0][1]))

        # Candidate sources: adjacent cells that were opponent-owned previously.
        for direction, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            src_r, src_c = dest_r - dr, dest_c - dc
            if src_r < 0 or src_c < 0 or src_r >= self.grid_size or src_c >= self.grid_size:
                continue
            if prev_opp[src_r, src_c] == 1:
                # Found a plausible source
                # Action format: [pass, row, col, direction, split]
                # to_pass=False -> 0
                return np.array([0, src_r, src_c, direction, 0], dtype=np.int8)

        return action
    
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


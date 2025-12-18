import numpy as np
from collections import deque


class MemoryAugmentation:
    def __init__(self, grid_shape: tuple[int, int], history_length: int = 7):
        self.grid_shape = grid_shape
        self.history_length = history_length
        self.reset()

    def reset(self):
        self.discovered_castles = np.zeros(self.grid_shape, dtype=np.float32)
        self.discovered_generals = np.zeros(self.grid_shape, dtype=np.float32)
        self.discovered_mountains = np.zeros(self.grid_shape, dtype=np.float32)
        self.last_seen_armies = np.zeros(self.grid_shape, dtype=np.float32)
        self.explored_cells = np.zeros(self.grid_shape, dtype=np.float32)
        self.opponent_visible_cells = np.zeros(self.grid_shape, dtype=np.float32)
        self.action_history = deque(maxlen=self.history_length * 2)

    def clone(self):
        new_mem = MemoryAugmentation(self.grid_shape, self.history_length)
        new_mem.discovered_castles = self.discovered_castles.copy()
        new_mem.discovered_generals = self.discovered_generals.copy()
        new_mem.discovered_mountains = self.discovered_mountains.copy()
        new_mem.last_seen_armies = self.last_seen_armies.copy()
        new_mem.explored_cells = self.explored_cells.copy()
        new_mem.opponent_visible_cells = self.opponent_visible_cells.copy()
        new_mem.action_history = deque(self.action_history, maxlen=self.history_length * 2)
        return new_mem

    def update(self, observation: dict, action_agent, action_opponent):
        obs_shape = observation["fog_cells"].shape
        if obs_shape != self.grid_shape:
            self.grid_shape = obs_shape
            self.reset()
        
        visible_mask = 1 - observation["fog_cells"] - observation["structures_in_fog"]
        
        castles_mask = observation["cities"]
        generals_mask = observation["generals"]
        mountains_mask = observation["mountains"]
        
        self.discovered_castles = np.maximum(self.discovered_castles, castles_mask * visible_mask)
        self.discovered_generals = np.maximum(self.discovered_generals, generals_mask * visible_mask)
        self.discovered_mountains = np.maximum(self.discovered_mountains, mountains_mask)
        
        self.last_seen_armies = np.where(visible_mask > 0, observation["armies"], self.last_seen_armies)
        
        self.explored_cells = np.maximum(self.explored_cells, visible_mask)
        
        opponent_mask = observation["opponent_cells"]
        self.opponent_visible_cells = np.maximum(self.opponent_visible_cells, opponent_mask)
        
        agent_action_array = self._action_to_array(action_agent)
        opponent_action_array = self._action_to_array(action_opponent)
        self.action_history.append((agent_action_array, opponent_action_array))

    def get_memory_features(self, out: np.ndarray = None) -> np.ndarray:
        n_channels = 6 + self.history_length * 2
        if out is None:
            out = np.zeros((n_channels, *self.grid_shape), dtype=np.float32)
        
        out[0] = self.discovered_castles
        out[1] = self.discovered_generals
        out[2] = self.discovered_mountains
        out[3] = self.last_seen_armies
        out[4] = self.explored_cells
        out[5] = self.opponent_visible_cells
        
        base_idx = 6
        for i in range(self.history_length):
            if i < len(self.action_history):
                agent_action, opponent_action = self.action_history[-(i+1)]
                self._action_to_map_inplace(agent_action, out[base_idx + i * 2])
                self._action_to_map_inplace(opponent_action, out[base_idx + i * 2 + 1])
            else:
                out[base_idx + i * 2].fill(0)
                out[base_idx + i * 2 + 1].fill(0)
        
        return out
    
    def _action_to_map_inplace(self, action: np.ndarray, out: np.ndarray):
        out.fill(0)
        if action[0] == 0:
            row, col = int(action[1]), int(action[2])
            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                out[row, col] = int(action[3]) + 1

    def _action_to_array(self, action):
        if isinstance(action, np.ndarray):
            return action.copy()
        try:
            if action.row is not None:
                return np.array([0, action.row, action.col, action.direction, int(action.split)], dtype=np.int8)
        except (AttributeError, TypeError):
            pass
        return np.array([1, 0, 0, 0, 0], dtype=np.int8)
    
    def _action_to_map(self, action: np.ndarray) -> np.ndarray:
        action_map = np.zeros(self.grid_shape, dtype=np.float32)
        if action[0] == 0:
            row, col = int(action[1]), int(action[2])
            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                direction = int(action[3])
                action_map[row, col] = direction + 1
        return action_map


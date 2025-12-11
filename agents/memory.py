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
        self.explored_cells = np.zeros(self.grid_shape, dtype=np.float32)
        self.opponent_visible_cells = np.zeros(self.grid_shape, dtype=np.float32)
        self.action_history = deque(maxlen=self.history_length * 2)

    def update(self, observation: dict, action_agent, action_opponent):
        visible_mask = 1 - observation["fog_cells"] - observation["structures_in_fog"]
        
        castles_mask = observation["cities"]
        generals_mask = observation["generals"]
        self.discovered_castles = np.maximum(self.discovered_castles, castles_mask * visible_mask)
        self.discovered_generals = np.maximum(self.discovered_generals, generals_mask * visible_mask)
        
        self.explored_cells = np.maximum(self.explored_cells, observation["owned_cells"])
        
        opponent_mask = observation["opponent_cells"]
        self.opponent_visible_cells = np.maximum(self.opponent_visible_cells, opponent_mask)
        
        agent_action_array = self._action_to_array(action_agent)
        opponent_action_array = self._action_to_array(action_opponent)
        self.action_history.append((agent_action_array, opponent_action_array))

    def get_memory_features(self) -> np.ndarray:
        memory_channels = [
            self.discovered_castles,
            self.discovered_generals,
            self.explored_cells,
            self.opponent_visible_cells,
        ]
        
        for i in range(self.history_length):
            if i < len(self.action_history):
                agent_action, opponent_action = self.action_history[-(i+1)]
                agent_action_map = self._action_to_map(agent_action)
                opponent_action_map = self._action_to_map(opponent_action)
            else:
                agent_action_map = np.zeros(self.grid_shape, dtype=np.float32)
                opponent_action_map = np.zeros(self.grid_shape, dtype=np.float32)
            memory_channels.append(agent_action_map)
            memory_channels.append(opponent_action_map)
        
        return np.stack(memory_channels, axis=0)

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


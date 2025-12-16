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
        self.last_seen_armies = np.zeros(self.grid_shape, dtype=np.float32)
        self.last_seen_turn = np.zeros(self.grid_shape, dtype=np.float32)
        self.current_turn = 0
        self.action_history = deque(maxlen=self.history_length * 2)

    def update(self, observation: dict, action_agent: np.ndarray, action_opponent: np.ndarray, turn_step: int = 0):
        self.current_turn = turn_step
        visible_mask = 1 - observation["fog_cells"] - observation["structures_in_fog"]
        
        castles_mask = observation["cities"]
        generals_mask = observation["generals"]
        self.discovered_castles = np.maximum(self.discovered_castles, castles_mask * visible_mask)
        self.discovered_generals = np.maximum(self.discovered_generals, generals_mask * visible_mask)
        
        self.explored_cells = np.maximum(self.explored_cells, observation["owned_cells"])
        
        opponent_mask = observation["opponent_cells"]
        self.opponent_visible_cells = np.maximum(self.opponent_visible_cells, opponent_mask)
        
        # Update last seen armies
        if "armies" in observation:
            current_armies = observation["armies"]
            # Only update visible cells
            self.last_seen_armies = np.where(visible_mask > 0, current_armies, self.last_seen_armies)
            self.last_seen_turn = np.where(visible_mask > 0, turn_step, self.last_seen_turn)
        
        self.action_history.append((action_agent.copy(), action_opponent.copy()))

    def get_memory_features(self) -> np.ndarray:
        # Calculate projected armies
        # Growth logic: Enemy Cities and Generals grow by 0.5 per turn (1 per 2 turns)
        # We use opponent_visible_cells as a proxy for enemy ownership history
        # Note: This is a heuristic. Ideally we should track ownership more precisely.
        growth_mask = (self.discovered_castles * self.opponent_visible_cells) + self.discovered_generals
        growth_mask = np.clip(growth_mask, 0, 1)
        
        turns_passed = self.current_turn - self.last_seen_turn
        projected_growth = turns_passed * 0.5 * growth_mask
        projected_armies = self.last_seen_armies + projected_growth
        
        memory_channels = [
            self.discovered_castles,
            self.discovered_generals,
            self.explored_cells,
            self.opponent_visible_cells,
            projected_armies,
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

    def _action_to_map(self, action: np.ndarray) -> np.ndarray:
        action_map = np.zeros(self.grid_shape, dtype=np.float32)
        if action[0] == 0:
            row, col = int(action[1]), int(action[2])
            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                direction = int(action[3])
                action_map[row, col] = direction + 1
        return action_map


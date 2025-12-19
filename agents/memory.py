import numpy as np


class MemoryAugmentation:
    def __init__(self, grid_shape: tuple[int, int], decay: float = 0.95):
        self.grid_shape = grid_shape
        self.decay = decay
        self.reset()

    def reset(self):
        self.discovered_castles = np.zeros(self.grid_shape, dtype=np.float32)
        self.discovered_generals = np.zeros(self.grid_shape, dtype=np.float32)
        self.discovered_mountains = np.zeros(self.grid_shape, dtype=np.float32)
        self.last_seen_armies = np.zeros(self.grid_shape, dtype=np.float32)
        self.explored_cells = np.zeros(self.grid_shape, dtype=np.float32)
        self.opponent_visible_cells = np.zeros(self.grid_shape, dtype=np.float32)
        
        # New features
        self.discovered_city_armies = np.zeros(self.grid_shape, dtype=np.float32)
        self.last_seen_timestep = np.zeros(self.grid_shape, dtype=np.float32)
        
        # Action maps
        # 4 directional channels: UP, DOWN, LEFT, RIGHT
        self.action_overlays = np.zeros((4, *self.grid_shape), dtype=np.float32)
        # 1 visit channel
        self.action_visit = np.zeros(self.grid_shape, dtype=np.float32)

    def update(self, observation: dict, action_agent: np.ndarray):
        visible_mask = 1 - observation["fog_cells"] - observation["structures_in_fog"]
        
        castles_mask = observation["cities"]
        generals_mask = observation["generals"]
        mountains_mask = observation["mountains"]
        armies = observation["armies"]
        
        # Update static/semi-static features with "last seen" logic
        self.discovered_castles = np.where(visible_mask, castles_mask, self.discovered_castles)
        self.discovered_generals = np.where(visible_mask, generals_mask, self.discovered_generals)
        self.discovered_mountains = np.where(visible_mask, mountains_mask, self.discovered_mountains)
        
        self.last_seen_armies = np.where(visible_mask, armies, self.last_seen_armies)
        
        # Update discovered city armies
        self.discovered_city_armies = np.where(visible_mask, castles_mask * armies, self.discovered_city_armies)
        
        self.explored_cells = np.maximum(self.explored_cells, visible_mask)
        
        opponent_mask = observation["opponent_cells"]
        self.opponent_visible_cells = np.where(visible_mask, opponent_mask, self.opponent_visible_cells)
        
        # Update last seen timestep
        # Update last seen timestep (age: 0 = just seen, N = N turns since last seen)
        self.last_seen_timestep = np.where(visible_mask, 0, self.last_seen_timestep + 1)
        
        # Update action maps
        # Decay
        self.action_overlays *= self.decay
        self.action_visit *= self.decay
        
        # Add new action
        if action_agent[0] == 0: # Move action
            row, col = int(action_agent[1]), int(action_agent[2])
            direction = int(action_agent[3])
            
            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                if 0 <= direction < 4:
                    self.action_overlays[direction, row, col] += 1.0
                self.action_visit[row, col] += 1.0

    def get_memory_features(self) -> np.ndarray:
        memory_channels = [
            self.discovered_castles,
            self.discovered_generals,
            self.discovered_mountains,
            self.last_seen_armies,
            self.explored_cells,
            self.opponent_visible_cells,
            self.discovered_city_armies,
            self.last_seen_timestep,
            self.action_visit,
        ]
        
        # Stack everything: 8 static + 1 visit + 4 directional = 13 channels
        # Note: action_overlays is (4, H, W), others are (H, W)
        
        feature_list = []
        for feat in memory_channels:
            feature_list.append(feat)
            
        # Add directional overlays
        for i in range(4):
            feature_list.append(self.action_overlays[i])

        # Final Channels: 
        """
        0: discovered_castles
        1: discovered_generals
        2: discovered_mountains
        3: last_seen_armies
        4: explored_cells
        5: opponent_visible_cells
        6: discovered_city_armies
        7: last_seen_timestep
        8: action_visit
        9-12: action_overlays (UP, DOWN, LEFT, RIGHT)
        """
            
        return np.stack(feature_list, axis=0)

import numpy as np
import torch

from generals.agents import Agent
from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

from .network import SOTANetwork
from .memory import MemoryAugmentation


class SOTAAgent(Agent):
    def __init__(
        self,
        sota_config,
        id: str = "SOTA",
        grid_size: int = 24,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: str | None = None,
        memory_channels: int = 18,
    ):
        super().__init__(id)
        assert sota_config is not None, "sota_config must be provided"
        assert sota_config['grid_size'] == grid_size, "grid_size in sota_config must match the provided grid_size"
        self.grid_size = grid_size
        self.device = torch.device(device)
        
        self.network = SOTANetwork(
            obs_channels=sota_config['obs_channels'],
            memory_channels=sota_config['memory_channels'],
            grid_size=sota_config['grid_size'],
            base_channels=sota_config['base_channels'],
        ).to(self.device)
        
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            else:
                state = ckpt
            self.network.load_state_dict(state)
        
        self.network.eval()
        
        self.memory = MemoryAugmentation((grid_size, grid_size))
        self.last_action = None
    
    def reset(self):
        self.memory.reset()
        self.last_action = None
    
    def act(self, observation: Observation) -> Action:
        # Keep shapes consistent with training: training pads BEFORE calling MemoryAugmentation.update().
        observation.pad_observation(pad_to=self.grid_size)

        # Update memory with the *current* observation and the *previous* actions.
        if self.last_action is not None:
            action_array = np.array([1, 0, 0, 0, 0], dtype=np.int8)
            try:
                if self.last_action.row is not None:
                    action_array[0] = 0
                    action_array[1] = self.last_action.row
                    action_array[2] = self.last_action.col
                    action_array[3] = self.last_action.direction
                    action_array[4] = int(self.last_action.split)
            except (AttributeError, TypeError):
                pass
            self.memory.update(
                self._obs_to_dict(observation),
                action_array,
            )
        
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            policy_logits, _ = self.network(obs_tensor, memory_tensor)
        
        policy_logits = policy_logits.squeeze(0)
        
        action = self._sample_action(policy_logits, observation)
        self.last_action = action
        
        return action
    

    """
    
Key	Shape	Description
0armies	(N,M)	Number of units in a visible cell regardless of the owner
1generals	(N,M)	Mask indicating visible cells containing a general
2cities	(N,M)	Mask indicating visible cells containing a city
3mountains	(N,M)	Mask indicating visible cells containing mountains
4neutral_cells	(N,M)	Mask indicating visible cells that are not owned by any agent
5owned_cells	(N,M)	Mask indicating visible cells owned by the agent
6opponent_cells	(N,M)	Mask indicating visible cells owned by the opponent
7fog_cells	(N,M)	Mask indicating fog cells that are not mountains or cities
8structures_in_fog	(N,M)	Mask showing cells containing either cities or mountains in fog

    """
    
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
            'timestep': tensor[13],
        }
    
    def _prepare_observation(self, obs: Observation) -> torch.Tensor:
        obs.pad_observation(pad_to=self.grid_size)
        tensor = obs.as_tensor().astype(np.float32)
        tensor = np.log1p(tensor)
        obs_tensor = torch.from_numpy(tensor).float()
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
        return obs_tensor
    
    def _prepare_memory(self) -> torch.Tensor:
        memory_features = self.memory.get_memory_features()
        memory_features = np.log1p(memory_features)
        memory_tensor = torch.from_numpy(memory_features).float()
        memory_tensor = memory_tensor.unsqueeze(0).to(self.device)
        return memory_tensor
    
    def _sample_action(self, policy_logits: torch.Tensor, observation: Observation) -> Action:
        valid_mask = compute_valid_move_mask(observation)
        
        policy_logits_np = policy_logits.cpu().numpy()
        
        h, w = valid_mask.shape[:2]
        
        pass_logit = policy_logits_np[0, 0, 0]
        
        action_logits = policy_logits_np[1:9].reshape(4, 2, h, w)
        
        masked_logits = []
        valid_actions = []
        
        for direction in range(4):
            for split in range(2):
                logits_slice = action_logits[direction, split]
                mask_slice = valid_mask[:, :, direction]
                
                valid_positions = np.argwhere(mask_slice > 0)
                for pos in valid_positions:
                    row, col = pos
                    masked_logits.append(logits_slice[row, col])
                    valid_actions.append((row, col, direction, split))
        
        if len(masked_logits) == 0:
            return Action(to_pass=True)
        
        all_logits = np.array([pass_logit] + masked_logits)
        probs = torch.softmax(torch.from_numpy(all_logits), dim=0).numpy()
        
        choice = np.argmax(probs)
        
        if choice == 0:
            return Action(to_pass=True)
        
        row, col, direction, split = valid_actions[choice - 1]
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))

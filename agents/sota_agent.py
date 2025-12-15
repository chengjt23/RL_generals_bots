import numpy as np
import torch

from generals.agents import Agent
from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

from .network import SOTANetwork


class SOTAAgent(Agent):
    """
    SOTA Agent with RNN-based memory encoding.
    
    The agent uses a recurrent neural network to maintain memory across timesteps,
    replacing the previous hand-crafted memory augmentation system.
    """
    
    def __init__(
        self,
        sota_config,
        id: str = "SOTA",
        grid_size: int = 24,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: str | None = None,
    ):
        super().__init__(id)
        assert sota_config is not None, "sota_config must be provided"
        assert sota_config['grid_size'] == grid_size, "grid_size in sota_config must match the provided grid_size"
        self.grid_size = grid_size
        self.device = torch.device(device)
        
        # Create network with RNN memory encoder
        self.network = SOTANetwork(
            obs_channels=sota_config['obs_channels'],
            memory_channels=sota_config['memory_channels'],
            grid_size=sota_config['grid_size'],
            base_channels=sota_config['base_channels'],
            rnn_hidden_channels=sota_config.get('rnn_hidden_channels', 32),
            rnn_num_layers=sota_config.get('rnn_num_layers', 2),
            use_rnn_memory=sota_config.get('use_rnn_memory', True),
        ).to(self.device)
        
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            else:
                state = ckpt
            self.network.load_state_dict(state)
        
        self.network.eval()
        
        # RNN hidden state (replaces MemoryAugmentation)
        self.hidden_state = None
        self.last_action = None
        self._prev_obs_snapshot: dict[str, np.ndarray] | None = None
    
    def reset(self):
        """Reset agent state including RNN hidden state."""
        self.hidden_state = None
        self.last_action = None
        self._prev_obs_snapshot = None
    
    def act(self, observation: Observation) -> Action:
        """
        Select action using RNN-based memory.
        
        The RNN maintains hidden state across timesteps, learning to encode
        relevant memory information from the observation history.
        """
        # Keep shapes consistent with training
        observation.pad_observation(pad_to=self.grid_size)
        
        # Prepare observation tensor
        obs_tensor = self._prepare_observation(observation)
        
        # Forward pass with RNN memory encoder
        with torch.no_grad():
            policy_logits, _, self.hidden_state = self.network(
                obs_tensor,
                hidden_state=self.hidden_state,
                return_hidden=True
            )
        
        policy_logits = policy_logits.squeeze(0)
        
        # Sample action from policy
        action = self._sample_action(policy_logits, observation)
        self.last_action = action
        
        # Snapshot current observation for debugging/analysis
        self._prev_obs_snapshot = self._snapshot_observation(observation)
        
        return action
    
    def _prepare_observation(self, obs: Observation) -> torch.Tensor:
        """Convert observation to tensor format."""
        # `act()` already pads; keep this idempotent for safety.
        obs.pad_observation(pad_to=self.grid_size)
        obs_tensor = torch.from_numpy(obs.as_tensor()).float()
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
        return obs_tensor

    def _snapshot_observation(self, obs: Observation) -> dict[str, np.ndarray]:
        """Store observation snapshot for debugging/analysis."""
        return {
            'armies': np.asarray(obs.armies).copy(),
            'opponent_cells': np.asarray(obs.opponent_cells).copy(),
            'fog_cells': np.asarray(obs.fog_cells).copy(),
            'structures_in_fog': np.asarray(obs.structures_in_fog).copy(),
        }
    
    def _sample_action(self, policy_logits: torch.Tensor, observation: Observation) -> Action:
        """Sample action from policy logits, respecting valid move mask."""
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


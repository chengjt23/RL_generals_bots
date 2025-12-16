import numpy as np
import torch
import torch.nn.functional as F

from generals.agents import Agent
from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

from .network import SOTANetwork
from .memory import MemoryAugmentation


class PPOAgent(Agent):
    def __init__(
        self,
        sota_config,
        id: str = "PPO",
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
        
        self.memory = MemoryAugmentation((grid_size, grid_size), history_length=7)
        self.last_action = None
        self.opponent_last_action = Action(to_pass=True)
        self._prev_obs_snapshot: dict[str, np.ndarray] | None = None
    
    def reset(self):
        self.memory.reset()
        self.last_action = None
        self.opponent_last_action = Action(to_pass=True)
        self._prev_obs_snapshot = None
    
    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        observation.pad_observation(pad_to=self.grid_size)

        if self._prev_obs_snapshot is not None:
            self.opponent_last_action = self._infer_opponent_action(self._prev_obs_snapshot, observation)

        if self.last_action is not None:
            self.memory.update(
                self._obs_to_dict(observation),
                np.asarray(self.last_action, dtype=np.int8),
                np.asarray(self.opponent_last_action, dtype=np.int8),
            )
        
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            policy_logits, _ = self.network(obs_tensor, memory_tensor)
        
        policy_logits = policy_logits.squeeze(0)
        
        if deterministic:
            action = self._sample_action_deterministic(policy_logits, observation)
        else:
            action = self._sample_action(policy_logits, observation)
        
        self.last_action = action
        self._prev_obs_snapshot = self._snapshot_observation(observation)
        
        return action
    
    def act_with_value(self, observation: Observation):
        observation.pad_observation(pad_to=self.grid_size)

        if self._prev_obs_snapshot is not None:
            self.opponent_last_action = self._infer_opponent_action(self._prev_obs_snapshot, observation)

        if self.last_action is not None:
            self.memory.update(
                self._obs_to_dict(observation),
                np.asarray(self.last_action, dtype=np.int8),
                np.asarray(self.opponent_last_action, dtype=np.int8),
            )
        
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            policy_logits, value = self.network(obs_tensor, memory_tensor)
        
        policy_logits = policy_logits.squeeze(0)
        value = value.squeeze(0)
        
        action, log_prob = self._sample_action_with_log_prob(policy_logits, observation)
        
        self.last_action = action
        self._prev_obs_snapshot = self._snapshot_observation(observation)
        
        return action, log_prob, value.item()
    
    def get_value(self, observation: Observation):
        observation.pad_observation(pad_to=self.grid_size)
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            _, value = self.network(obs_tensor, memory_tensor)
        
        return value.squeeze(0).item()
    
    def evaluate_actions(self, obs_batch, memory_batch, action_batch):
        policy_logits, values = self.network(obs_batch, memory_batch)
        
        batch_size = obs_batch.shape[0]
        h, w = policy_logits.shape[2], policy_logits.shape[3]
        
        log_probs = []
        entropies = []
        
        for i in range(batch_size):
            action = action_batch[i]
            logits = policy_logits[i]
            obs_tensor = obs_batch[i]
            
            obs = Observation.from_numpy_array(obs_tensor.cpu().numpy())
            valid_mask = compute_valid_move_mask(obs)
            
            pass_logit = logits[0, 0, 0]
            
            all_logits_list = [pass_logit]
            action_positions = []
            
            for direction in range(4):
                for split in range(2):
                    action_idx = 1 + direction * 2 + split
                    mask_slice = valid_mask[:, :, direction]
                    valid_positions = torch.from_numpy(mask_slice).nonzero()
                    
                    for pos in valid_positions:
                        row_pos, col_pos = pos[0].item(), pos[1].item()
                        action_logit = logits[action_idx, row_pos, col_pos]
                        all_logits_list.append(action_logit)
                        action_positions.append((row_pos, col_pos, direction, split))
            
            all_logits = torch.stack(all_logits_list)
            probs = F.softmax(all_logits, dim=0)
            log_probs_tensor = F.log_softmax(all_logits, dim=0)
            
            pass_flag = action[0].item()
            if pass_flag == 1:
                chosen_idx = 0
            else:
                row = action[1].item()
                col = action[2].item()
                direction = action[3].item()
                split = action[4].item()
                
                chosen_idx = 0
                for idx, (r, c, d, s) in enumerate(action_positions):
                    if r == row and c == col and d == direction and s == split:
                        chosen_idx = idx + 1
                        break
                
                if chosen_idx == 0:
                    raise ValueError(
                        f"Action not found in valid actions: row={row}, col={col}, "
                        f"direction={direction}, split={split}. "
                        f"Available actions: {len(action_positions)}"
                    )
            
            log_prob = log_probs_tensor[chosen_idx]
            entropy = -(probs * log_probs_tensor).sum()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = values.squeeze(-1)
        
        return log_probs, values, entropies
    
    def _obs_to_dict(self, obs: Observation) -> dict:
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
    
    def _prepare_observation(self, obs: Observation) -> torch.Tensor:
        obs.pad_observation(pad_to=self.grid_size)
        obs_tensor = torch.from_numpy(obs.as_tensor()).float()
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
        return obs_tensor

    def _snapshot_observation(self, obs: Observation) -> dict[str, np.ndarray]:
        return {
            'armies': np.asarray(obs.armies).copy(),
            'opponent_cells': np.asarray(obs.opponent_cells).copy(),
            'fog_cells': np.asarray(obs.fog_cells).copy(),
            'structures_in_fog': np.asarray(obs.structures_in_fog).copy(),
        }

    def _infer_opponent_action(self, prev: dict[str, np.ndarray], obs: Observation) -> Action:
        fog = np.asarray(obs.fog_cells)
        sif = np.asarray(obs.structures_in_fog)
        visible = (fog == 0) & (sif == 0)

        prev_opp = prev['opponent_cells']
        cur_opp = np.asarray(obs.opponent_cells)

        new_opp = visible & (cur_opp.astype(bool)) & (~prev_opp.astype(bool))
        new_positions = np.argwhere(new_opp)
        if new_positions.shape[0] != 1:
            return Action(to_pass=True)

        dest_r, dest_c = (int(new_positions[0][0]), int(new_positions[0][1]))

        for direction, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            src_r, src_c = dest_r - dr, dest_c - dc
            if src_r < 0 or src_c < 0 or src_r >= self.grid_size or src_c >= self.grid_size:
                continue
            if prev_opp[src_r, src_c] == 1:
                return Action(to_pass=False, row=src_r, col=src_c, direction=direction, to_split=False)

        return Action(to_pass=True)
    
    def _prepare_memory(self) -> torch.Tensor:
        memory_features = self.memory.get_memory_features()
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
        
        choice = np.random.choice(len(probs), p=probs)
        
        if choice == 0:
            return Action(to_pass=True)
        
        row, col, direction, split = valid_actions[choice - 1]
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))
    
    def _sample_action_deterministic(self, policy_logits: torch.Tensor, observation: Observation) -> Action:
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
        
        choice = np.argmax(all_logits)
        
        if choice == 0:
            return Action(to_pass=True)
        
        row, col, direction, split = valid_actions[choice - 1]
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))

    def _sample_action_with_log_prob(self, policy_logits: torch.Tensor, observation: Observation):
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
            return Action(to_pass=True), 0.0
        
        all_logits = np.array([pass_logit] + masked_logits)
        probs = torch.softmax(torch.from_numpy(all_logits), dim=0).numpy()
        log_probs = torch.log_softmax(torch.from_numpy(all_logits), dim=0).numpy()
        
        choice = np.random.choice(len(probs), p=probs)
        log_prob = log_probs[choice]
        
        if choice == 0:
            return Action(to_pass=True), log_prob
        
        row, col, direction, split = valid_actions[choice - 1]
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split)), log_prob


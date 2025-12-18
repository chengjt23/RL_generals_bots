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
        self._debug_print_count = 0
    
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
    
    def act_with_value_batch(self, observations, memories, prev_obs_snapshots=None, last_actions=None, opponent_last_actions=None):
        """
        Batch version of act_with_value that processes multiple observations.
        
        Args:
            observations: List of Observation objects
            memories: List of MemoryAugmentation objects (will be updated in-place)
            prev_obs_snapshots: Optional list of previous observation snapshots for opponent action inference
            last_actions: Optional list of last actions for memory update
            opponent_last_actions: Optional list of opponent's last actions for memory update
        
        Returns:
            actions: List of Action objects
            log_probs: List of log probabilities (float)
            values: List of state values (float)
        """
        batch_size = len(observations)
        
        # Update memories before forward pass
        # Note: opponent_last_actions should already be inferred in collect_trajectories
        if last_actions is not None:
            for i in range(batch_size):
                memory = memories[i]
                last_action = last_actions[i]
                obs = observations[i]
                
                # Pad observation before updating memory to ensure consistent grid_shape
                obs.pad_observation(pad_to=self.grid_size)
                
                # Use provided opponent_last_actions if available, otherwise infer
                if opponent_last_actions is not None and i < len(opponent_last_actions):
                    opponent_action = opponent_last_actions[i]
                elif prev_obs_snapshots is not None and prev_obs_snapshots[i] is not None:
                    # Fallback: infer opponent action if not provided
                    opponent_action = self._infer_opponent_action(prev_obs_snapshots[i], obs)
                else:
                    opponent_action = Action(to_pass=True)
                
                # Update memory with current observation and previous actions
                if last_action is not None:
                    memory.update(
                        self._obs_to_dict(obs),
                        np.asarray(last_action, dtype=np.int8),
                        np.asarray(opponent_action, dtype=np.int8),
                    )
        
        obs_tensors = []
        memory_tensors = []
        
        for i in range(batch_size):
            obs = observations[i]
            obs.pad_observation(pad_to=self.grid_size)
            obs_tensor = torch.from_numpy(obs.as_tensor()).float()
            obs_tensors.append(obs_tensor)
            
            memory_tensor = torch.from_numpy(memories[i].get_memory_features()).float()
            memory_tensors.append(memory_tensor)
        
        obs_batch = torch.stack(obs_tensors).to(self.device)
        memory_batch = torch.stack(memory_tensors).to(self.device)
        
        with torch.no_grad():
            policy_logits_batch, values_batch = self.network(obs_batch, memory_batch)
        
        actions = []
        log_probs = []
        values = []
        
        for i in range(batch_size):
            policy_logits = policy_logits_batch[i]
            value = values_batch[i]
            
            action, log_prob = self._sample_action_with_log_prob(policy_logits, observations[i])
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
        
        return actions, log_probs, values
    
    def act_with_value_batch_fast(self, observations, memories, obs_buffer, mem_buffer, device,
                                   prev_obs_snapshots=None, last_actions=None, opponent_last_actions=None):
        batch_size = len(observations)
        
        if last_actions is not None:
            for i in range(batch_size):
                memory = memories[i]
                last_action = last_actions[i]
                obs = observations[i]
                obs.pad_observation(pad_to=self.grid_size)
                
                if opponent_last_actions is not None and i < len(opponent_last_actions):
                    opponent_action = opponent_last_actions[i]
                elif prev_obs_snapshots is not None and prev_obs_snapshots[i] is not None:
                    opponent_action = self._infer_opponent_action(prev_obs_snapshots[i], obs)
                else:
                    opponent_action = Action(to_pass=True)
                
                if last_action is not None:
                    memory.update(
                        self._obs_to_dict(obs),
                        np.asarray(last_action, dtype=np.int8),
                        np.asarray(opponent_action, dtype=np.int8),
                    )
        
        obs_np_list = []
        mem_np_list = []
        for i in range(batch_size):
            obs = observations[i]
            obs_tensor = self._prepare_observation(obs)
            mem_tensor = self._prepare_memory_from_memory(memories[i])
            
            obs_buffer[i].copy_(obs_tensor.squeeze(0))
            mem_buffer[i].copy_(mem_tensor.squeeze(0))
            
            obs_np = obs_tensor.squeeze(0).cpu().numpy()
            mem_np = mem_tensor.squeeze(0).cpu().numpy()
            obs_np_list.append(obs_np)
            mem_np_list.append(mem_np)
        
        obs_batch = obs_buffer[:batch_size].to(device, non_blocking=True)
        mem_batch = mem_buffer[:batch_size].to(device, non_blocking=True)
        
        with torch.no_grad():
            policy_logits_batch, values_batch = self.network(obs_batch, mem_batch)
        
        actions = []
        log_probs = []
        values = []
        
        for i in range(batch_size):
            action, log_prob = self._sample_action_with_log_prob(policy_logits_batch[i], observations[i])
            actions.append(action)
            log_probs.append(log_prob)
            values.append(values_batch[i].item())
        
        return actions, log_probs, values, obs_np_list, mem_np_list
    
    def get_value(self, observation: Observation):
        observation.pad_observation(pad_to=self.grid_size)
        obs_tensor = self._prepare_observation(observation)
        memory_tensor = self._prepare_memory()
        
        with torch.no_grad():
            _, value = self.network(obs_tensor, memory_tensor)
        
        return value.squeeze(0).item()
    
    def evaluate_actions(self, obs_batch, memory_batch, action_batch, observations=None, valid_masks=None):
        policy_logits, values = self.network(obs_batch, memory_batch)
        
        # with torch.no_grad():
        #     p_max = policy_logits.max().item()
        #     p_min = policy_logits.min().item()
        #     p_range = p_max - p_min
        #     if p_range > 20:
        #         print(f"[NUMERICAL CHECK] Logit Range: {p_range:.2f} ({p_min:.2f} to {p_max:.2f}) -> OVER-SATURATED!")
        
        batch_size, channels, h, w = policy_logits.shape
        device = policy_logits.device
        
        pass_logits = policy_logits[:, 0:1, 0, 0]
        move_logits = policy_logits[:, 1:9, :, :].reshape(batch_size, -1)
        flat_logits = torch.cat([pass_logits, move_logits], dim=1)
        
        full_mask = torch.zeros(batch_size, 1 + 8 * h * w, dtype=torch.bool, device=device)
        full_mask[:, 0] = True
        
        if valid_masks is not None:
            full_mask[:, 1:] = valid_masks.reshape(batch_size, -1).bool()
        elif observations is not None:
            vm_list = [compute_valid_move_mask(o) for o in observations]
            vm_tensor = torch.from_numpy(np.stack(vm_list)).to(device).permute(0, 3, 1, 2)
            vm_tensor = vm_tensor.repeat_interleave(2, dim=1)
            full_mask[:, 1:] = vm_tensor.reshape(batch_size, -1).bool()
        
        flat_logits = torch.where(full_mask, flat_logits, torch.tensor(-1e9, device=device))
        
        log_probs_all = F.log_softmax(flat_logits, dim=1)
        probs_all = F.softmax(flat_logits, dim=1)
        
        is_pass = action_batch[:, 0].long()
        rows = action_batch[:, 1].long()
        cols = action_batch[:, 2].long()
        dirs = action_batch[:, 3].long()
        splits = action_batch[:, 4].long()
        
        action_indices = 1 + (dirs * 2 + splits) * (h * w) + rows * w + cols
        final_indices = torch.where(is_pass == 1, torch.zeros_like(action_indices), action_indices)
        selected_log_probs = log_probs_all.gather(1, final_indices.unsqueeze(1)).squeeze(1)
        
        entropy = -(probs_all * log_probs_all).sum(dim=1)
        
        return selected_log_probs, values.squeeze(-1), entropy
    
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
        tensor = obs.as_tensor().astype(np.float32)
        
        tensor = np.log1p(tensor)
        
        obs_tensor = torch.from_numpy(tensor).float()
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
        return obs_tensor

    def _snapshot_observation(self, obs: Observation) -> dict[str, np.ndarray]:
        return {
            'opponent_cells': np.array(obs.opponent_cells, dtype=np.bool_, copy=True),
        }

    def _infer_opponent_action(self, prev: dict[str, np.ndarray], obs: Observation) -> Action:
        prev_opp = prev['opponent_cells']
        cur_opp = np.asarray(obs.opponent_cells, dtype=np.bool_)
        fog = np.asarray(obs.fog_cells, dtype=np.bool_)
        sif = np.asarray(obs.structures_in_fog, dtype=np.bool_)
        visible = (~fog) & (~sif)
        
        new_opp = visible & cur_opp & (~prev_opp)
        positions = np.argwhere(new_opp)
        
        if positions.shape[0] == 0:
            return Action(to_pass=True)
        
        gs = self.grid_size
        
        for pos in positions:
            dest_r, dest_c = int(pos[0]), int(pos[1])
            for direction, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                src_r, src_c = dest_r - dr, dest_c - dc
                if 0 <= src_r < gs and 0 <= src_c < gs and prev_opp[src_r, src_c]:
                    return Action(to_pass=False, row=src_r, col=src_c, direction=direction, to_split=False)
        
        return Action(to_pass=True)
    
    def _prepare_memory(self) -> torch.Tensor:
        memory_features = self.memory.get_memory_features().astype(np.float32)
        
        memory_features = np.log1p(memory_features)
        
        memory_tensor = torch.from_numpy(memory_features).float()
        memory_tensor = memory_tensor.unsqueeze(0).to(self.device)
        return memory_tensor
    
    def _prepare_memory_from_memory(self, memory) -> torch.Tensor:
        memory_features = memory.get_memory_features().astype(np.float32)
        
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
        probs = torch.softmax(torch.from_numpy(all_logits), dim=0).numpy()
        
        top_probs, top_indices = torch.topk(torch.from_numpy(probs), k=3)
        print(f"Top 5 Actions Probabilities: {top_probs.tolist()}")
        print(f"Top 5 Actions Indices: {top_indices.tolist()}")
        
        choice = np.argmax(all_logits)
        
        if choice == 0:
            return Action(to_pass=True)
        
        row, col, direction, split = valid_actions[choice - 1]
        
        is_owned = observation.owned_cells[row, col]
        army_count = observation.armies[row, col]
        
        print(f"[CRITICAL DEBUG] Agent chose Move at ({row}, {col}). Owned? {is_owned}, Armies: {army_count}")
        
        if not is_owned or army_count <= 1:
            print("!!! ERROR: Mask failed! Agent chose an invalid cell. This is why it looks like")
        
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))

    def _sample_action_with_log_prob(self, policy_logits: torch.Tensor, observation: Observation):
        valid_mask = compute_valid_move_mask(observation)
        h, w = self.grid_size, self.grid_size
        device = policy_logits.device
        
        pass_logits = policy_logits[0:1, 0, 0]
        move_logits = policy_logits[1:9, :, :].reshape(-1)
        flat_logits = torch.cat([pass_logits, move_logits], dim=0)
        
        full_mask = torch.zeros(1 + 8 * h * w, dtype=torch.bool, device=device)
        full_mask[0] = True
        
        vm_tensor = torch.from_numpy(valid_mask).to(device).permute(2, 0, 1)
        vm_tensor = vm_tensor.repeat_interleave(2, dim=0)
        full_mask[1:] = vm_tensor.reshape(-1).bool()
        
        masked_logits = torch.where(full_mask, flat_logits, torch.tensor(-1e9, device=device))
        
        probs = torch.softmax(masked_logits, dim=0)
        log_probs = torch.log_softmax(masked_logits, dim=0)
        
        # top_probs, top_indices = torch.topk(probs, k=5)
        # print(f"Top 5 Actions Probabilities: {top_probs.tolist()}")
        # print(f"Top 5 Actions Indices: {top_indices.tolist()}")
        
        choice = torch.multinomial(probs, num_samples=1).item()
        selected_log_prob = log_probs[choice].item()
        
        if choice == 0:
            return Action(to_pass=True), selected_log_prob
        
        rem = choice - 1
        plane_size = h * w
        dir_and_split = rem // plane_size
        cell_idx = rem % plane_size
        row = cell_idx // w
        col = cell_idx % w
        direction = dir_and_split // 2
        split = dir_and_split % 2
        
        return Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split)), selected_log_prob


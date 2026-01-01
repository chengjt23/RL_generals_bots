import torch
import numpy as np
from env.gymnasium_generals import GymnasiumGenerals
from model.network import SOTANetwork
from model.memory import MemoryAugmentation
from generals.core.action import Action
from train_ppo import build_action_distribution

def debug_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup environment
    env = GymnasiumGenerals(agents=["player_0", "player_1"], render_mode=None)
    
    # Setup model
    model = SOTANetwork(
        obs_channels=15,
        memory_channels=13,
        grid_size=24
    ).to(device)
    model.eval()
    
    # Setup memory
    memories = {
        agent: MemoryAugmentation(
            grid_shape=(24, 24)
        ) for agent in env.agents
    }
    
    obs, infos = env.reset()
    
    # Run a few steps
    for step in range(10):
        print(f"Step {step}")
        
        actions = []
        
        for idx, agent in enumerate(env.agents):
            obs_agent = obs[idx]
            mem_agent = memories[agent].get_memory_features()
            valid_mask = infos[agent]["masks"]
            print(f"Mask dtype: {valid_mask.dtype}")
            
            # Convert to tensor
            obs_tensor = torch.from_numpy(obs_agent).float().unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(valid_mask).bool().unsqueeze(0).to(device)
            mem_tensor = torch.from_numpy(mem_agent).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_logits, value = model(obs_tensor, mem_tensor)
                
                # Use build_action_distribution to sample action
                dist = build_action_distribution(action_logits, valid_mask, 24)
                action_idx = dist.sample()
                
                # Check if sampled action is valid in the mask tensor
                # Reconstruct mask as done in update()
                B = 1
                grid = 24
                valid_dirs = mask_tensor.reshape(B, grid * grid, 4)
                valid_moves = valid_dirs.repeat_interleave(2, dim=2)  # (B, cells, 8)
                valid_moves = valid_moves.view(B, -1)  # (B, cells*8)
                
                # Check if action_idx corresponds to a valid move
                if action_idx.item() == 0:
                    # Pass
                    pass
                else:
                    move_idx = action_idx.item() - 1
                    is_valid = valid_moves[0, move_idx].item()
                    if not is_valid:
                        print(f"CRITICAL ERROR: Sampled action {action_idx.item()} is INVALID in mask tensor!")
                        print(f"  Mask at index {move_idx}: {valid_moves[0, move_idx]}")
                        # Decode to see where it is
                        split = move_idx % 2
                        direction = (move_idx // 2) % 4
                        cell_idx = (move_idx // 8)
                        row = cell_idx // grid
                        col = cell_idx % grid
                        print(f"  Action details: r={row}, c={col}, d={direction}, s={split}")
                        print(f"  Mask in numpy: {valid_mask[row, col, direction]}")
                        print(f"  Mask in tensor: {mask_tensor[0, row, col, direction]}")

                # Decode action for env
                if action_idx.item() == 0:
                    actions.append(Action(True))
                else:
                    move_idx = action_idx.item() - 1
                    split = move_idx % 2
                    direction = (move_idx // 2) % 4
                    cell_idx = (move_idx // 8)
                    row = cell_idx // grid
                    col = cell_idx % grid
                    actions.append(Action(False, row, col, direction, bool(split)))

            # Check if mask changes after step
            mask_before = valid_mask.copy()
            
            # ... (action selection) ...
            
        next_obs, _, terminated, truncated, next_infos = env.step(actions)
        
        # Check if infos mask changed
        for idx, agent in enumerate(env.agents):
            mask_after = infos[agent]["masks"]
            if not np.array_equal(mask_before, mask_after):
                print(f"CRITICAL: Mask for {agent} CHANGED after env.step!")
                # Check where it changed
                diff = mask_before != mask_after
                print(f"  Changed at {np.sum(diff)} positions.")
        
        if terminated or truncated:
            obs, infos = env.reset()
        else:
            obs, infos = next_obs, next_infos

if __name__ == "__main__":
    debug_masks()

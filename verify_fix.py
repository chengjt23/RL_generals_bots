import torch
import numpy as np
from torch.distributions import Categorical
import yaml
from generals.core.action import Action
from env.gymnasium_generals import GymnasiumGenerals
from model.memory import MemoryAugmentation
from model.network import SOTANetwork
from train_ppo import build_action_distribution, decode_action_index, obs_tensor_to_dict, MEMORY_CHANNELS

def verify_fix():
    # 1. Setup
    print("Setting up environment and model...")
    config_path = "config_ppo.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") 
    grid_size = 24
    
    env = GymnasiumGenerals(
        agents=["player_0", "player_1"],
        pad_observations_to=grid_size,
        render_mode=None,
    )
    
    model = SOTANetwork(
        obs_channels=15,
        memory_channels=MEMORY_CHANNELS,
        grid_size=grid_size,
        base_channels=64,
    ).to(device)
    
    # Load checkpoint
    ckpt_path = "/root/shared-nvme/oyx_new/RL_generals_bots/model/epoch_37_loss_1.7065.pt"
    print(f"Loading checkpoint from {ckpt_path}...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    
    # Initialize memory
    memory = MemoryAugmentation((grid_size, grid_size))
    
    # 2. Rollout Loop
    print("\n--- Performing Rollout Loop (until non-pass action) ---")
    obs, infos = env.reset()
    
    found_action = False
    steps = 0
    
    while not found_action and steps < 100:
        obs_agent = obs[0]
        mem_agent = memory.get_memory_features()
        valid_mask = infos["player_0"]["masks"]
        
        # Prepare tensors
        obs_t = torch.from_numpy(obs_agent).float().unsqueeze(0).to(device)
        mem_t = torch.from_numpy(mem_agent).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(valid_mask).bool().to(device)
        
        # Run model in EVAL mode (as in rollout)
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(obs_t, mem_t)
            dist = build_action_distribution(policy_logits, valid_mask, grid_size)
            action_idx = dist.sample()
            old_logprob = dist.log_prob(action_idx)
        
        if action_idx.item() != 0:
            print(f"Found non-pass action at step {steps}!")
            found_action = True
            break
            
        # Step env
        to_pass, row, col, direction, split = decode_action_index(action_idx.item(), grid_size)
        act = Action(to_pass=to_pass, row=row, col=col, direction=direction, to_split=bool(split))
        action_array = np.array([int(to_pass), row, col, direction, split], dtype=np.int8)
        
        # Dummy actions for others
        actions = [act]
        for _ in range(len(env.agents) - 1):
            actions.append(Action(to_pass=True, row=0, col=0, direction=0, to_split=False))
            
        next_obs, _, terminated, truncated, next_infos = env.step(actions)
        
        # Update memory
        obs_dict = obs_tensor_to_dict(next_obs[0])
        memory.update(obs_dict, action_array)
        
        obs = next_obs
        infos = next_infos
        steps += 1

    if not found_action:
        print("Could not find non-pass action in 100 steps. Using last step.")

    print(f"Action Index: {action_idx.item()}")
    print(f"Old Logprob (Rollout): {old_logprob.item()}")
    
    # 3. Verify Fix: Update Step in EVAL Mode
    print("\n--- Verifying Fix: Update Step in EVAL Mode ---")
    
    # Ensure model is in eval mode (this is the fix)
    model.eval()
    
    # We need gradients now
    policy_logits_new, _ = model(obs_t, mem_t)
    
    # Rebuild distribution logic from update()
    B = policy_logits_new.shape[0]
    pass_logits = policy_logits_new[:, 0, 0, 0]
    move_logits = policy_logits_new[:, 1:9]
    move_logits = move_logits.permute(0, 2, 3, 1).reshape(B, grid_size * grid_size, 8)
    
    valid_dirs = mask_t.reshape(B, grid_size * grid_size, 4)
    valid_moves = valid_dirs.repeat_interleave(2, dim=2)
    
    masked_logits = move_logits.masked_fill(~valid_moves, float("-inf"))
    flat_logits = torch.cat([pass_logits.unsqueeze(1), masked_logits.view(B, -1)], dim=1)
    
    dist_new = Categorical(logits=flat_logits)
    new_logprob = dist_new.log_prob(action_idx)
    
    print(f"New Logprob (Eval): {new_logprob.item()}")
    diff = abs(old_logprob.item() - new_logprob.item())
    print(f"Diff: {diff}")
    
    ratio = torch.exp(new_logprob - old_logprob)
    print(f"Ratio: {ratio.item()}")
    
    if diff < 1e-5:
        print("SUCCESS: Logprobs match! Ratio is ~1.0.")
    else:
        print("FAILURE: Logprobs mismatch!")

    # 4. Verify Gradients
    print("\n--- Verifying Gradients in EVAL Mode ---")
    loss = -new_logprob # Dummy loss
    loss.backward()
    
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            # print(f"Gradient found for {name}")
            break
            
    if has_grads:
        print("SUCCESS: Gradients are computed even in eval mode.")
    else:
        print("FAILURE: No gradients computed!")

if __name__ == "__main__":
    verify_fix()

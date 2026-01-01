
import torch
import numpy as np
from model.network import SOTANetwork
from model.memory import MemoryAugmentation
from env.gymnasium_generals import GymnasiumGenerals

def check_value_initialization():
    ckpt_path = "/root/shared-nvme/oyx_new/RL_generals_bots/model/epoch_37_loss_1.7065.pt"
    device = "cpu"
    
    # Load model
    model = SOTANetwork(obs_channels=15, memory_channels=13, grid_size=24, base_channels=64).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    # Filter and load weights
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    
    print(f"Loaded {len(filtered)}/{len(model_state)} weights.")
    
    # Check if value head weights were loaded
    value_head_keys = [k for k in model_state.keys() if "value_head" in k]
    loaded_value_keys = [k for k in filtered.keys() if "value_head" in k]
    print(f"Value head keys loaded: {len(loaded_value_keys)}/{len(value_head_keys)}")
    
    # Run a dummy observation
    env = GymnasiumGenerals(agents=["player_0", "player_1"])
    obs, infos = env.reset()
    obs_t = torch.from_numpy(obs[0]).float().unsqueeze(0).to(device)
    mem_t = torch.zeros(1, 13, 24, 24).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(obs_t, mem_t)
        
    print(f"Initial Value Prediction: {value.item()}")
    
    # If value head was NOT loaded, it's random.
    if len(loaded_value_keys) == 0:
        print("Value head was NOT loaded from checkpoint (Random Initialization).")
    else:
        print("Value head WAS loaded from checkpoint.")

if __name__ == "__main__":
    check_value_initialization()

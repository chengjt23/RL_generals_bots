
import torch
from model.network import SOTANetwork
from model.memory import MemoryAugmentation

def inspect_checkpoint():
    ckpt_path = "/root/shared-nvme/oyx_new/RL_generals_bots/model/epoch_37_loss_1.7065.pt"
    device = "cpu"
    
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    
    print("Checkpoint keys:")
    for k in state.keys():
        print(k)
        
    # Check if value head weights exist
    # Assuming SOTANetwork has a value head named 'value_head' or similar.
    # I need to check SOTANetwork definition.

if __name__ == "__main__":
    inspect_checkpoint()

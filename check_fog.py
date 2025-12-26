import numpy as np
from generals.envs import PettingZooGenerals
from generals.core.action import Action

def check_fog():
    print("Initializing PettingZooGenerals...")
    env = PettingZooGenerals(agents=["Agent1", "Agent2"], render_mode=None)
    
    print("Resetting environment...")
    obs_dict, info = env.reset()
    
    agent1_obs = obs_dict["Agent1"]
    tensor = agent1_obs.as_tensor()
    
    generals = tensor[1]
    fog_cells = tensor[7]
    
    print(f"Fog cells count: {np.sum(fog_cells)}")
    print(f"Generals count: {np.sum(generals)}")
    
    # Check for generals in fog
    generals_in_fog = generals * fog_cells
    count_generals_in_fog = np.sum(generals_in_fog > 0)
    
    print(f"Generals in fog (non-zero count): {count_generals_in_fog}")
    
    if count_generals_in_fog > 0:
        print("WARNING: Generals detected in fog! Agent has map hack.")
    else:
        print("No generals detected in fog.")
        
    # Also check if we can see opponent general
    # Usually there are 2 generals. If we see 2, and one is opponent...
    # But we only see our own general initially.
    if np.sum(generals) > 1:
        print("WARNING: More than 1 general visible initially! Might be seeing opponent.")
    else:
        print("Only 1 general visible initially (expected).")

    env.close()

if __name__ == "__main__":
    check_fog()

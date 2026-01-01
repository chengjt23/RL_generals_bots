import yaml
import torch
import numpy as np
from train_ppo import PPORunner

def verify_real_training():
    print("--- Verifying Fix using PPORunner from train_ppo.py ---")
    
    # Load config to disable swanlab for this test
    with open("config_ppo.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Disable swanlab to avoid creating a new run
    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["use_swanlab"] = False
    
    with open("config_test.yaml", "w") as f:
        yaml.dump(config, f)
        
    print("Initializing Runner...")
    runner = PPORunner("config_test.yaml")
    
    print("Collecting Rollout (this uses the real env and model)...")
    obs, infos = runner.env.reset()
    runner._reset_memories()
    
    # Collect one full rollout
    obs, infos, global_step = runner.collect_rollout(obs, infos, 0)
    
    print("Computing Advantages...")
    last_value = runner._compute_last_value(obs, infos)
    advantages, returns = runner.buffer.compute_advantages(last_value, runner.hyper.gamma, runner.hyper.gae_lambda)
    
    # Normalize advantages as in train()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    print("Running Update (this uses the fixed update method)...")
    metrics = runner.update(advantages, returns)
    
    print("\n--- Results ---")
    print(f"Clip Rate: {metrics['clip_rate']:.4f}")
    print(f"Ratio Mean: {metrics['ratio_mean']:.4f}")
    print(f"Approx KL: {metrics['approx_kl']:.4f}")
    
    if metrics['clip_rate'] < 0.05 and 0.95 < metrics['ratio_mean'] < 1.05:
        print("\nSUCCESS: The fix is working correctly in the real training loop.")
        print("Ratio is close to 1.0 and clip rate is low.")
    else:
        print("\nFAILURE: Metrics still indicate instability.")

if __name__ == "__main__":
    verify_real_training()

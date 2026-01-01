
import torch
import numpy as np
import yaml
from train_ppo import PPORunner
import sys

def verify_training():
    print("Starting training verification...")
    
    # Load config
    config_path = "config_ppo.yaml"
    
    # Initialize Runner
    runner = PPORunner(config_path)
    
    # Override hyperparameters for quick check
    # Run for ~80 updates (approx 20,000 steps) to verify stability over a longer period
    runner.hyper.total_env_steps = 20480 
    runner.hyper.rollout_length = 256
    runner.log_interval = 1
    
    # Capture metrics
    metrics_history = []
    
    original_update = runner.update
    
    def captured_update(advantages, returns):
        metrics = original_update(advantages, returns)
        metrics_history.append(metrics)
        
        # Print every 5 updates to avoid spam
        if len(metrics_history) % 5 == 0 or len(metrics_history) == 1:
            print(f"Update {len(metrics_history)}: "
                  f"Entropy={metrics['entropy']:.4f}, "
                  f"Policy Loss={metrics['policy_loss']:.4f}, "
                  f"Value Loss={metrics['value_loss']:.4f}, "
                  f"KL={metrics['approx_kl']:.4f}, "
                  f"Clip Rate={metrics['clip_rate']:.4f}")
        return metrics
    
    runner.update = captured_update
    
    # Run training
    try:
        runner.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Analyze results
    if not metrics_history:
        print("No updates performed.")
        return
    
    initial_entropy = metrics_history[0]['entropy']
    final_entropy = metrics_history[-1]['entropy']
    max_kl = max(m['approx_kl'] for m in metrics_history)
    
    print("\n--- Analysis ---")
    print(f"Initial Entropy: {initial_entropy:.4f}")
    print(f"Final Entropy: {final_entropy:.4f}")
    print(f"Max KL: {max_kl:.4f}")
    
    # Checks
    if final_entropy > 3.0:
        print("❌ FAILURE: Entropy is too high (> 3.0). Model is random.")
    elif final_entropy > initial_entropy + 0.5:
        print("⚠️ WARNING: Entropy is rising significantly.")
    else:
        print("✅ Entropy looks stable.")
        
    if max_kl > 0.2:
        print("❌ FAILURE: KL Divergence is too high (> 0.2). Policy is changing too fast.")
    else:
        print("✅ KL Divergence is within safe limits.")

if __name__ == "__main__":
    verify_training()

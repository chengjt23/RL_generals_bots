import torch
from tqdm import tqdm
from data.dataloader import GeneralsReplayDataset
from agents.network import SOTANetwork

print("=" * 60)
print("Testing Behavior Cloning Setup")
print("=" * 60)

print("\n1. Testing Data Loader...")
try:
    dataset = GeneralsReplayDataset(
        data_dir="data/generals_io_replays",
        grid_size=24,
        max_replays=10,
        min_stars=70,
        max_turns=500,
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) > 0:
        obs, memory, action, player_idx = dataset[0]
        print(f"   ✓ Sample shape: obs={obs.shape}, memory={memory.shape}, action={action.shape}")
        print(f"   ✓ Memory channels: {memory.shape[0]} (expected: 18)")
        print(f"   ✓ Observation channels: {obs.shape[0]}")
        
        # Verify memory content
        memory_np = memory.numpy()
        non_zero_channels = (memory_np != 0).any(axis=(1, 2)).sum()
        print(f"   ✓ Non-zero memory channels: {non_zero_channels}")
        
        print(f"   Testing batch loading...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_count = 0
        for batch_obs, batch_memory, batch_actions, batch_player_idx in tqdm(loader, desc="   Loading batches", total=min(5, len(loader)), leave=False):
            batch_count += 1
            if batch_count >= 5:
                break
        print(f"   ✓ Batch loading works: batch_obs={batch_obs.shape}, batch_memory={batch_memory.shape}")
    else:
        print("   ✗ No samples found in dataset!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing Model...")
try:
    model = SOTANetwork(
        obs_channels=15,
        memory_channels=18,
        grid_size=24,
        base_channels=32,
    )
    print(f"   ✓ Model created with memory augmentation")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    dummy_obs = torch.randn(2, 1, 15, 24, 24)
    dummy_memory = torch.randn(2, 1, 18, 24, 24)
    policy, value, _ = model(dummy_obs, dummy_memory)
    print(f"   ✓ Forward pass with memory: policy={policy.shape}, value={value.shape}")
    
    # Test without memory (should create zeros internally)
    policy_no_mem, value_no_mem, _ = model(dummy_obs, None)
    print(f"   ✓ Forward pass without memory: policy={policy_no_mem.shape}, value={value_no_mem.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Training Components...")
try:
    import yaml
    with open("configs/config_base.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Config loaded")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    print(f"   ✓ Optimizer created")
    
    print(f"   ✓ Batch size: {config['training']['batch_size']}")
    print(f"   ✓ Learning rate: {config['training']['learning_rate']}")
    print(f"   ✓ Epochs: {config['training']['num_epochs']}")
    print(f"   ✓ Device: {config['training']['device']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✓ Setup test complete!")
print("=" * 60)
print("\nReady to start training with:")
print("  cd RL")
print("  python train_bc.py --config configs/config_base.yaml")


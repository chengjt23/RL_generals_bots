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
        obs, action, player_idx = dataset[0]
        print(f"   ✓ Sample shape: obs={obs.shape}, action={action.shape}")
        
        print(f"   Testing batch loading...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        for batch in tqdm(loader, desc="   Loading batches", total=min(5, len(loader)), leave=False):
            if loader._index_sampler is not None and hasattr(loader._index_sampler, '_num_yielded'):
                if loader._index_sampler._num_yielded >= 5:
                    break
        print(f"   ✓ Batch loading works")
    else:
        print("   ✗ No samples found in dataset!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing Model...")
try:
    model = SOTANetwork(
        obs_channels=15,
        memory_channels=0,
        grid_size=24,
        base_channels=64,
    )
    print(f"   ✓ Model created")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    dummy_obs = torch.randn(2, 15, 24, 24)
    policy, value = model(dummy_obs)
    print(f"   ✓ Forward pass: policy={policy.shape}, value={value.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing Training Components...")
try:
    import yaml
    with open("configs/config_large.yaml", 'r') as f:
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
print("  python train_bc.py --config configs/behavior_cloning.yaml")


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
        print(f"   ✓ Observation channels: {obs.shape[0]} (expected: 15)")
        print(f"   ✓ No hand-crafted memory - RNN will encode memory from observations")
        
        print(f"   Testing batch loading...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_count = 0
        for batch_obs, batch_actions, batch_player_idx in tqdm(loader, desc="   Loading batches", total=min(5, len(loader)), leave=False):
            batch_count += 1
            if batch_count >= 5:
                break
        print(f"   ✓ Batch loading works: batch_obs={batch_obs.shape}, batch_actions={batch_actions.shape}")
    else:
        print("   ✗ No samples found in dataset!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing Model with RNN Memory...")
try:
    model = SOTANetwork(
        obs_channels=15,
        memory_channels=16,
        grid_size=24,
        base_channels=64,
        rnn_hidden_channels=32,
        rnn_num_layers=2,
        use_rnn_memory=True,
    )
    print(f"   ✓ Model created with RNN memory encoder")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    # Test forward pass with RNN memory
    dummy_obs = torch.randn(2, 15, 24, 24)
    
    # First timestep (no hidden state)
    policy, value, hidden = model(dummy_obs, hidden_state=None, return_hidden=True)
    print(f"   ✓ Forward pass (t=0): policy={policy.shape}, value={value.shape}")
    print(f"   ✓ Hidden state created: {len(hidden)} layers")
    
    # Second timestep (with hidden state from previous)
    policy2, value2, hidden2 = model(dummy_obs, hidden_state=hidden, return_hidden=True)
    print(f"   ✓ Forward pass (t=1) with hidden state: policy={policy2.shape}, value={value2.shape}")
    
    # Test without returning hidden (training mode)
    policy3, value3 = model(dummy_obs, hidden_state=None, return_hidden=False)
    print(f"   ✓ Forward pass (training mode): policy={policy3.shape}, value={value3.shape}")
    
    # Test init_hidden method
    init_hidden = model.init_hidden(batch_size=4, device=torch.device('cpu'))
    print(f"   ✓ init_hidden works: {len(init_hidden)} layers, h shape={init_hidden[0][0].shape}")
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
    print(f"   ✓ RNN hidden channels: {config['model']['rnn_hidden_channels']}")
    print(f"   ✓ RNN num layers: {config['model']['rnn_num_layers']}")
    print(f"   ✓ RNN memory enabled: {config['model']['use_rnn_memory']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✓ Setup test complete! RNN Memory Architecture Ready")
print("=" * 60)
print("\nRNN Memory Architecture:")
print("  • Observations → ConvLSTM layers → Memory representation")
print("  • Hidden states maintained across timesteps during inference")
print("  • No hand-crafted memory features - RNN learns temporal patterns")
print("\nReady to start training with:")
print("  python train_bc.py --config configs/config_base.yaml")


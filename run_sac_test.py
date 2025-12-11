import torch
from agents.sac_agent import SACAgent
from agents.replay_buffer import ReplayBuffer

def test_sac_networks():
    print("Testing SAC networks initialization...")
    agent = SACAgent(
        id="SAC",
        grid_size=24,
        device="cpu",
        bc_model_path=None,
        memory_channels=18
    )
    print("✓ SAC Agent initialized successfully")
    
    batch_size = 4
    obs = torch.randn(batch_size, 15, 24, 24)
    memory = torch.randn(batch_size, 18, 24, 24)
    
    print("Testing Actor network...")
    policy_logits = agent.actor(obs, memory)
    print(f"✓ Actor output shape: {policy_logits.shape}")
    
    print("Testing Critic networks...")
    q1 = agent.critic_1(obs, memory)
    q2 = agent.critic_2(obs, memory)
    print(f"✓ Critic 1 output shape: {q1.shape}")
    print(f"✓ Critic 2 output shape: {q2.shape}")
    
    print("\nTesting Replay Buffer...")
    buffer = ReplayBuffer(
        capacity=1000,
        obs_shape=(15, 24, 24),
        memory_shape=(18, 24, 24),
        device="cpu"
    )
    
    for i in range(10):
        buffer.store(
            obs=torch.randn(15, 24, 24).numpy(),
            memory=torch.randn(18, 24, 24).numpy(),
            action=torch.randint(0, 24, (4,)).numpy(),
            reward=0.5,
            next_obs=torch.randn(15, 24, 24).numpy(),
            next_memory=torch.randn(18, 24, 24).numpy(),
            done=0.0
        )
    
    print(f"✓ Buffer size: {len(buffer)}")
    
    batch = buffer.sample(4)
    print(f"✓ Sampled batch with {len(batch)} keys")
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_sac_networks()


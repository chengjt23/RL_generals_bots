import sys
sys.path.append('..')

import torch
from agents.sac_agent import SACAgent
from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent


def run_sac_game(checkpoint_path=None, bc_model_path=None, num_games=5):
    agent = SACAgent(
        id="SAC",
        grid_size=24,
        device="cuda" if torch.cuda.is_available() else "cpu",
        bc_model_path=bc_model_path,
        memory_channels=18
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"Loaded SAC checkpoint from {checkpoint_path}")
    
    agent.actor.eval()
    
    env = PettingZooGenerals(agents=["SAC", "RandomAgent"], render_mode=None)
    opponent = RandomAgent()
    
    wins = 0
    total_rewards = 0
    
    for game_num in range(num_games):
        obs_dict, info = env.reset()
        agent.reset()
        terminated = truncated = False
        episode_reward = 0
        step = 0
        
        print(f"\nGame {game_num + 1}/{num_games}")
        
        while not (terminated or truncated):
            sac_action = agent.act(obs_dict["SAC"], deterministic=True)
            opponent_action = opponent.act(obs_dict["RandomAgent"])
            actions_dict = {"SAC": sac_action, "RandomAgent": opponent_action}
            
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            episode_reward += rewards_dict["SAC"]
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: Reward = {episode_reward:.2f}")
        
        if rewards_dict["SAC"] > rewards_dict["RandomAgent"]:
            wins += 1
            print(f"  Result: WIN")
        else:
            print(f"  Result: LOSS")
        
        total_rewards += episode_reward
        print(f"  Total Reward: {episode_reward:.2f}")
    
    win_rate = wins / num_games
    avg_reward = total_rewards / num_games
    
    print(f"\n{'='*50}")
    print(f"Results over {num_games} games:")
    print(f"Win Rate: {win_rate*100:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'='*50}")
    
    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to SAC checkpoint')
    parser.add_argument('--bc_model', type=str, default=None, help='Path to BC pretrained model')
    parser.add_argument('--num_games', type=int, default=5, help='Number of games to play')
    args = parser.parse_args()
    
    run_sac_game(
        checkpoint_path=args.checkpoint,
        bc_model_path=args.bc_model,
        num_games=args.num_games
    )

